"""Rug Filter v4 — T+5min CTA-style ensemble (R1 single rule + R4 decision tree).

Loads ensemble pickle from `outputs/phase26/rug_v4_ensemble_balanced.pkl`:
  - R1: recovery_attempts_count_300s >= r1_threshold (binary)
  - R4: depth-3 sklearn DecisionTreeClassifier on 5 features (predict_proba)
  - Decision (C-balanced): (r1_flag) OR (r4_proba >= cutoff)

Scores graduations at T+5min using 5 forward-only features computed from
the bot's own swaps table.

Shadow-only during validation phase (2026-05-06 onwards).
Frozen spec: model_specs/2026-05-06_rug_filter_v4_CTA_FROZEN_SPEC.md
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from meme_sniper.L4_Signal_and_Model_Inference import ShadowGuard

try:  # production: controllers.generic.ms.models (Hummingbot loads as controllers.generic.*)
    from controllers.generic.ms.models import registry as _model_registry
except ImportError:  # in-container test channel / direct file import
    from ms.models import registry as _model_registry

try:
    import numpy as np
    import pandas as pd
    HAS_SKLEARN_STACK = True
except ImportError:
    HAS_SKLEARN_STACK = False

logger = logging.getLogger(__name__)

# Frozen 5-feature set from spec §4.2
FEATURES_V4 = [
    "peak_pnl_pct_300s",
    "recovery_attempts_count_300s",
    "hhi_top5_buyers_60s",
    "sell_sol_share_300s",
    "swap_size_cv_300s",
]

# Minimum reference price floor (matches 26c MIN_REF_PRICE).
# Tokens with median-of-5 ref price < this are SKIP_NO_DATA — degenerate
# liquidity / dust pricing produces unbounded peak_pnl_pct (e.g.
# ref=1e-12, peak=1 → peak_pnl=2.7e12 outlier observed in training).
MIN_REF_PRICE = 1e-9

VERSION = "v4.0.1-2026-05-06-cta-balanced-refguard"


@dataclass
class V4ScoreResult:
    r1_flag: int                  # 0 or 1
    r4_proba: float               # [0, 1] or NaN
    decision: str                 # REJECT | PASS | SKIP_NO_DATA | SKIP_MODEL_ERROR
    reason: str
    n_swaps: int
    features: Dict[str, float] = field(default_factory=dict)


def _hhi(values) -> float:
    a = np.array(values, dtype=float)
    total = a.sum()
    if total <= 0:
        return 0.0
    p = a / total
    return float((p ** 2).sum())


def _compute_5_features(swaps_df: pd.DataFrame, graduation_time: float,
                        window_s: int = 300) -> Optional[Dict[str, float]]:
    """Compute the 5 forward-only features in [graduation_time, graduation_time + 300s].

    Mirrors `scripts/26e_feature_audit.py` compute functions for these 5 cols.

    Expects swaps_df columns: timestamp (or block_time), trader_address,
    is_buy, volume_sol (or sol_amount), price_sol (or effective_price_sol).
    """
    if swaps_df is None or len(swaps_df) == 0:
        return None
    s = swaps_df.copy()
    if "block_time" not in s.columns and "timestamp" in s.columns:
        s = s.rename(columns={"timestamp": "block_time"})
    if "sol_amount" not in s.columns and "volume_sol" in s.columns:
        s = s.rename(columns={"volume_sol": "sol_amount"})
    if "effective_price_sol" not in s.columns and "price_sol" in s.columns:
        s = s.rename(columns={"price_sol": "effective_price_sol"})

    s["sol_amount"] = pd.to_numeric(s["sol_amount"], errors="coerce").fillna(0)
    s["block_time"] = pd.to_numeric(s["block_time"], errors="coerce")

    # Forward-only window [grad, grad + 300] + dust filter
    grad_t = float(graduation_time)
    s = s[(s["block_time"] >= grad_t) & (s["block_time"] <= grad_t + window_s)]
    s = s[s["sol_amount"] >= 0.05]
    if len(s) < 5:
        return None

    s = s.sort_values("block_time").reset_index(drop=True)

    out: Dict[str, float] = {}

    # ── peak_pnl_pct_300s + recovery_attempts_count_300s (path geometry) ──
    px = pd.to_numeric(s["effective_price_sol"], errors="coerce").values
    valid = (px > 0) & np.isfinite(px)
    if valid.sum() < 5:
        return None
    px_valid = px[valid]
    # Reference: median of first 5 valid prices (matches 26e P1-1 fix)
    ref_window = px_valid[: min(5, len(px_valid))]
    ref_window = ref_window[(ref_window > 0) & np.isfinite(ref_window)]
    if len(ref_window) < 3:
        return None
    ref = float(np.median(ref_window))
    # Audit fix 2026-05-06: reject degenerate ref prices (< 1e-9 SOL) that
    # produce unbounded peak_pnl_pct. Caller treats this as SKIP_NO_DATA.
    if ref <= 0 or ref < MIN_REF_PRICE:
        return None

    peak = float(px_valid.max())
    # Cap peak_pnl_pct at 100 (10000%) — anything beyond is data anomaly.
    # Training holdout 99th percentile is 7.65; 100 is well above any
    # realistic pump.
    raw_pnl = (peak - ref) / ref
    out["peak_pnl_pct_300s"] = min(raw_pnl, 100.0) if np.isfinite(raw_pnl) else 0.0

    # recovery_attempts_count_300s: # of higher-highs >5% above last_high
    last_high = float(px_valid[0])
    count = 0
    for v in px_valid[1:]:
        if v > last_high * 1.05:
            count += 1
            last_high = float(v)
    out["recovery_attempts_count_300s"] = float(count)

    # ── hhi_top5_buyers_60s (concentration) ──
    win60 = s[(s["block_time"] <= grad_t + 60) & (s["is_buy"] == 1)]
    if len(win60) > 0:
        by_trader = win60.groupby("trader_address")["sol_amount"].sum().sort_values(
            ascending=False)
        if len(by_trader) > 0 and by_trader.sum() > 0:
            shares = by_trader / by_trader.sum()
            top5 = shares.head(5)
            # 26e formula: hhi_top5 = sum(top5_shares^2) / 0.2
            out["hhi_top5_buyers_60s"] = float((top5 ** 2).sum() / 0.2)
        else:
            out["hhi_top5_buyers_60s"] = 0.0
    else:
        out["hhi_top5_buyers_60s"] = 0.0

    # ── sell_sol_share_300s ──
    total_sol = float(s["sol_amount"].sum())
    if total_sol > 0:
        out["sell_sol_share_300s"] = float(
            s.loc[s["is_buy"] == 0, "sol_amount"].sum() / total_sol)
    else:
        out["sell_sol_share_300s"] = 0.0

    # ── swap_size_cv_300s ──
    sizes = s["sol_amount"].values
    if len(sizes) > 1 and sizes.mean() > 0:
        out["swap_size_cv_300s"] = float(sizes.std() / sizes.mean())
    else:
        out["swap_size_cv_300s"] = 0.0

    return out


class RugFilterV4:
    """C-balanced ensemble (R1 binary OR R4 proba >= cutoff)."""

    VERSION = VERSION

    def __init__(self, model_path: str, cutoff: float = 0.65,
                 window_s: int = 300, min_swaps: int = 5,
                 *, metrics: Optional[Any] = None):
        if not HAS_SKLEARN_STACK:
            raise RuntimeError("RugFilterV4 requires numpy, pandas, sklearn")
        self.guarded = _model_registry.get_or_load(model_path, metrics=metrics)
        self._pkg = self.guarded.unwrap()
        self.cutoff = float(cutoff)
        self.window_s = int(window_s)
        self.min_swaps = int(min_swaps)

        # Validate pickle structure
        for k in ("r1_feature", "r1_threshold", "r4_features", "r4_tree"):
            if k not in self._pkg:
                raise ValueError(f"Ensemble pickle missing key: {k}")
        self.r1_feature: str = self._pkg["r1_feature"]
        self.r1_threshold: float = float(self._pkg["r1_threshold"])
        self.r4_features = list(self._pkg["r4_features"])
        self.r4_tree = self._pkg["r4_tree"]
        # Sanity: r4_features must equal FEATURES_V4 (frozen)
        if list(self.r4_features) != FEATURES_V4:
            logger.warning(
                f"RugFilterV4: pickle r4_features differs from FEATURES_V4 "
                f"(pickle={self.r4_features}, frozen={FEATURES_V4})")

    def score_features(self, features: Dict[str, float], n_swaps: int
                       ) -> V4ScoreResult:
        """Score a pre-computed feature dict. Returns V4ScoreResult."""
        try:
            x = np.array([features.get(f, 0.0) for f in self.r4_features],
                         dtype=float).reshape(1, -1)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            r4_proba = float(self.r4_tree.predict_proba(x)[0, 1])
        except Exception as e:
            return V4ScoreResult(
                r1_flag=0, r4_proba=float("nan"),
                decision="SKIP_MODEL_ERROR",
                reason=f"r4_predict_failed:{type(e).__name__}",
                n_swaps=n_swaps, features=features)

        r1_val = float(features.get(self.r1_feature, 0.0))
        r1_flag = int(r1_val >= self.r1_threshold)

        if r1_flag or r4_proba >= self.cutoff:
            decision = "REJECT"
            reason = f"r1={r1_flag} r4_proba={r4_proba:.3f}>=cutoff{self.cutoff:.2f}"
        else:
            decision = "PASS"
            reason = f"r1=0 r4_proba={r4_proba:.3f}<{self.cutoff:.2f}"
        return V4ScoreResult(
            r1_flag=r1_flag, r4_proba=r4_proba,
            decision=decision, reason=reason,
            n_swaps=n_swaps, features=features)

    def score_from_sqlite(self, db_path: str, mint_address: str,
                          graduation_time: float) -> V4ScoreResult:
        """Pull swaps from `swaps` table and score this mint at T+window_s."""
        try:
            conn = sqlite3.connect(db_path)
            try:
                grad_t = float(graduation_time)
                rows = conn.execute(
                    "SELECT timestamp, trader_address, is_buy, "
                    "volume_sol, price_sol FROM swaps "
                    "WHERE mint_address = ? "
                    "AND timestamp >= ? AND timestamp <= ?",
                    (mint_address, grad_t, grad_t + self.window_s)
                ).fetchall()
            finally:
                conn.close()
        except Exception as e:
            return V4ScoreResult(
                r1_flag=0, r4_proba=float("nan"),
                decision="SKIP_MODEL_ERROR",
                reason=f"db_read_failed:{type(e).__name__}",
                n_swaps=0, features={})

        n_swaps = len(rows)
        if n_swaps < self.min_swaps:
            return V4ScoreResult(
                r1_flag=0, r4_proba=float("nan"),
                decision="SKIP_NO_DATA",
                reason=f"n_swaps={n_swaps}<{self.min_swaps}",
                n_swaps=n_swaps, features={})

        df = pd.DataFrame(rows, columns=[
            "timestamp", "trader_address", "is_buy",
            "volume_sol", "price_sol"])
        try:
            features = _compute_5_features(df, grad_t, self.window_s)
        except Exception as e:
            return V4ScoreResult(
                r1_flag=0, r4_proba=float("nan"),
                decision="SKIP_MODEL_ERROR",
                reason=f"feature_compute_failed:{type(e).__name__}",
                n_swaps=n_swaps, features={})

        if features is None:
            return V4ScoreResult(
                r1_flag=0, r4_proba=float("nan"),
                decision="SKIP_NO_DATA",
                reason="features_none_after_filter",
                n_swaps=n_swaps, features={})

        return self.score_features(features, n_swaps)
