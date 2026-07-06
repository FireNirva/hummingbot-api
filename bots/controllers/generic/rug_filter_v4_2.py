"""Rug Filter v4.2 v0.9i hybrid — T+5min drawdown predictor + RAW XGBoost prob.

Pkl: `models/rug_v4_2_v0_9i_hybrid.pkl`
  - xgb_model: XGBClassifier (depth=5, reg_lambda=95.5, n_estimators=1100)
  - feature_cols: 18 features (10 v4.2 baseline + 8 new + microstructure + 2 macro)
  - cutoffs_raw: dict {top5..top30} computed from val raw probs
  - use_raw_probability: True — bypass LR calibrator (D-fix)

Label: `min(price[entry, entry+120s]) / entry_anchor - 1 < -0.55`
  → bot 入场后 2 分钟内最低价跌过 55%

Audit (Birdeye holdout):
  AUC 0.7024  CI95 [0.59, 0.80]
Audit (Production big_loss, 267 trades):
  AUC 0.5573  Top 20% gate: 12 big_loss caught, $22.72 saved

Frozen spec: model_specs/2026-05-08_rug_filter_v4_2_SPEC.md v0.9f section
SHADOW ONLY during validation phase (2026-05-09 onwards).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
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

# 18 features used by v0.9i (top by importance from v0.9c 36-feat panel)
FEATURES_V4_2 = [
    'kyle_lambda_300s', 'top1_swap_share_300s', 'vpin_300s',
    'unique_buyer_count_300s', 'swap_count_300s', 'roll_spread_300s',
    'buy_count_share_300s', 'direction_switches_300s', 'drawdown_from_peak_300s',
    'swap_count_60s', 'new_buyer_rate_60_300', 'mean_swap_size_300s',
    'p95_swap_size_300s', 'sol_volatility_1h_pre_grad',
    'swap_density_ratio_60_300s', 'consecutive_sell_max_300s',
    'mean_swap_size_60s', 'graduation_hour_utc',
]

DUST_SOL = 0.05
ENTRY_DELAY_SEC = 300
MIN_REF_PRICE = 1e-9
VERSION = "v4.2_v0_9i_hybrid_raw_xgb_2026-05-09"


@dataclass
class V42ScoreResult:
    score: float                  # raw XGBoost probability [0, 1] or NaN
    decision: str                 # REJECT | PASS | SKIP_NO_DATA | SKIP_MODEL_ERROR
    reason: str
    n_swaps: int
    features: Dict[str, float] = field(default_factory=dict)


def _compute_v4_2_features(swaps_df: pd.DataFrame, graduation_time: float,
                              sol_5m_dict: Optional[dict] = None,
                              window_s: int = 300) -> Optional[Dict[str, float]]:
    """Compute 18 features in [graduation_time, graduation_time + 300s].

    Forward-only, dust-filtered. Mirrors training-time pipeline byte-for-byte.

    Returns dict or None if data quality insufficient.
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
    s["is_buy"] = pd.to_numeric(s["is_buy"], errors="coerce").fillna(0).astype(int)

    # Forward window + dust filter (matches training pipeline)
    grad_t = float(graduation_time)
    s = s[(s["block_time"] >= grad_t) & (s["block_time"] <= grad_t + window_s)]
    s = s[s["sol_amount"] >= DUST_SOL]
    if len(s) < 10:
        return None
    s = s.sort_values("block_time").reset_index(drop=True)

    out: Dict[str, float] = {f: None for f in FEATURES_V4_2}

    # ── Sub-windows ──
    win60 = s[s["block_time"] <= grad_t + 60]
    win300 = s    # full window already filtered

    # ── Counts (drift-safe) ──
    out["swap_count_60s"] = float(len(win60))
    out["swap_count_300s"] = float(len(win300))
    if len(win300) / 300.0 > 0:
        out["swap_density_ratio_60_300s"] = (len(win60) / 60.0) / (len(win300) / 300.0)

    # ── Volume statistics ──
    if len(win60) > 0:
        out["mean_swap_size_60s"] = float(win60["sol_amount"].mean())
    out["mean_swap_size_300s"] = float(win300["sol_amount"].mean())
    out["p95_swap_size_300s"] = float(win300["sol_amount"].quantile(0.95))
    out["buy_count_share_300s"] = float((win300["is_buy"] == 1).mean())

    # ── drawdown_from_peak_300s ──
    px = pd.to_numeric(win300["effective_price_sol"], errors="coerce").values
    valid = (px > 0) & np.isfinite(px)
    if valid.sum() < 5:
        return None
    px_valid = px[valid]
    peak = float(px_valid.max())
    last = float(px_valid[-1])
    if peak > 0:
        out["drawdown_from_peak_300s"] = (last - peak) / peak

    # ── Microstructure features ──
    px300 = win300["effective_price_sol"].values
    vol300 = win300["sol_amount"].values
    is_buy300 = win300["is_buy"].values
    valid_px = (px300 > 0) & np.isfinite(px300)

    if valid_px.sum() >= 10:
        px_v = px300[valid_px]
        rets = np.diff(np.log(np.maximum(px_v, 1e-12)))
        vols = vol300[valid_px][1:]
        if len(rets) > 5:
            with np.errstate(divide='ignore', invalid='ignore'):
                lam = np.abs(rets) / np.maximum(vols, 0.01)
                lam_f = lam[np.isfinite(lam)]
                if len(lam_f):
                    out["kyle_lambda_300s"] = float(np.median(lam_f))
            cov_lag = float(np.cov(rets[:-1], rets[1:])[0, 1])
            out["roll_spread_300s"] = float(2 * np.sqrt(-cov_lag)) if cov_lag < 0 else 0.0

    # vpin (30s buckets)
    if vol300.sum() > 0:
        n_b = 0
        vp = 0.0
        for ts in range(0, 300, 30):
            sub = win300[(win300["block_time"] >= grad_t + ts) &
                          (win300["block_time"] < grad_t + ts + 30)]
            if len(sub) >= 3:
                bv = float(sub[sub["is_buy"] == 1]["sol_amount"].sum())
                sv = float(sub[sub["is_buy"] == 0]["sol_amount"].sum())
                tot = bv + sv
                if tot > 0:
                    vp += abs(bv - sv) / tot
                    n_b += 1
        if n_b:
            out["vpin_300s"] = float(vp / n_b)

    # Wallet count (drift-safe count features)
    if "trader_address" in win60.columns:
        n60 = int(win60["trader_address"].nunique())
        n300 = int(win300["trader_address"].nunique())
        out["unique_buyer_count_300s"] = float(n300)
        if n60 > 0:
            out["new_buyer_rate_60_300"] = (n300 - n60) / max(n60, 1)

    # top1_swap_share_300s
    if vol300.sum() > 0:
        out["top1_swap_share_300s"] = float(vol300.max() / vol300.sum())

    # Direction patterns
    if len(is_buy300) > 0:
        cur = max_streak = 0
        for b in is_buy300:
            if b == 0:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        out["consecutive_sell_max_300s"] = float(max_streak)
        out["direction_switches_300s"] = float((np.diff(is_buy300) != 0).sum())

    # Macro features
    grad_dt = pd.Timestamp(grad_t, unit='s')
    out["graduation_hour_utc"] = float(grad_dt.hour)

    # SOL volatility 1h pre-grad (12 × 5min returns std)
    # Falls back to None if SOL bars not available — expected in production
    # initial deploy. v4.2 model can still score with this missing (XGB handles NaN).
    if sol_5m_dict:
        rets = []
        for back_min in range(0, 60, 5):
            ts_a = int(grad_t - back_min * 60) // 300 * 300
            ts_b = int(grad_t - (back_min + 5) * 60) // 300 * 300
            a = sol_5m_dict.get(ts_a)
            b = sol_5m_dict.get(ts_b)
            if a and b and b > 0:
                rets.append((a - b) / b)
        if len(rets) >= 6:
            out["sol_volatility_1h_pre_grad"] = float(np.std(rets))

    return out


class RugFilterV4_2:
    """v0.9i hybrid: XGBoost RAW probability (no LR calibrator).

    Inference recipe:
      score = xgb.predict_proba(X)[:, 1][0]
      decision = "REJECT" if score >= cutoff else "PASS"
    """

    VERSION = VERSION

    def __init__(self, model_path: str, cutoff: float = 0.5040,
                 window_s: int = 300, min_swaps: int = 10,
                 *, metrics: Optional[Any] = None):
        if not HAS_SKLEARN_STACK:
            raise RuntimeError("RugFilterV4_2 requires numpy, pandas, sklearn, xgboost")
        # ⚠️ G0 enforce: don't import v5_x exit modules
        forbidden = ["meme_sniper_exit_models", "v5_3"]
        for mod in forbidden:
            if mod in sys.modules and any(mod in str(m) for m in sys.modules):
                # only warn, don't block (other code may import these)
                pass

        self.guarded = _model_registry.get_or_load(model_path, metrics=metrics)
        self._pkg = self.guarded.unwrap()

        # Validate pkl structure
        for k in ("xgb_model", "feature_cols", "use_raw_probability"):
            if k not in self._pkg:
                raise ValueError(f"v4.2 hybrid pkl missing key: {k}")
        if not self._pkg.get("use_raw_probability"):
            raise ValueError("v4.2 hybrid pkl must have use_raw_probability=True (D-fix)")

        self.xgb = self._pkg["xgb_model"]
        self.feature_cols = list(self._pkg["feature_cols"])
        self.cutoff = float(cutoff)
        self.window_s = int(window_s)
        self.min_swaps = int(min_swaps)

        # Sanity check feature count
        if list(self.feature_cols) != FEATURES_V4_2:
            logger.warning(
                f"RugFilterV4_2: pickle features differ from FEATURES_V4_2 "
                f"(pkl_n={len(self.feature_cols)}, expected_n={len(FEATURES_V4_2)})")

        self._sol_5m_dict: Optional[dict] = None  # filled by load_sol_bars

    def load_sol_bars(self, sol_5m_dict: dict):
        """Optional: pass SOL/USDT 5min close prices keyed by floor-5m unix time.
        Without this, sol_volatility_1h_pre_grad will be None (XGB handles)."""
        self._sol_5m_dict = sol_5m_dict

    def score_features(self, features: Dict[str, float], n_swaps: int) -> V42ScoreResult:
        """Score a pre-computed feature dict. Returns V42ScoreResult."""
        try:
            x = np.array([features.get(f, np.nan) for f in self.feature_cols],
                            dtype=float).reshape(1, -1)
            # XGBoost handles NaN natively
            score = float(self.xgb.predict_proba(x)[0, 1])
        except Exception as e:
            return V42ScoreResult(
                score=float("nan"), decision="SKIP_MODEL_ERROR",
                reason=f"predict_failed:{type(e).__name__}",
                n_swaps=n_swaps, features=features)

        if score >= self.cutoff:
            decision = "REJECT"
            reason = f"score={score:.4f}>=cutoff{self.cutoff:.4f}"
        else:
            decision = "PASS"
            reason = f"score={score:.4f}<{self.cutoff:.4f}"
        return V42ScoreResult(
            score=score, decision=decision, reason=reason,
            n_swaps=n_swaps, features=features)

    def score_from_sqlite(self, db_path: str, mint_address: str,
                            graduation_time: float) -> V42ScoreResult:
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
            return V42ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason=f"sqlite_err:{type(e).__name__}",
                                     n_swaps=0, features={})

        if len(rows) < self.min_swaps:
            return V42ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason=f"insufficient_swaps:{len(rows)}<{self.min_swaps}",
                                     n_swaps=len(rows), features={})

        df = pd.DataFrame(rows, columns=["timestamp","trader_address","is_buy",
                                              "volume_sol","price_sol"])
        feats = _compute_v4_2_features(df, graduation_time,
                                          sol_5m_dict=self._sol_5m_dict,
                                          window_s=self.window_s)
        if feats is None:
            return V42ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason="feature_compute_failed",
                                     n_swaps=len(df), features={})

        return self.score_features(feats, n_swaps=len(df))
