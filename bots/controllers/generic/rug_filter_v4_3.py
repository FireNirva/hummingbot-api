"""Rug Filter v4.3 — Cross-source-aligned retrain on Chainstack production data.

Spec: model_specs/2026-05-13_rug_filter_v4_3_SPEC.md

Pkl: `models/rug_v4_3_winner.pkl`
  - xgb_model: XGBClassifier
  - feature_cols: 24 features (v4.2's 18 + 6 NEW per §4)
  - cutoffs_raw: dict {top5..top30} computed from val raw probs
  - use_raw_probability: True

Label (§5.3): `min(price[entry, entry+120s]) / anchor - 1 ≤ -0.55`
  - anchor = median(price[entry, entry+10s]) if ≥3 swaps else first post-entry swap
  - entry_t = graduation_time + 300s

Features (§4): 11 Tier-1 + 3 Tier-2 + 1 Tier-3 (sandwich drawdown) + 7 Tier-4 microstructure + 2 macro.

NEW vs v4.2:
  - hc_hhi_300s, hc_top3_share_300s — top buyer concentration
  - ofi_300s — order flow imbalance
  - n_local_maxima_300s_sw — sandwich-confirmed local maxima count
  - total_vol_sol_300s, sell_share_300s — wave-stable totals
  - drawdown_from_peak_300s_sw — replaces raw drawdown_from_peak_300s

Spec gates G25-G28 enforce byte-parity with this module.

Constants:
  DUST_SOL = 0.05 (unchanged from v4.2)
  ENTRY_DELAY_SEC = 300 (unchanged)
  SANDWICH_WINDOW_SEC = 5.0
  SANDWICH_MIN_CONF = 2 (means ≥3 swaps incl. self per geyser_stream.py:855)
  SANDWICH_THRESHOLD = 0.80
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

# 22 features per spec §4 (v4.2's 18 + 6 NEW = 24, minus 2 dropped by G_micro)
# G_micro audit 2026-05-14 (Stage 1) dropped:
#   - kyle_lambda_300s (ρ=0.8418, just below 0.85 gate)
#   - roll_spread_300s (ρ=0.6049, severe cross-source drift)
# Output: outputs/phase14b_chainstack/tier4_alignment_summary_backfill_missing_2mo.csv
FEATURES_V4_3 = [
    # Tier 1 — wave-stable + cross-source ρ > 0.95 (11)
    'swap_count_60s', 'swap_count_300s', 'swap_density_ratio_60_300s',
    'mean_swap_size_60s', 'mean_swap_size_300s', 'p95_swap_size_300s',
    'total_vol_sol_300s', 'unique_buyers_300s',
    'hc_hhi_300s', 'hc_top3_share_300s', 'n_local_maxima_300s_sw',
    # Tier 2 — cross-source OK but wave-variable (3)
    'buy_count_share_300s', 'sell_share_300s', 'ofi_300s',
    # Tier 3 — sandwich-filtered replacement (1)
    'drawdown_from_peak_300s_sw',
    # Tier 4 — v0.9f microstructure (5/7 surviving G_micro)
    'vpin_300s',
    'top1_swap_share_300s', 'direction_switches_300s',
    'consecutive_sell_max_300s', 'new_buyer_rate_60_300',
    # Macro (2)
    'sol_volatility_1h_pre_grad', 'graduation_hour_utc',
]
assert len(FEATURES_V4_3) == 22, f"Expected 22 features (24 - 2 G_micro drops), got {len(FEATURES_V4_3)}"

DUST_SOL = 0.05
ENTRY_DELAY_SEC = 300
MIN_REF_PRICE = 1e-9
SANDWICH_WINDOW_SEC = 5.0
SANDWICH_MIN_CONF = 2     # "other swaps" — total in window = min_conf + 1 = 3
SANDWICH_THRESHOLD = 0.80
VERSION = "v4.3_cross_source_aligned_2026-05-13"


@dataclass
class V43ScoreResult:
    score: float
    decision: str
    reason: str
    n_swaps: int
    features: Dict[str, float] = field(default_factory=dict)


def sandwich_confirmed_mask(timestamps: np.ndarray, prices: np.ndarray,
                              window_sec: float = SANDWICH_WINDOW_SEC,
                              min_conf: int = SANDWICH_MIN_CONF) -> np.ndarray:
    """For each swap i, mark True iff ≥ min_conf OTHER swaps within ±window_sec
    have price ≥ 0.80 × prices[i].

    Semantic byte-parity with `geyser_stream.py:855 get_peak_price_since` (verified
    2026-05-13): live uses `len(nearby) >= min_confirmations + 1` where `nearby`
    INCLUDES candidate r itself. Here `prices[lo:hi]` likewise INCLUDES prices[i]
    (timestamps[i] is within its own ±window_sec), so `count_above_80 >= min_conf + 1`
    matches live byte-for-byte. Default min_conf=2 → requires ≥3 swaps (candidate + 2).

    Assumes timestamps is monotone non-decreasing.
    """
    n = len(timestamps)
    confirmed = np.zeros(n, dtype=bool)
    if n == 0:
        return confirmed
    for i in range(n):
        if prices[i] <= 0:
            continue
        lo = np.searchsorted(timestamps, timestamps[i] - window_sec)
        hi = np.searchsorted(timestamps, timestamps[i] + window_sec, side='right')
        # prices[lo:hi] INCLUDES candidate at index i
        count_above_80 = (prices[lo:hi] >= SANDWICH_THRESHOLD * prices[i]).sum()
        confirmed[i] = count_above_80 >= min_conf + 1
    return confirmed


def compute_label_v4_3(swaps: pd.DataFrame, graduation_time: float) -> Optional[int]:
    """Compute v4.3 binary rug label for one mint — byte-parity with v0.9f canonical.

    Spec §5.3. Required cols: timestamp, price_sol (>0).
    Assumes input pre-filtered for status='ok' and dust 0.05 SOL.

    Returns:
        1 if rug (drawdown ≤ -55% in 2-min label window from 10s-median anchor)
        0 if not rug
        None if label cannot be computed (insufficient data → drop mint)
    """
    ANCHOR_WINDOW_SEC = 10
    LABEL_WINDOW_SEC = 120
    DRAWDOWN_THRESHOLD = -0.55
    MIN_ANCHOR_SWAPS = 3

    entry_t = graduation_time + ENTRY_DELAY_SEC

    # Step 1: ANCHOR = median price in [entry_t, entry_t + 10s], else fallback first swap
    anchor_window = swaps[
        (swaps['timestamp'] >= entry_t) &
        (swaps['timestamp'] <= entry_t + ANCHOR_WINDOW_SEC) &
        (swaps['price_sol'] > 0)
    ]
    if len(anchor_window) >= MIN_ANCHOR_SWAPS:
        anchor = float(np.median(anchor_window['price_sol']))
    else:
        post_entry = swaps[
            (swaps['timestamp'] >= entry_t) &
            (swaps['price_sol'] > 0)
        ].sort_values('timestamp')
        if len(post_entry) == 0:
            return None
        anchor = float(post_entry.iloc[0]['price_sol'])

    if anchor <= 0:
        return None

    # Step 2: drawdown in [entry_t, entry_t + 120s]
    label_window = swaps[
        (swaps['timestamp'] >= entry_t) &
        (swaps['timestamp'] <= entry_t + LABEL_WINDOW_SEC) &
        (swaps['price_sol'] > 0)
    ]
    if len(label_window) < 3:
        return None
    min_price = float(label_window['price_sol'].min())
    drawdown = (min_price / anchor) - 1.0

    return int(drawdown <= DRAWDOWN_THRESHOLD)


def _compute_v4_3_features(swaps_df: pd.DataFrame, graduation_time: float,
                              sol_5m_dict: Optional[dict] = None,
                              window_s: int = 300) -> Optional[Dict[str, float]]:
    """Compute 24 features in [graduation_time, graduation_time + 300s].

    Forward-only, dust-filtered. Mirrors training-time pipeline byte-for-byte.

    Args:
        swaps_df: required cols block_time / timestamp, sol_amount / volume_sol,
                  effective_price_sol / price_sol, is_buy, trader_address.
        graduation_time: T0 Unix seconds.
        sol_5m_dict: optional SOL/USDT 5min close prices keyed by floor-5m unix time.
        window_s: forward window (default 300s).

    Returns dict or None if data insufficient.
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

    grad_t = float(graduation_time)
    s = s[(s["block_time"] >= grad_t) & (s["block_time"] <= grad_t + window_s)]
    s = s[s["sol_amount"] >= DUST_SOL]
    if len(s) < 10:
        return None
    s = s.sort_values("block_time").reset_index(drop=True)

    out: Dict[str, float] = {f: None for f in FEATURES_V4_3}

    win60 = s[s["block_time"] <= grad_t + 60]
    win300 = s

    # ── Tier 1: count features ──
    out["swap_count_60s"] = float(len(win60))
    out["swap_count_300s"] = float(len(win300))
    if len(win300) > 0:
        out["swap_density_ratio_60_300s"] = (len(win60) / 60.0) / (len(win300) / 300.0)

    # ── Tier 1: volume statistics ──
    if len(win60) > 0:
        out["mean_swap_size_60s"] = float(win60["sol_amount"].mean())
    out["mean_swap_size_300s"] = float(win300["sol_amount"].mean())
    out["p95_swap_size_300s"] = float(win300["sol_amount"].quantile(0.95))
    out["total_vol_sol_300s"] = float(win300["sol_amount"].sum())

    # ── Tier 1 + Tier 2: trader / direction shares ──
    if "trader_address" in win60.columns:
        n60_traders = int(win60["trader_address"].nunique())
        n300_traders = int(win300["trader_address"].nunique())
        out["unique_buyers_300s"] = float(n300_traders)
        if n60_traders > 0:
            out["new_buyer_rate_60_300"] = (n300_traders - n60_traders) / max(n60_traders, 1)

    out["buy_count_share_300s"] = float((win300["is_buy"] == 1).mean())
    out["sell_share_300s"] = float((win300["is_buy"] == 0).mean())

    # OFI (order flow imbalance): (buy_vol - sell_vol) / total_vol
    buy_vol = float(win300[win300["is_buy"] == 1]["sol_amount"].sum())
    sell_vol = float(win300[win300["is_buy"] == 0]["sol_amount"].sum())
    tot_vol = buy_vol + sell_vol
    if tot_vol > 0:
        out["ofi_300s"] = (buy_vol - sell_vol) / tot_vol

    # ── Tier 1 NEW: buyer concentration (HHI + top3 share) ──
    # Computed over BUY-side trader volumes
    buys = win300[win300["is_buy"] == 1]
    if len(buys) > 0 and "trader_address" in buys.columns:
        trader_buy_vols = buys.groupby("trader_address")["sol_amount"].sum()
        total_buy_vol = float(trader_buy_vols.sum())
        if total_buy_vol > 0:
            shares = trader_buy_vols / total_buy_vol
            out["hc_hhi_300s"] = float((shares ** 2).sum())
            top3 = shares.sort_values(ascending=False).head(3).sum()
            out["hc_top3_share_300s"] = float(top3)

    # ── Tier 3: sandwich-filtered drawdown_from_peak_300s_sw + n_local_maxima_300s_sw ──
    ts300 = win300["block_time"].values
    px300 = pd.to_numeric(win300["effective_price_sol"], errors="coerce").values
    valid_px_mask = (px300 > 0) & np.isfinite(px300)
    if valid_px_mask.sum() < 5:
        return None
    ts_v = ts300[valid_px_mask]
    px_v = px300[valid_px_mask]

    sw_mask = sandwich_confirmed_mask(ts_v, px_v,
                                          window_sec=SANDWICH_WINDOW_SEC,
                                          min_conf=SANDWICH_MIN_CONF)
    confirmed_px = px_v[sw_mask]
    if len(confirmed_px) >= 2:
        peak_sw = float(confirmed_px.max())
        last_sw = float(confirmed_px[-1])
        if peak_sw > 0:
            out["drawdown_from_peak_300s_sw"] = (last_sw - peak_sw) / peak_sw
        if len(confirmed_px) >= 3:
            mids = confirmed_px[1:-1]
            is_max = (mids > confirmed_px[:-2]) & (mids > confirmed_px[2:])
            out["n_local_maxima_300s_sw"] = float(int(is_max.sum()))
        else:
            out["n_local_maxima_300s_sw"] = 0.0

    # ── Tier 4: microstructure (v0.9f) ──
    vol300 = win300["sol_amount"].values
    is_buy300 = win300["is_buy"].values

    if valid_px_mask.sum() >= 10:
        rets = np.diff(np.log(np.maximum(px_v, 1e-12)))
        vols_post = vol300[valid_px_mask][1:]
        if len(rets) > 5:
            with np.errstate(divide='ignore', invalid='ignore'):
                lam = np.abs(rets) / np.maximum(vols_post, 0.01)
                lam_f = lam[np.isfinite(lam)]
                if len(lam_f):
                    out["kyle_lambda_300s"] = float(np.median(lam_f))
            cov_lag = float(np.cov(rets[:-1], rets[1:])[0, 1])
            out["roll_spread_300s"] = float(2 * np.sqrt(-cov_lag)) if cov_lag < 0 else 0.0

    # VPIN (30s buckets)
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

    if vol300.sum() > 0:
        out["top1_swap_share_300s"] = float(vol300.max() / vol300.sum())

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

    # ── Macro ──
    grad_dt = pd.Timestamp(grad_t, unit='s')
    out["graduation_hour_utc"] = float(grad_dt.hour)

    if sol_5m_dict:
        rets_macro = []
        for back_min in range(0, 60, 5):
            ts_a = int(grad_t - back_min * 60) // 300 * 300
            ts_b = int(grad_t - (back_min + 5) * 60) // 300 * 300
            a = sol_5m_dict.get(ts_a)
            b = sol_5m_dict.get(ts_b)
            if a and b and b > 0:
                rets_macro.append((a - b) / b)
        if len(rets_macro) >= 6:
            out["sol_volatility_1h_pre_grad"] = float(np.std(rets_macro))

    return out


class RugFilterV4_3:
    """v4.3: XGBoost RAW probability (no LR calibrator), 24 features.

    Inference recipe:
      score = xgb.predict_proba(X)[:, 1][0]
      decision = "REJECT" if score >= cutoff else "PASS"
    """

    VERSION = VERSION

    def __init__(self, model_path: str, cutoff: float = 0.5,
                 window_s: int = 300, min_swaps: int = 10,
                 *, metrics: Optional[Any] = None):
        if not HAS_SKLEARN_STACK:
            raise RuntimeError("RugFilterV4_3 requires numpy, pandas, sklearn, xgboost")

        self.guarded = _model_registry.get_or_load(model_path, metrics=metrics)
        self._pkg = self.guarded.unwrap()

        for k in ("xgb_model", "feature_cols", "use_raw_probability"):
            if k not in self._pkg:
                raise ValueError(f"v4.3 pkl missing key: {k}")
        if not self._pkg.get("use_raw_probability"):
            raise ValueError("v4.3 pkl must have use_raw_probability=True")

        self.xgb = self._pkg["xgb_model"]
        self.feature_cols = list(self._pkg["feature_cols"])
        self.cutoff = float(cutoff)
        self.window_s = int(window_s)
        self.min_swaps = int(min_swaps)

        if list(self.feature_cols) != FEATURES_V4_3:
            logger.warning(
                f"RugFilterV4_3: pickle features differ from FEATURES_V4_3 "
                f"(pkl_n={len(self.feature_cols)}, expected_n={len(FEATURES_V4_3)})")

        self._sol_5m_dict: Optional[dict] = None

    def load_sol_bars(self, sol_5m_dict: dict):
        self._sol_5m_dict = sol_5m_dict

    def score_features(self, features: Dict[str, float], n_swaps: int) -> V43ScoreResult:
        try:
            x = np.array([features.get(f, np.nan) for f in self.feature_cols],
                            dtype=float).reshape(1, -1)
            score = float(self.xgb.predict_proba(x)[0, 1])
        except Exception as e:
            return V43ScoreResult(
                score=float("nan"), decision="SKIP_MODEL_ERROR",
                reason=f"predict_failed:{type(e).__name__}",
                n_swaps=n_swaps, features=features)

        if score >= self.cutoff:
            decision = "REJECT"
            reason = f"score={score:.4f}>=cutoff{self.cutoff:.4f}"
        else:
            decision = "PASS"
            reason = f"score={score:.4f}<{self.cutoff:.4f}"
        return V43ScoreResult(
            score=score, decision=decision, reason=reason,
            n_swaps=n_swaps, features=features)

    def score_from_swaps(self, df: "pd.DataFrame", graduation_time: float) -> V43ScoreResult:
        """Score from an injected pre-loaded, window-filtered swaps DataFrame.

        RLock-safe entry point: the caller loads swaps via TradeDB (which holds
        the RLock) and passes the resulting DataFrame here.  No sqlite3.connect
        is opened inside this method.

        Contract (parity with score_from_sqlite):
          - df must have columns [timestamp, trader_address, is_buy,
            volume_sol, price_sol] in that order (same as the SQL SELECT).
          - df must already be filtered to the scoring window
            [graduation_time, graduation_time + window_s] — the caller is
            responsible for applying the same bounds as the SQL WHERE clause.
          - Returns an identical V43ScoreResult to score_from_sqlite when df
            equals the rows score_from_sqlite would have fetched.
        """
        if len(df) < self.min_swaps:
            return V43ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason=f"insufficient_swaps:{len(df)}<{self.min_swaps}",
                                     n_swaps=len(df), features={})
        feats = _compute_v4_3_features(df, graduation_time,
                                          sol_5m_dict=self._sol_5m_dict,
                                          window_s=self.window_s)
        if feats is None:
            return V43ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason="features_None_insufficient_data",
                                     n_swaps=len(df), features={})
        return self.score_features(feats, n_swaps=len(df))

    def score_from_sqlite(self, db_path: str, mint_address: str,
                            graduation_time: float) -> V43ScoreResult:
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
            return V43ScoreResult(score=float("nan"), decision="SKIP_NO_DATA",
                                     reason=f"sqlite_err:{type(e).__name__}",
                                     n_swaps=0, features={})

        df = pd.DataFrame(rows, columns=["timestamp", "trader_address", "is_buy",
                                              "volume_sol", "price_sol"])
        return self.score_from_swaps(df, graduation_time)
