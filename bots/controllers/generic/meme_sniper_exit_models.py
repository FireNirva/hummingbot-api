"""
14y dual-model exit prediction — pure-logic inference module.

Loads two frozen models (Tier B 19-feature and F2a+HC 25-feature) and exposes
feature-compute + predict functions. No I/O, no side effects; safe to import
in the controller.

All feature-compute functions are byte-for-byte identical to the training-time
functions in:
  - scripts/14y_build_sliding_panel.py (Tier A 11 features)
  - scripts/14y_augment_tier_b.py (Tier B 5 features, minus the redundant
    tb_ofi_60s which equals sf_last2m_net_flow by construction)
  - scripts/14y_augment_holder_concentration.py (HC 6 features)

A parity test (tests/test_exit_models_feature_parity.py) verifies that for
20 random rows of the training panel, the functions here reproduce the stored
feature values to 1e-6.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_MODULE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _MODULE_DIR / "models"

# --- feature ordering (order-sensitive — matches training) ---
TIER_A_ORIG = [
    "sf_sell_accel", "sf_sell_vol_accel", "sf_large_sell_count", "sf_buyer_decel",
    "sf_early_buyer_sell", "sf_whale_sell_frac", "sf_buy_size_ratio",
    "sf_last2m_net_flow", "sf_new_sellers_last3", "sf_sell_cluster",
    "sf_large_buy_gone",
]
TIER_A_NEW = ["sf_swap_density", "sf_swap_density_late", "sf_dt_from_grad"]
TIER_B_NEW = [
    "tb_ofi_180s", "tb_vpin_abs_180s", "tb_kyle_lambda_180s",
    "tb_amihud_illiq_180s", "tb_top3_netflow_180s",
]
HC_FEATURES = [
    "hc_hhi_t", "hc_top1_share_t", "hc_top3_share_t",
    "hc_gini_t", "hc_entropy_t", "hc_n_net_long_t",
]
TIER_B_FEATURE_ORDER = TIER_A_ORIG + TIER_A_NEW + TIER_B_NEW         # 19
F2A_HC_FEATURE_ORDER = TIER_A_ORIG + TIER_A_NEW + TIER_B_NEW + HC_FEATURES  # 25

# --- windowing constants (MUST match training) ---
FEATURE_WINDOW = 180        # [t-180, t]
EARLY_SEG_END = 120         # early segment: [t-180, t-120]
LATE_SEG_START = 60         # late segment:  [t-60,  t]
MIN_SWAPS_FEATURE_WIN = 8
LARGE_SELL_SOL = 2.0
LARGE_BUY_SOL = 1.0
CLUSTER_WINDOW_SEC = 30
SMOOTH_WINDOW = 5
EPS = 1e-12


# ---------------------------------------------------------------------------
# Tier A 11 features (windowing relative to t)
# ---------------------------------------------------------------------------

def compute_tier_a(
    win: pd.DataFrame, early: pd.DataFrame, late: pd.DataFrame,
    pre_t: pd.DataFrame,
) -> dict | None:
    """
    Inputs must already be sliced by the caller (all block_time ≤ t):
      win   : swaps in [t-180, t]       (≡ 14l 'last3')
      early : swaps in [t-180, t-120]   (≡ 14l 'first2')
      late  : swaps in [t-60,  t]       (≡ 14l 'last2')
      pre_t : all swaps with block_time ≤ t
    Returns None if insufficient swaps to compute reliable features.
    """
    if len(win) < MIN_SWAPS_FEATURE_WIN:
        return None

    sells_win = win[win["is_buy"] == 0]
    buys_win = win[win["is_buy"] == 1]
    sells_early = early[early["is_buy"] == 0]
    sells_late = late[late["is_buy"] == 0]
    buys_early = early[early["is_buy"] == 1]
    buys_late = late[late["is_buy"] == 1]

    feats: dict[str, float] = {}

    feats["sf_sell_accel"] = len(sells_late) / max(len(sells_early), 1)

    vol_early = float(sells_early["sol_amount"].sum())
    vol_late = float(sells_late["sol_amount"].sum())
    feats["sf_sell_vol_accel"] = vol_late / max(vol_early, 0.01)

    feats["sf_large_sell_count"] = int((sells_win["sol_amount"] > LARGE_SELL_SOL).sum())

    ub_early = buys_early["trader_address"].nunique()
    ub_late = buys_late["trader_address"].nunique()
    feats["sf_buyer_decel"] = ub_late / max(ub_early, 1)

    early_buyers = set(buys_early["trader_address"].unique()) - {""}
    if early_buyers:
        late_sellers = set(sells_late["trader_address"].unique())
        feats["sf_early_buyer_sell"] = len(early_buyers & late_sellers) / len(early_buyers)
    else:
        feats["sf_early_buyer_sell"] = 0.0

    buys_pre = pre_t[pre_t["is_buy"] == 1]
    if len(buys_pre):
        bv = buys_pre.groupby("trader_address")["sol_amount"].sum()
        bv = bv[bv.index != ""]
        if len(bv) >= 3:
            top3 = set(bv.nlargest(3).index)
            sellers_win = set(sells_win["trader_address"].unique())
            feats["sf_whale_sell_frac"] = len(top3 & sellers_win) / 3
        else:
            feats["sf_whale_sell_frac"] = 0.0
    else:
        feats["sf_whale_sell_frac"] = 0.0

    bs_early = float(buys_early["sol_amount"].mean()) if len(buys_early) else 0.0
    bs_late = float(buys_late["sol_amount"].mean()) if len(buys_late) else 0.0
    feats["sf_buy_size_ratio"] = bs_late / max(bs_early, 0.01)

    buy_vol_late = float(buys_late["sol_amount"].sum())
    feats["sf_last2m_net_flow"] = buy_vol_late - vol_late

    # new sellers: appear in late seg but not seen before (t - LATE_SEG_START)
    pre_late_mask_t = win["block_time"].max() - LATE_SEG_START
    pre_late = pre_t[pre_t["block_time"] < pre_late_mask_t]
    known = set(pre_late["trader_address"].unique()) - {""}
    new_sellers = set(sells_late["trader_address"].unique()) - known - {""}
    feats["sf_new_sellers_last3"] = len(new_sellers)

    if len(sells_win) >= 2:
        times = sells_win["block_time"].sort_values().to_numpy()
        max_cluster = 0
        for ts in times:
            cluster = int(((times >= ts) & (times <= ts + CLUSTER_WINDOW_SEC)).sum())
            if cluster > max_cluster:
                max_cluster = cluster
        feats["sf_sell_cluster"] = max_cluster
    else:
        feats["sf_sell_cluster"] = len(sells_win)

    lb_early = int((buys_early["sol_amount"] > LARGE_BUY_SOL).sum())
    lb_late = int((buys_late["sol_amount"] > LARGE_BUY_SOL).sum())
    feats["sf_large_buy_gone"] = lb_late / max(lb_early, 1)

    return feats


# ---------------------------------------------------------------------------
# Tier B 5 features (same windowing)
# ---------------------------------------------------------------------------

def compute_tier_b(
    win: pd.DataFrame, late: pd.DataFrame, pre_t: pd.DataFrame,
    prices_win: np.ndarray,
) -> dict:
    """prices_win: smoothed prices at each swap in win, same ordering."""
    feats: dict[str, float] = {}

    def signed_flow(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return float((df["sol_amount"] * (2 * df["is_buy"] - 1)).sum())

    feats["tb_ofi_180s"] = signed_flow(win)

    total_vol = float(win["sol_amount"].sum())
    if total_vol > 0:
        buy_vol = float(win.loc[win["is_buy"] == 1, "sol_amount"].sum())
        sell_vol = total_vol - buy_vol
        feats["tb_vpin_abs_180s"] = abs(buy_vol - sell_vol) / total_vol
    else:
        feats["tb_vpin_abs_180s"] = 0.0

    if len(win) >= 3 and np.all(np.isfinite(prices_win)) and np.all(prices_win > 0):
        lp = np.log(prices_win)
        r = np.diff(lp)
        q = (win["sol_amount"].to_numpy()[1:]
             * (2 * win["is_buy"].to_numpy()[1:] - 1))
        abs_r = np.abs(r)
        var_q = float(np.var(q))
        if var_q > EPS:
            cov_rq = float(np.mean(abs_r * q) - np.mean(abs_r) * np.mean(q))
            feats["tb_kyle_lambda_180s"] = cov_rq / var_q
        else:
            feats["tb_kyle_lambda_180s"] = 0.0
        abs_q = np.abs(q)
        valid = abs_q > EPS
        if valid.any():
            feats["tb_amihud_illiq_180s"] = float(np.mean(abs_r[valid] / abs_q[valid]))
        else:
            feats["tb_amihud_illiq_180s"] = 0.0
    else:
        feats["tb_kyle_lambda_180s"] = 0.0
        feats["tb_amihud_illiq_180s"] = 0.0

    buys_pre = pre_t[pre_t["is_buy"] == 1]
    if len(buys_pre):
        bv = buys_pre.groupby("trader_address")["sol_amount"].sum()
        bv = bv[bv.index != ""]
        if len(bv) >= 3:
            top3 = set(bv.nlargest(3).index)
            top3_win = win[win["trader_address"].isin(top3)]
            feats["tb_top3_netflow_180s"] = signed_flow(top3_win)
        else:
            feats["tb_top3_netflow_180s"] = 0.0
    else:
        feats["tb_top3_netflow_180s"] = 0.0

    return feats


# ---------------------------------------------------------------------------
# HC 6 features (holder concentration from cumulative net positions)
# ---------------------------------------------------------------------------

def compute_hc(pre_t: pd.DataFrame) -> dict:
    """Net-long holder concentration. pre_t = swaps with block_time ≤ t."""
    if len(pre_t) == 0:
        return {
            "hc_hhi_t": 0.0, "hc_top1_share_t": 0.0, "hc_top3_share_t": 0.0,
            "hc_gini_t": 0.0, "hc_entropy_t": 0.0, "hc_n_net_long_t": 0,
        }
    signed = pre_t["sol_amount"] * (2 * pre_t["is_buy"] - 1)
    net = pre_t.assign(signed=signed).groupby("trader_address")["signed"].sum()
    net = net[net.index != ""]
    long_positions = net[net > 0]
    n_long = int(len(long_positions))
    if n_long == 0:
        return {
            "hc_hhi_t": 0.0, "hc_top1_share_t": 0.0, "hc_top3_share_t": 0.0,
            "hc_gini_t": 0.0, "hc_entropy_t": 0.0, "hc_n_net_long_t": 0,
        }
    total = float(long_positions.sum())
    shares = long_positions.to_numpy() / total
    shares_sorted = np.sort(shares)[::-1]

    hhi = float(np.sum(shares_sorted ** 2))
    top1 = float(shares_sorted[0])
    top3 = float(shares_sorted[:3].sum())
    n = len(shares_sorted)
    if n >= 2:
        asc = np.sort(shares)
        idx = np.arange(1, n + 1)
        gini = float((2 * np.sum(idx * asc) / (n * np.sum(asc))) - (n + 1) / n)
    else:
        gini = 0.0
    entropy = float(-np.sum(shares * np.log(np.clip(shares, 1e-12, 1.0))))

    return {
        "hc_hhi_t": hhi, "hc_top1_share_t": top1, "hc_top3_share_t": top3,
        "hc_gini_t": gini, "hc_entropy_t": entropy, "hc_n_net_long_t": n_long,
    }


# ---------------------------------------------------------------------------
# Orchestration: slice swaps by t, compute all features
# ---------------------------------------------------------------------------

def compute_all_features(
    swaps: pd.DataFrame, t: int, grad_time: int,
    *, feature_window: int = FEATURE_WINDOW,
) -> dict[str, Any] | None:
    """
    Produce all 25 features plus the Tier A counts (n_swaps_window / _late)
    used as sf_swap_density / sf_swap_density_late.

    swaps: sorted by block_time ascending, all columns
      token_address, block_time, trader_address, is_buy, sol_amount,
      token_amount, effective_price_sol
    t:         decision time (unix seconds)
    grad_time: token's graduation time (unix seconds)
    feature_window: lookback window in seconds (default 180 = training
      distribution). Pass 300 only as a sparse-data fallback; sum-type
      features (sf_swap_density, tb_ofi_180s, etc.) become scale-shifted
      relative to training, so use only when 180s would otherwise return None.

    Returns None if cannot compute (too few swaps).
    """
    if len(swaps) == 0:
        return None

    block_times = swaps["block_time"].to_numpy()
    pre_t_mask = block_times <= t
    if pre_t_mask.sum() < MIN_SWAPS_FEATURE_WIN:
        return None

    # Phase 16.3 audit-6 fix — use boolean mask instead of positional slice.
    # Old: swaps.iloc[:pre_t_mask.sum()] assumed swaps is sorted ascending,
    # which is true for the SQL-fed path (`ORDER BY timestamp`) but fragile
    # if a future caller passes an unsorted DataFrame (e.g., shuffled, or
    # filtered after a join). Boolean mask is robust to row order.
    pre_t = swaps.loc[pre_t_mask]

    win_mask = (block_times >= t - feature_window) & (block_times <= t)
    if win_mask.sum() < MIN_SWAPS_FEATURE_WIN:
        return None
    win = swaps.loc[win_mask]
    early = swaps.loc[
        (block_times >= t - feature_window) & (block_times <= t - EARLY_SEG_END)
    ]
    late = swaps.loc[(block_times >= t - LATE_SEG_START) & (block_times <= t)]

    # Smoothed price for Tier B
    smoothed = (
        swaps["effective_price_sol"]
        .rolling(SMOOTH_WINDOW, min_periods=1).median().to_numpy()
    )
    prices_win = smoothed[win_mask]

    # Tier A
    a = compute_tier_a(win, early, late, pre_t)
    if a is None:
        return None

    # Tier A "new" additions
    a["sf_swap_density"] = float(win_mask.sum())
    a["sf_swap_density_late"] = float(len(late))
    a["sf_dt_from_grad"] = float(t - grad_time)

    # Tier B
    b = compute_tier_b(win, late, pre_t, prices_win)

    # HC
    hc = compute_hc(pre_t)

    return {**a, **b, **hc}


def to_feature_array(
    feats: dict[str, Any], order: list[str],
) -> np.ndarray:
    """Order a feature dict into the array shape expected by the model."""
    return np.asarray([float(feats[k]) for k in order], dtype=float)


# ---------------------------------------------------------------------------
# Model loading + prediction
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(filename: str):
    if filename in _MODEL_CACHE:
        return _MODEL_CACHE[filename]
    path = _MODELS_DIR / filename
    with path.open("rb") as f:
        obj = pickle.load(f)
    _MODEL_CACHE[filename] = obj
    return obj


def load_tier_b():
    """⚠️ DEPRECATED 2026-04-25 — DO NOT USE FOR DECISIONS.

    Tier B 19-feature `p_drop_raw` model. Real-trade backtest (29 trades)
    showed AUC 0.43 (INVERTED — score anti-correlated with bot losses):
      Avg score on RUG: 0.422  vs  WIN: 0.477  (wrong direction)
      Recall@0.50 = 12%, FPR@0.50 = 43%
    See `Phase_14y_v2_TargetAligned_FROZEN_2026-04-25.md` §1.1 for backtest
    details. Loader retained for historical pkl access only; no production
    code path should compute predictions from this model.
    """
    return _load_model("swap_exit_tier_b_model.pkl")


def load_f2a_hc():
    return _load_model("swap_exit_f2a_hc_model.pkl")


def load_f2a_hc_live_v1():
    """⚠️ DEPRECATED 2026-04-25 — DO NOT USE FOR DECISIONS.

    Bot-schema retrain (2026-04-24 A/B). Score distribution collapsed to
    [0.03, 0.255] — never crosses any actionable cutoff in production.
    Real-trade backtest: recall = 0/10. Spearman ρ = 0.73 with `p_rug_raw`
    (fully redundant). See `Phase_14y_v4_PositionConditional_FROZEN_2026-04-25.md`
    Appendix C audit. Loader retained for historical pkl access only.
    """
    return _load_model("swap_exit_f2a_hc_live_v1.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# v3 Enhanced — event-anchored rug predictor (LogReg + StandardScaler)
# Trained via source-stratified split + 28 features (4 redundant dropped)
# See Phase_14y_v3_EventAnchored_FROZEN_2026-04-25.md §14.8
# ─────────────────────────────────────────────────────────────────────────────

# Features dropped from v3 audit (redundant, |r|>0.85 with kept feature)
V3_REDUNDANT_DROPPED = {
    "sf_last2m_net_flow",   # r=1.000 with tb_ofi_60s
    "sf_swap_density_late", # r=0.932 with sf_swap_density
    "hc_top1_share_t",      # r=0.969 with hc_hhi_t
    "hc_hhi_t",             # dropped; keep top3_share + entropy + gini
}

V3_EXTRA_FEATURES = [
    "tb_ofi_60s", "tb_vpin_60s", "sf_price_return_60s",
    "sf_price_volatility_60s", "sf_vol_decay_ratio",
    "sf_buyer_repeat_rate", "sf_time_since_last_big_sell",
]
V3_ENHANCED_FEATURE_ORDER = [
    f for f in F2A_HC_FEATURE_ORDER if f not in V3_REDUNDANT_DROPPED
] + V3_EXTRA_FEATURES  # 28 features


def compute_v3_extra_features(
    swaps: pd.DataFrame, t: int, *, feature_window: int = FEATURE_WINDOW,
) -> dict | None:
    """Compute 7 v3-specific features. last_60 always uses fixed 60s; early_60
    uses [t-feature_window, t-60). Default feature_window=180 = training.
    """
    bt = swaps["block_time"].to_numpy()
    pre_t_mask = bt <= t
    if pre_t_mask.sum() < 5:
        return None

    pre_t = swaps.loc[pre_t_mask]  # boolean mask (audit-6: was iloc + assumed sort)
    last_60 = pre_t[pre_t["block_time"] >= t - 60]
    early_60 = pre_t[(pre_t["block_time"] >= t - feature_window) & (pre_t["block_time"] < t - 60)]

    feats = {}

    # tb_ofi_60s
    if len(last_60) > 0:
        feats["tb_ofi_60s"] = float(
            (last_60["sol_amount"] * (2 * last_60["is_buy"] - 1)).sum())
    else:
        feats["tb_ofi_60s"] = 0.0

    # tb_vpin_60s
    if len(last_60) > 0:
        total_vol = float(last_60["sol_amount"].sum())
        if total_vol > 0:
            buy_vol = float(last_60.loc[last_60["is_buy"] == 1, "sol_amount"].sum())
            sell_vol = total_vol - buy_vol
            feats["tb_vpin_60s"] = abs(buy_vol - sell_vol) / total_vol
        else:
            feats["tb_vpin_60s"] = 0.0
    else:
        feats["tb_vpin_60s"] = 0.0

    # sf_price_return_60s
    if len(last_60) > 0 and len(early_60) > 0:
        p_now = float(last_60["effective_price_sol"].iloc[-1])
        p_60s_ago = float(early_60["effective_price_sol"].iloc[-1])
        feats["sf_price_return_60s"] = float(
            np.log(p_now / p_60s_ago) if (p_60s_ago > 0 and p_now > 0) else 0.0)
    else:
        feats["sf_price_return_60s"] = 0.0

    # sf_price_volatility_60s
    if len(last_60) >= 3:
        prices = last_60["effective_price_sol"].to_numpy()
        if np.all(prices > 0):
            feats["sf_price_volatility_60s"] = float(np.std(np.diff(np.log(prices))))
        else:
            feats["sf_price_volatility_60s"] = 0.0
    else:
        feats["sf_price_volatility_60s"] = 0.0

    # sf_vol_decay_ratio
    vol_last = float(last_60["sol_amount"].sum())
    vol_early = float(early_60["sol_amount"].sum())
    feats["sf_vol_decay_ratio"] = float(vol_last / max(vol_early, 0.01))

    # sf_buyer_repeat_rate
    buyers_last = set(last_60.loc[last_60["is_buy"] == 1, "trader_address"].unique()) - {""}
    if buyers_last:
        traders_before = set(pre_t.loc[pre_t["block_time"] < t - 60, "trader_address"].unique()) - {""}
        feats["sf_buyer_repeat_rate"] = float(len(buyers_last & traders_before) / len(buyers_last))
    else:
        feats["sf_buyer_repeat_rate"] = 0.0

    # sf_time_since_last_big_sell
    big_sells = pre_t[(pre_t["is_buy"] == 0) & (pre_t["sol_amount"] >= 5.0)]
    if len(big_sells) > 0:
        feats["sf_time_since_last_big_sell"] = float(t - big_sells["block_time"].iloc[-1])
    else:
        feats["sf_time_since_last_big_sell"] = 9999.0

    return feats


def load_14y_v3_enhanced():
    """Event-anchored rug predictor (v3). LogReg + StandardScaler over 28 features.
    Payload has: lr_model, scaler, feature_cols, cutoff.
    See Phase_14y_v3_EventAnchored_FROZEN_2026-04-25.md §14.8.
    """
    return _load_model("swap_exit_f2a_hc_v3_enhanced.pkl")


V3_FALLBACK_WINDOWS = (180, 300)
V3_LAST_WINDOW_USED: int | None = None  # diagnostic — 180 = primary, 300 = sparse-fallback


def predict_14y_v3(swaps: pd.DataFrame, t: int, grad_time: int) -> float | None:
    """Score with v3 LogReg. Returns rug probability in [0, 1] or None if
    insufficient data.

    Multi-window inference (2026-04-25): tries the primary 180s window first
    (training distribution). When that returns None due to swap-density
    shortfall, retries at 300s as a sparse-data fallback. The 300s path is
    OOD relative to training (sum-type features ~1.67× larger), but a
    conservative score is preferable to silently dropping the position from
    the shadow log. Module-level `V3_LAST_WINDOW_USED` records which window
    actually scored — caller can log/persist it for offline auditing.
    """
    global V3_LAST_WINDOW_USED
    V3_LAST_WINDOW_USED = None

    try:
        m_data = load_14y_v3_enhanced()
    except Exception:
        return None
    lr = m_data.get("lr_model")
    scaler = m_data.get("scaler")
    feature_cols = m_data.get("feature_cols", V3_ENHANCED_FEATURE_ORDER)
    if lr is None or scaler is None:
        return None

    for fw in V3_FALLBACK_WINDOWS:
        base_feats = compute_all_features(swaps, t, grad_time, feature_window=fw)
        if base_feats is None:
            continue
        extras = compute_v3_extra_features(swaps, t, feature_window=fw)
        if extras is None:
            continue
        all_feats = {**base_feats, **extras}
        x = np.asarray([[float(all_feats.get(f, 0.0)) for f in feature_cols]])
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_s = scaler.transform(x)
        V3_LAST_WINDOW_USED = fw
        return float(lr.predict_proba(x_s)[0, 1])
    return None


def predict(
    swaps: pd.DataFrame, t: int, grad_time: int,
) -> dict[str, float] | None:
    """Compute features once, score both models."""
    feats = compute_all_features(swaps, t, grad_time)
    if feats is None:
        return None
    tb = load_tier_b()["model"]
    fh = load_f2a_hc()["model"]
    x_tb = to_feature_array(feats, TIER_B_FEATURE_ORDER).reshape(1, -1)
    x_fh = to_feature_array(feats, F2A_HC_FEATURE_ORDER).reshape(1, -1)
    p_drop = float(tb.predict_proba(x_tb)[0, 1])
    p_rug = float(fh.predict_proba(x_fh)[0, 1])
    return {"p_drop_raw": p_drop, "p_rug_raw": p_rug}


# ─────────────────────────────────────────────────────────────────────────────
# v4 Position-Conditional — 3-horizon ensemble (XGBoost + isotonic calibration)
# Trained on 109K rows (Birdeye Mar 5-Apr 13 + bot Apr 15-25, OOS-clean split).
# Features = 32 v3 + 3 position-state. Drops y_remaining (first-tick AUC ≈ random).
# Ensemble = 0.6 × calibrated p(y_60s) + 0.4 × calibrated p(y_120s).
# Cutoff 0.475 (defensible, not test-tuned).
# See Phase_14y_v4_PositionConditional_FROZEN_2026-04-25.md (Appendix C-E).
# Shadow-only — no production gate until Day 20 live data review.
# ─────────────────────────────────────────────────────────────────────────────

V4_PS_FEATURES = ["ps_current_pnl_pct", "ps_hold_sec", "ps_in_ec_window"]
V4_FEATURES = F2A_HC_FEATURE_ORDER[:]  # base 25
# Append the 7 v3 extras and 3 PS features in the same order used at training:
V4_V3_EXTRAS = [
    "tb_ofi_60s", "tb_vpin_60s", "sf_price_return_60s",
    "sf_price_volatility_60s", "sf_vol_decay_ratio",
    "sf_buyer_repeat_rate", "sf_time_since_last_big_sell",
]
V4_FEATURE_ORDER = V4_FEATURES + V4_V3_EXTRAS + V4_PS_FEATURES  # 35 total
V4_HORIZONS_DEPLOYED = ("y_60s", "y_120s")          # drop y_remaining
V4_ENSEMBLE_WEIGHTS = {"y_60s": 0.6, "y_120s": 0.4}
V4_DEPLOY_CUTOFF = 0.475

V4_BOT_SL_PCT = 0.25
V4_BOT_EC_PCT = 0.20
V4_BOT_EC_WINDOW_SEC = 120


def compute_v4_position_state(price_t: float, entry_price: float,
                               hold_sec: int) -> dict | None:
    """3 entry-relative features. Mirrors training-time logic in
    `scripts/build_v4_dataset.py:compute_position_state_features`.
    """
    if (entry_price <= 0 or price_t <= 0
            or not np.isfinite(entry_price) or not np.isfinite(price_t)):
        return None
    pnl_pct = (price_t - entry_price) / entry_price
    if not np.isfinite(pnl_pct):
        return None
    pnl_pct = float(np.clip(pnl_pct, -1.0, 5.0))
    return {
        "ps_current_pnl_pct": pnl_pct,
        "ps_hold_sec": float(hold_sec),
        "ps_in_ec_window": int(hold_sec < V4_BOT_EC_WINDOW_SEC),
    }


def load_14y_v4_3horizon():
    """Position-conditional v4 ensemble (2026-04-25). Payload contains:
      - winners: dict[label → {type, model, scaler}]   (XGBoost won all)
      - calibrators: dict[label → IsotonicRegression]
      - feature_cols: ordered list of 35 features
    See Phase_14y_v4_PositionConditional_FROZEN_2026-04-25.md.
    """
    return _load_model("swap_exit_v4_3horizon.pkl")


# ────────────────────────────────────────────────────────────────────────
# Phase 15d v5 profit-protect features (2026-04-27)
# ────────────────────────────────────────────────────────────────────────
# v5 thesis: v4.1 was a rug-CONFIRMER (fired at -25% pnl). v5 retrains the
# same architecture with PROFIT-PROTECT label + 4 peak/reversal features
# to fire BEFORE the drawdown completes, while position is still in profit.
#
# Citations (see Phase_15d_V5_Profit_Protect_DESIGN_2026-04-27.md §4.2):
#   - mh_sell_intensity_60s ← arXiv 2408.03594 (Hawkes OFI forecast),
#     simplified to EWMA proxy with τ=30s decay
#   - mh_spread_change_180s ← arXiv 2504.15790 (P&D microstructure)
#     spread divergence-then-collapse pattern at pump termination
#   - ps_peak_pnl_so_far + ps_drawdown_from_peak ← LP meta-labeling

V5_LARGE_SELL_SOL = 2.0
V5_HAWKES_DECAY_SEC = 30.0
V5_HAWKES_WINDOW_SEC = 60
V5_SPREAD_PAST_WINDOW = (240, 60)   # (start, end) seconds before t
V5_SPREAD_RECENT_WINDOW_SEC = 60


def compute_v5_extras(swaps: pd.DataFrame, t: int,
                      entry_time: int, entry_price: float) -> dict | None:
    """4 v5 profit-protect features. Mirrors training-time builder.

    Features:
      - ps_peak_pnl_so_far: max((p_τ - entry)/entry) over [entry_time, t]
      - ps_drawdown_from_peak: current_pnl - peak_pnl (always ≤ 0)
      - mh_sell_intensity_60s: EWMA-decayed sum of large sells (≥2 SOL)
        in last 60s, decay τ=30s. Hawkes-process proxy.
      - mh_spread_change_180s: ((HL_recent - HL_past) / HL_past) where
        HL = (high - low) / median over the window. Negative = spread
        collapsing = pump termination signal.

    All features use only swap data with block_time ≤ t (no leakage).
    Returns None if insufficient data.
    """
    if (entry_price <= 0 or not np.isfinite(entry_price)
            or t <= entry_time):
        return None

    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    vol = swaps["sol_amount"].to_numpy()

    # Post-entry slice (entry_time ≤ block_time ≤ t)
    post_mask = (bt >= entry_time) & (bt <= t)
    if post_mask.sum() < 2:
        return None
    post_px = px[post_mask]
    valid_px = post_px[(post_px > 0) & np.isfinite(post_px)]
    if len(valid_px) < 2:
        return None

    peak_price = float(valid_px.max())
    current_price = float(valid_px[-1])
    if peak_price <= 0 or current_price <= 0:
        return None
    peak_pnl = (peak_price - entry_price) / entry_price
    cur_pnl = (current_price - entry_price) / entry_price
    if not (np.isfinite(peak_pnl) and np.isfinite(cur_pnl)):
        return None
    peak_pnl = float(np.clip(peak_pnl, -1.0, 50.0))  # safety cap (fat tail)
    cur_pnl = float(np.clip(cur_pnl, -1.0, 50.0))
    drawdown_from_peak = cur_pnl - peak_pnl  # ≤ 0 by construction

    # mh_sell_intensity_60s — EWMA over last 60s for large sells
    is_buy = swaps["is_buy"].to_numpy().astype(int)
    win_mask = (bt > t - V5_HAWKES_WINDOW_SEC) & (bt <= t)
    win_bt = bt[win_mask]
    win_vol = vol[win_mask]
    win_buy = is_buy[win_mask]
    sell_mask = (win_buy == 0) & (win_vol >= V5_LARGE_SELL_SOL)
    if sell_mask.sum() > 0:
        decay_weights = np.exp(-(t - win_bt[sell_mask]) / V5_HAWKES_DECAY_SEC)
        sell_intensity = float(np.sum(decay_weights))
    else:
        sell_intensity = 0.0

    # mh_spread_change_180s — spread now vs spread 60-240s ago
    recent_mask = (bt > t - V5_SPREAD_RECENT_WINDOW_SEC) & (bt <= t)
    past_mask = ((bt > t - V5_SPREAD_PAST_WINDOW[0]) &
                 (bt <= t - V5_SPREAD_PAST_WINDOW[1]))
    recent_px = px[recent_mask]
    past_px = px[past_mask]
    recent_px = recent_px[(recent_px > 0) & np.isfinite(recent_px)]
    past_px = past_px[(past_px > 0) & np.isfinite(past_px)]
    if len(recent_px) >= 2 and len(past_px) >= 2:
        med_recent = float(np.median(recent_px))
        med_past = float(np.median(past_px))
        if med_recent > 0 and med_past > 0:
            spread_recent = (recent_px.max() - recent_px.min()) / med_recent
            spread_past = (past_px.max() - past_px.min()) / med_past
            if spread_past > 1e-9:
                spread_change = (spread_recent - spread_past) / spread_past
            else:
                spread_change = 0.0
        else:
            spread_change = 0.0
    else:
        spread_change = 0.0
    spread_change = float(np.clip(spread_change, -10.0, 10.0))

    return {
        "ps_peak_pnl_so_far": peak_pnl,
        "ps_drawdown_from_peak": drawdown_from_peak,
        "mh_sell_intensity_60s": sell_intensity,
        "mh_spread_change_180s": spread_change,
    }


def load_14y_v5_3horizon():
    """v5 profit-protect ensemble (Phase 15d). 39 features = 35 v4 + 4 v5."""
    return _load_model("swap_exit_v5_3horizon.pkl")


def load_14y_v5_2():
    """v5.2 lean profit-protect ensemble. 15 features (top by gain), trained
    on bot-policy-bootstrapped sim cohort. 22/22 pre-deploy audit PASS.

    Reference: outputs/v5_2_lean/v5_2_predeploy_audit.json
    """
    return _load_model("swap_exit_v5_2_top15.pkl")


V5_FEATURE_ORDER_NEW = [
    "ps_peak_pnl_so_far",
    "ps_drawdown_from_peak",
    "mh_sell_intensity_60s",
    "mh_spread_change_180s",
]
V5_ENSEMBLE_WEIGHTS = {"y_dd_60s": 0.5, "y_dd_120s": 0.5}
V5_DEPLOY_CUTOFF = 0.40   # placeholder — calibrated from holdout

# v5.2 deploy constants — 15 features, cutoff 0.30 (LOOCV best on 37 trades)
V5_2_FEATURE_ORDER = [
    "ps_current_pnl_pct",
    "ps_peak_pnl_so_far",
    "tb_ofi_180s",
    "sf_sell_cluster",
    "sf_price_return_60s",
    "sf_swap_density",
    "sf_swap_density_late",
    "sf_price_volatility_60s",
    "sf_large_sell_count",
    "tb_vpin_abs_180s",
    "sf_dt_from_grad",
    "tb_amihud_illiq_180s",
    "ps_hold_sec",
    "sf_buyer_repeat_rate",
    "tb_vpin_60s",
]
V5_2_ENSEMBLE_WEIGHTS = {"y_dd_60s": 0.5, "y_dd_120s": 0.5}
V5_2_DEPLOY_CUTOFF = 0.30


def predict_14y_v5_2(swaps: pd.DataFrame, t: int, grad_time: int,
                      entry_time: int, entry_price: float
                      ) -> dict[str, float] | None:
    """Score with v5.2 profit-protect ensemble. Returns dict with:
      - p_dd_60s   : calibrated P(drawdown ≥ 5pp from current in next 60s |
                                  position in profit ≥ +5%)
      - p_dd_120s  : same for 120s horizon
      - p_dd_v5    : 0.5 × p_dd_60s + 0.5 × p_dd_120s (deploy ensemble)
    Or None if features can't be computed.

    Live deployment gate: fire exit when p_dd_v5 ≥ V5_2_DEPLOY_CUTOFF (0.30).
    """
    if (entry_price <= 0 or not np.isfinite(entry_price)
            or t < entry_time):
        return None

    base_feats = compute_all_features(swaps, t, grad_time)
    if base_feats is None:
        return None
    extras = compute_v3_extra_features(swaps, t)
    if extras is None:
        return None

    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    idx = int(np.searchsorted(bt, t, side="right") - 1)
    if idx < 0 or px[idx] <= 0 or not np.isfinite(px[idx]):
        return None
    price_t = float(px[idx])
    hold_sec = int(t - entry_time)
    ps = compute_v4_position_state(price_t, entry_price, hold_sec)
    if ps is None:
        return None

    v5e = compute_v5_extras(swaps, t, entry_time, entry_price)
    if v5e is None:
        return None

    all_feats = {**base_feats, **extras, **ps, **v5e}
    try:
        X = np.array([[all_feats[f] for f in V5_2_FEATURE_ORDER]], dtype=float)
    except KeyError as e:
        return None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    payload = load_14y_v5_2()
    p60 = payload["winners"]["y_dd_60s"]["model"].predict_proba(X)[:, 1]
    p120 = payload["winners"]["y_dd_120s"]["model"].predict_proba(X)[:, 1]
    p60 = float(payload["calibrators"]["y_dd_60s"].transform(p60)[0])
    p120 = float(payload["calibrators"]["y_dd_120s"].transform(p120)[0])
    return {
        "p_dd_60s": p60,
        "p_dd_120s": p120,
        "p_dd_v5": 0.5 * p60 + 0.5 * p120,
    }


# ─────────────────────── v5.3 (Phase 15e, 2026-04-29) ───────────────────────
# Retrained on 1.57M rows from the historical BigWinner + V-shape entry distribution.
# Bot policy in training matches CURRENT live (trail_020/drop_010/sl_030/TL_900).
# 22/22 pre-deploy audit PASS.

def load_14y_v5_3():
    """v5.3 lean profit-protect ensemble. Trained Phase 15e."""
    return _load_model("swap_exit_v5_3_top15.pkl")


V5_3_FEATURE_ORDER = [
    "ps_current_pnl_pct",
    "sf_swap_density_late",
    "sf_large_sell_count",
    "sf_swap_density",
    "sf_sell_cluster",
    "sf_price_volatility_60s",
    "sf_price_return_60s",
    "hc_n_net_long_t",
    "tb_amihud_illiq_180s",
    "tb_vpin_abs_180s",
    "ps_hold_sec",
    "ps_peak_pnl_so_far",
    "hc_top3_share_t",
    "tb_vpin_60s",
    "hc_hhi_t",
]
V5_3_ENSEMBLE_WEIGHTS = {"y_dd_60s": 0.5, "y_dd_120s": 0.5}
V5_3_DEPLOY_CUTOFF = 0.50


def predict_14y_v5_3(swaps: pd.DataFrame, t: int, grad_time: int,
                      entry_time: int, entry_price: float
                      ) -> dict[str, float] | None:
    """Score with v5.3 profit-protect ensemble. Returns:
      p_dd_60s, p_dd_120s, p_dd_v5_3 (= 0.5*p60 + 0.5*p120) or None.

    Live gate: fire exit when p_dd_v5_3 ≥ V5_3_DEPLOY_CUTOFF (0.50).

    NOTE bundle structure: `winners[label]` is the model object directly
    (not nested in a dict like v5.2). See train_v5_3_joint_sweep.py.
    """
    if (entry_price <= 0 or not np.isfinite(entry_price)
            or t < entry_time):
        return None

    base_feats = compute_all_features(swaps, t, grad_time)
    if base_feats is None:
        return None
    extras = compute_v3_extra_features(swaps, t)
    if extras is None:
        return None

    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    idx = int(np.searchsorted(bt, t, side="right") - 1)
    if idx < 0 or px[idx] <= 0 or not np.isfinite(px[idx]):
        return None
    price_t = float(px[idx])
    hold_sec = int(t - entry_time)
    ps = compute_v4_position_state(price_t, entry_price, hold_sec)
    if ps is None:
        return None

    v5e = compute_v5_extras(swaps, t, entry_time, entry_price)
    if v5e is None:
        return None

    all_feats = {**base_feats, **extras, **ps, **v5e}
    try:
        X = np.array([[all_feats[f] for f in V5_3_FEATURE_ORDER]], dtype=float)
    except KeyError:
        return None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    payload = load_14y_v5_3()
    # v5.3 bundle: winners[label] is model directly (not dict)
    p60_raw = payload["winners"]["y_dd_60s"].predict_proba(X)[:, 1]
    p120_raw = payload["winners"]["y_dd_120s"].predict_proba(X)[:, 1]
    p60 = float(payload["calibrators"]["y_dd_60s"].transform(p60_raw)[0])
    p120 = float(payload["calibrators"]["y_dd_120s"].transform(p120_raw)[0])
    return {
        "p_dd_60s": p60,
        "p_dd_120s": p120,
        "p_dd_v5_3": 0.5 * p60 + 0.5 * p120,
    }


def _v4_score_horizon(payload: dict, X: np.ndarray, label: str) -> float:
    """Apply winner model + calibrator for one horizon. Returns calibrated
    probability ∈ [0, 1].
    """
    w = payload["winners"][label]
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if w["type"] == "logreg":
        Xs = w["scaler"].transform(Xc)
        raw = w["model"].predict_proba(Xs)[:, 1]
    else:  # xgboost
        raw = w["model"].predict_proba(Xc)[:, 1]
    cal = payload.get("calibrators", {}).get(label)
    return float(cal.transform(raw)[0]) if cal is not None else float(raw[0])


def predict_14y_v4(swaps: pd.DataFrame, t: int, grad_time: int,
                    entry_time: int, entry_price: float
                    ) -> dict[str, float] | None:
    """Score with v4 position-conditional ensemble. Returns dict with:
      - p_y_60s   : calibrated P(SL/EC fires in next 60s)
      - p_y_120s  : calibrated P(SL/EC fires in next 120s)
      - p_rug_v4  : 0.6 × p_y_60s + 0.4 × p_y_120s (deploy ensemble)
    Or None if features can't be computed.

    `entry_time` and `entry_price` are the bot's actual entry — required for
    position-state features. At inference, the controller passes its own state.
    """
    if (entry_price <= 0 or not np.isfinite(entry_price)
            or t < entry_time):
        return None

    base_feats = compute_all_features(swaps, t, grad_time)
    if base_feats is None:
        return None
    extras = compute_v3_extra_features(swaps, t)
    if extras is None:
        return None

    # Latest mid price ≤ t for position state
    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    idx = int(np.searchsorted(bt, t, side="right") - 1)
    if idx < 0 or px[idx] <= 0 or not np.isfinite(px[idx]):
        return None
    price_t = float(px[idx])

    hold_sec = int(t - entry_time)
    ps = compute_v4_position_state(price_t, entry_price, hold_sec)
    if ps is None:
        return None

    all_feats = {**base_feats, **extras, **ps}

    try:
        m_data = load_14y_v4_3horizon()
    except Exception:
        return None

    feat_cols = m_data.get("feature_cols", V4_FEATURE_ORDER)
    x = np.asarray([[float(all_feats.get(f, 0.0)) for f in feat_cols]])

    out = {}
    for label, w in V4_ENSEMBLE_WEIGHTS.items():
        out[f"p_{label}"] = _v4_score_horizon(m_data, x, label)
    out["p_rug_v4"] = sum(out[f"p_{lbl}"] * w for lbl, w in V4_ENSEMBLE_WEIGHTS.items())
    return out


# ────────────────────── v5.5.2 (Phase 15h, 2026-05-11) ──────────────────────
# Sandwich-spike-filtered retrain after first v5_5_softexit (DBaby case
# -$1.12 on raw_pnl=+204% spike from one 0.003 SOL tx). v5.5.1's
# compute_v5_5_features read last raw swap price → vulnerable to sandwich
# spikes lifting price 3-5× for <1s. 6/12 features affected.
#
# Fix: rolling-median outlier reject (window=30, max_ratio=3.0) applied to
# raw swap stream BEFORE feature computation. 1.22% of swaps dropped.
#
# Audit (Phase 3): Path C 26/28 PASS, N2 cutoff stability NOW PASSES (was
# FAIL 0.086 → PASS 0.017). Sandwich filter resolved the structural
# sim/real cohort drift. New minor FAILs: N1 anchor variance (0.067, noise)
# + M2 shuffle baseline (0.441, unlucky shuffle draw, not leakage).
# Reference: reports/04_exit_model/Phase_15h_V5_5_2_Retrain_Verdict_2026-05-11.md
#
# v5.5.1 kept below for reference / rollback. v5.5.2 supersedes for production.

# Sandwich filter params — MUST match training (build_v5_5_2_panel.py)
V5_5_SANDWICH_FILTER_WINDOW = 30
V5_5_SANDWICH_FILTER_MAX_RATIO = 3.0


def filter_sandwich_spikes_v5_5(bt: np.ndarray, px: np.ndarray,
                                 vol: np.ndarray, isb: np.ndarray,
                                 window: int = V5_5_SANDWICH_FILTER_WINDOW,
                                 max_ratio: float = V5_5_SANDWICH_FILTER_MAX_RATIO):
    """Drop sandwich-spike swaps via trailing rolling-median outlier reject.

    Byte-parity twin of build_v5_5_2_panel.py:filter_sandwich_spikes.
    Returns filtered (bt, px, vol, isb) arrays. n_dropped tracked but not returned.

    For each swap, compute trailing rolling median over last N valid swaps.
    Drop swap if px not in [median/max_ratio, median*max_ratio].
    First 4 swaps kept regardless (insufficient history).
    """
    n = len(bt)
    if n < 5:
        return bt, px, vol, isb

    valid = (px > 0) & np.isfinite(px)
    keep = np.zeros(n, dtype=bool)
    px_series = pd.Series(np.where(valid, px, np.nan))
    rolling_median = px_series.rolling(window=window, min_periods=5).median()
    for i in range(n):
        if not valid[i]:
            continue
        med = rolling_median.iloc[i]
        if pd.isna(med) or med <= 0:
            keep[i] = True
            continue
        ratio = px[i] / med
        if (1.0 / max_ratio) <= ratio <= max_ratio:
            keep[i] = True
    return bt[keep], px[keep], vol[keep], isb[keep]


# v5.5.1 (LEGACY — kept for rollback, no longer the active production model):
V5_5_FEATURE_ORDER = [
    "ps_peak_pnl_so_far",
    "ps_hold_sec",
    "sf_price_volatility_60s",
    "sf_price_return_60s",
    "sf_swap_rate_180s",
    "sf_swap_rate_late_60s",
    "sf_large_sell_share_180s",
    "sf_sell_share_60s",
    "tb_amihud_illiq_180s",
    "tb_vpin_abs_180s",
    "tb_vpin_60s",
    "price_drawdown_60s",
]
V5_5_DEPLOY_CUTOFF = 0.60
V5_5_PROFIT_GATE = 0.05            # only fire when cur_pnl ≥ +5%
V5_5_LARGE_SELL_SOL = 1.0          # ≥1 SOL counts as "large sell" (training-aligned)


def compute_v5_5_raw_cur_pnl(swaps: pd.DataFrame, t: float,
                              entry_time: float, entry_price: float) -> float | None:
    """Returns raw (unclipped) current-PnL ratio.

    Byte-parity twin of training-time `compute_raw_cur_pnl` in
    scripts/build_v5_5_panel.py. Used for the >=5% profit precondition.
    Returns None if insufficient swaps in [entry_time, t].
    """
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    valid_window = (bt >= entry_time) & (bt <= t)
    if valid_window.sum() < 3:
        return None
    vpx = px[valid_window]
    vpx = vpx[(vpx > 0) & np.isfinite(vpx)]
    if len(vpx) < 3:
        return None
    return (float(vpx[-1]) - entry_price) / entry_price


def compute_v5_5_features(swaps: pd.DataFrame, t: float,
                           entry_time: float, entry_price: float) -> dict | None:
    """12 drift-aware v5.5 features (byte-parity twin of training-time
    `compute_v5_5_features` in scripts/build_v5_5_panel.py).

    Note: training panel computes 13 raw features (including ps_current_pnl_pct);
    the Path C 12-feature model drops ps_current_pnl_pct. We do NOT compute it
    here — the 12 listed in V5_5_FEATURE_ORDER are exactly what the model uses.

    Returns None if insufficient swap coverage.
    """
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    vol = swaps["sol_amount"].to_numpy()
    isb = swaps["is_buy"].to_numpy().astype(int)

    if len(bt) < 5:
        return None
    valid_window = (bt >= entry_time) & (bt <= t)
    if valid_window.sum() < 3:
        return None
    valid_px_window = px[valid_window]
    valid_px = valid_px_window[(valid_px_window > 0) & np.isfinite(valid_px_window)]
    if len(valid_px) < 3:
        return None

    out: dict = {}

    # ── Position state (clip range mirrors training) ──
    pnl_running = (valid_px - entry_price) / entry_price
    peak_running = float(np.maximum.accumulate(pnl_running)[-1])
    out["ps_peak_pnl_so_far"] = float(np.clip(peak_running, -1.0, 5.0))
    out["ps_hold_sec"] = float(t - entry_time)

    # ── 60s price-based features ──
    mask_60 = (bt >= t - 60) & (bt <= t)
    sub_px_60_raw = px[mask_60]
    sub_px_60 = sub_px_60_raw[(sub_px_60_raw > 0) & np.isfinite(sub_px_60_raw)]
    if len(sub_px_60) >= 3:
        rets = np.diff(sub_px_60) / np.maximum(sub_px_60[:-1], 1e-12)
        out["sf_price_volatility_60s"] = float(np.std(rets))
        out["sf_price_return_60s"] = float((sub_px_60[-1] - sub_px_60[0])
                                            / max(sub_px_60[0], 1e-12))
    else:
        out["sf_price_volatility_60s"] = 0.0
        out["sf_price_return_60s"] = 0.0

    # ── Rate-based features ──
    mask_180 = (bt >= t - 180) & (bt <= t)
    n_180 = int(mask_180.sum())
    n_60 = int(mask_60.sum())
    out["sf_swap_rate_180s"] = float(n_180 / 180.0)
    out["sf_swap_rate_late_60s"] = float(n_60 / 60.0)
    n_large_sell = int((mask_180 & (isb == 0) & (vol >= V5_5_LARGE_SELL_SOL)).sum())
    out["sf_large_sell_share_180s"] = float(n_large_sell / max(n_180, 1))
    n_sell_60 = int((mask_60 & (isb == 0)).sum())
    out["sf_sell_share_60s"] = float(n_sell_60 / max(n_60, 1))

    # ── Tier-B microstructure ──
    if n_180 >= 3:
        sub_px = px[mask_180]
        sub_vol = vol[mask_180]
        valid = (sub_px > 0) & np.isfinite(sub_px) & (sub_vol > 0)
        if valid.sum() >= 3:
            sub_px_v = sub_px[valid]
            sub_vol_v = sub_vol[valid]
            rets_abs = np.abs(np.diff(sub_px_v) / np.maximum(sub_px_v[:-1], 1e-12))
            out["tb_amihud_illiq_180s"] = float(
                np.mean(rets_abs / np.maximum(sub_vol_v[1:], 1e-9))
            )
        else:
            out["tb_amihud_illiq_180s"] = 0.0
    else:
        out["tb_amihud_illiq_180s"] = 0.0

    for w_sec, key in [(60, "tb_vpin_60s"), (180, "tb_vpin_abs_180s")]:
        m = (bt >= t - w_sec) & (bt <= t)
        if m.sum() >= 3:
            buy_v = float(vol[m & (isb == 1)].sum())
            sell_v = float(vol[m & (isb == 0)].sum())
            out[key] = float(abs(buy_v - sell_v) / max(buy_v + sell_v, 1e-9))
        else:
            out[key] = 0.0

    # ── Drawdown in last 60s ──
    if len(sub_px_60) >= 3:
        peak = np.maximum.accumulate(sub_px_60)
        dd = (peak - sub_px_60) / np.maximum(peak, 1e-12)
        out["price_drawdown_60s"] = float(dd.max())
    else:
        out["price_drawdown_60s"] = 0.0

    return out


def load_14y_v5_5():
    """v5.5.1 12-feature Chainstack-native exit model (Phase 15g, 2026-05-11)."""
    return _load_model("swap_exit_v5_5_1_12feature.pkl")


def predict_14y_v5_5(swaps: pd.DataFrame, t: float, grad_time: float,
                      entry_time: float, entry_price: float
                      ) -> dict[str, float] | None:
    """Score with v5.5.1 12-feature Chainstack-native model. Returns dict:
      - p_dd_v5_5    : XGBoost P(drawdown ≥ 5pp in next 60s)
      - raw_cur_pnl  : unclipped current PnL for the ≥5% precondition
      - features     : 12-feature dict (for logging/audit)
    Or None if features can't be computed.

    Live gate: fire exit when (raw_cur_pnl ≥ V5_5_PROFIT_GATE)
                          AND (p_dd_v5_5 ≥ V5_5_DEPLOY_CUTOFF).
    Note: cutoff is Day-7 recalibrated on live distribution.
    """
    if (entry_price <= 0 or not np.isfinite(entry_price) or t < entry_time):
        return None

    feats = compute_v5_5_features(swaps, t, entry_time, entry_price)
    if feats is None:
        return None

    raw_pnl = compute_v5_5_raw_cur_pnl(swaps, t, entry_time, entry_price)
    if raw_pnl is None:
        return None

    try:
        payload = load_14y_v5_5()
    except Exception:
        return None

    # BUGFIX (Phase 15j 2026-05-11): apply training-time outlier clips
    # (v5.5.1 trained with same clip_outliers as v5.5.2/3). Helper defined below.
    feats = _apply_training_outlier_clips(feats, payload)

    feat_cols = payload.get("feature_cols", V5_5_FEATURE_ORDER)
    X = np.array([[feats[f] for f in feat_cols]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = payload["model"]
    p = float(model.predict_proba(X)[:, 1][0])
    return {
        "p_dd_v5_5": p,
        "raw_cur_pnl": float(raw_pnl),
        "features": feats,
    }


# ────────────────────── v5.5.2 PRODUCTION (Phase 15h, 2026-05-11) ──────────────────────
# Reuses V5_5_FEATURE_ORDER (same 12 features). The difference is the swap
# stream is sandwich-filtered before feature compute → 6/12 vulnerable
# features now spike-clean.

V5_5_2_DEPLOY_CUTOFF = 0.50         # Path C training optimum @ realistic PnL (was 0.60 in v5.5.1)
V5_5_2_PROFIT_GATE = 0.05           # unchanged
V5_5_FEATURE_ORDER_V2 = V5_5_FEATURE_ORDER  # alias for clarity


def _apply_v5_5_filter(swaps: pd.DataFrame) -> pd.DataFrame:
    """Apply sandwich filter to swap stream. Returns FILTERED DataFrame
    (same columns, fewer rows). Idempotent if applied twice (filter
    won't drop more on already-clean data).
    """
    if swaps is None or len(swaps) == 0:
        return swaps
    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    vol = swaps["sol_amount"].to_numpy()
    isb = swaps["is_buy"].to_numpy().astype(int)
    bt_f, px_f, vol_f, isb_f = filter_sandwich_spikes_v5_5(bt, px, vol, isb)
    # Rebuild DataFrame preserving original column order
    return pd.DataFrame({
        "block_time": bt_f,
        "effective_price_sol": px_f,
        "sol_amount": vol_f,
        "is_buy": isb_f,
    })


def compute_v5_5_2_raw_cur_pnl(swaps: pd.DataFrame, t: float,
                                entry_time: float, entry_price: float) -> float | None:
    """v5.5.2 raw_cur_pnl — applies sandwich filter before computation."""
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    filt = _apply_v5_5_filter(swaps)
    return compute_v5_5_raw_cur_pnl(filt, t, entry_time, entry_price)


def compute_v5_5_2_features(swaps: pd.DataFrame, t: float,
                             entry_time: float, entry_price: float) -> dict | None:
    """v5.5.2 features — applies sandwich filter before computation.
    Byte-parity twin of build_v5_5_2_panel.py:compute_v5_5_features (with
    sandwich filter applied at swap-stream level upstream)."""
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    filt = _apply_v5_5_filter(swaps)
    return compute_v5_5_features(filt, t, entry_time, entry_price)


def load_14y_v5_5_2():
    """v5.5.2 sandwich-filtered 12-feature Chainstack-native exit model."""
    return _load_model("swap_exit_v5_5_2_12feature.pkl")


def _apply_training_outlier_clips(feats: dict, payload: dict) -> dict:
    """Apply training-time outlier clips (upper bound) read from pkl audit_modifications.

    BUGFIX (Phase 15j, 2026-05-11): training script `train_v5_5_2_path_BC.py` (and
    v5.5.3) clipped 3 features at train-set p99 BEFORE fitting the model:
      sf_price_volatility_60s, sf_price_return_60s, tb_amihud_illiq_180s
    These thresholds are saved in `payload["audit_modifications"]["outlier_clip_thresholds"]`.
    Production must apply the same clip so the model sees features in its training range.

    Without this, ~1% of production rows fed feature values above train-set p99 to a
    model that never saw those values during fit. XGBoost trees route out-of-distribution
    inputs to the rightmost branch — not catastrophic but introduces unknown drift.

    Returns a new dict with clipped values; original `feats` is mutated for caller use.
    """
    clips = (payload.get("audit_modifications") or {}).get("outlier_clip_thresholds") or {}
    for feat_name, upper in clips.items():
        if feat_name in feats and feats[feat_name] is not None:
            try:
                feats[feat_name] = float(min(feats[feat_name], float(upper)))
            except (TypeError, ValueError):
                pass
    return feats


def predict_14y_v5_5_2(swaps: pd.DataFrame, t: float, grad_time: float,
                        entry_time: float, entry_price: float
                        ) -> dict[str, float] | None:
    """Score with v5.5.2 sandwich-filtered model. Same API as predict_14y_v5_5.
    Returns:
      - p_dd_v5_5    : XGBoost P(drawdown ≥ 5pp in next 60s)  [key kept for log compat]
      - raw_cur_pnl  : unclipped current PnL  [for ≥5% precondition]
      - features     : 12-feature dict (post training-clip; matches what model scored)
      - n_swaps_in / n_swaps_filtered: for observability (filter drop rate)
    Live gate: fire when (raw_cur_pnl ≥ V5_5_2_PROFIT_GATE) AND
               (p_dd_v5_5 ≥ V5_5_2_DEPLOY_CUTOFF).
    """
    if (entry_price <= 0 or not np.isfinite(entry_price) or t < entry_time):
        return None

    n_in = len(swaps)
    filt = _apply_v5_5_filter(swaps)
    n_out = len(filt)

    feats = compute_v5_5_features(filt, t, entry_time, entry_price)
    if feats is None:
        return None

    raw_pnl = compute_v5_5_raw_cur_pnl(filt, t, entry_time, entry_price)
    if raw_pnl is None:
        return None

    try:
        payload = load_14y_v5_5_2()
    except Exception:
        return None

    # BUGFIX (Phase 15j 2026-05-11): apply training-time outlier clips
    feats = _apply_training_outlier_clips(feats, payload)

    feat_cols = payload.get("feature_cols", V5_5_FEATURE_ORDER)
    X = np.array([[feats[f] for f in feat_cols]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = payload["model"]
    p = float(model.predict_proba(X)[:, 1][0])
    return {
        "p_dd_v5_5": p,
        "raw_cur_pnl": float(raw_pnl),
        "features": feats,
        "n_swaps_in": int(n_in),
        "n_swaps_filtered": int(n_out),
    }


# ────────────────── v5.5.3 PRODUCTION (Phase 15i, 2026-05-11) ──────────────────
# Inherits V5.5.2 sandwich filter + adds 3 feature fragility fixes identified in
# audit `audits/2026-05-11_v5_5_2_score_bias_replay_results.md`:
#   - tb_amihud_illiq_180s: per-swap clip [0, 100] before mean
#     (v5.5.2 training panel: max=10.78M, p999=4.36M, driven by dust swap vol≈1e-9)
#   - sf_price_return_60s: output clip [-1, 5] (match peak_pnl convention)
#     (v5.5.2 training panel: max=4265 = 426,500% 60s return)
#   - sf_price_volatility_60s: per-return clip [-5, 5] before std
#     (v5.5.2 training panel: max=1086, driven by div-by-near-zero per-return)
#
# Constants MUST match `scripts/build_v5_5_3_panel.py` byte-for-byte (training).
# Sandwich filter unchanged from V5.5.2.

V5_5_3_AMIHUD_PER_SWAP_CAP = 100.0
V5_5_3_RETURN_CLIP_LO = -1.0
V5_5_3_RETURN_CLIP_HI = 5.0
V5_5_3_RET_PERSWAP_CLIP = 5.0

V5_5_3_DEPLOY_CUTOFF = 0.50          # placeholder — re-tune after Path C sweep on retrained model
V5_5_3_PROFIT_GATE = 0.05            # unchanged from v5.5.2


def compute_v5_5_3_features_raw(swaps: pd.DataFrame, t: float,
                                  entry_time: float, entry_price: float) -> dict | None:
    """v5.5.3 features (clipped) — operates on already-sandwich-filtered swaps.

    Byte-parity twin of `scripts/build_v5_5_3_panel.py:compute_v5_5_features`.
    """
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    bt = swaps["block_time"].to_numpy()
    px = swaps["effective_price_sol"].to_numpy()
    vol = swaps["sol_amount"].to_numpy()
    isb = swaps["is_buy"].to_numpy().astype(int)

    if len(bt) < 5:
        return None
    valid_window = (bt >= entry_time) & (bt <= t)
    if valid_window.sum() < 3:
        return None
    valid_px_window = px[valid_window]
    valid_px = valid_px_window[(valid_px_window > 0) & np.isfinite(valid_px_window)]
    if len(valid_px) < 3:
        return None

    out: dict = {}
    pnl_running = (valid_px - entry_price) / entry_price
    peak_running = float(np.maximum.accumulate(pnl_running)[-1])
    out["ps_peak_pnl_so_far"] = float(np.clip(peak_running, -1.0, 5.0))
    out["ps_hold_sec"] = float(t - entry_time)

    mask_60 = (bt >= t - 60) & (bt <= t)
    sub_px_60_raw = px[mask_60]
    sub_px_60 = sub_px_60_raw[(sub_px_60_raw > 0) & np.isfinite(sub_px_60_raw)]
    if len(sub_px_60) >= 3:
        # v5.5.3 FIX: per-return clip before std
        rets = np.diff(sub_px_60) / np.maximum(sub_px_60[:-1], 1e-12)
        rets_clipped = np.clip(rets, -V5_5_3_RET_PERSWAP_CLIP, V5_5_3_RET_PERSWAP_CLIP)
        out["sf_price_volatility_60s"] = float(np.std(rets_clipped))
        # v5.5.3 FIX: output clip [-1, 5]
        ret_60s = (sub_px_60[-1] - sub_px_60[0]) / max(sub_px_60[0], 1e-12)
        out["sf_price_return_60s"] = float(np.clip(ret_60s, V5_5_3_RETURN_CLIP_LO, V5_5_3_RETURN_CLIP_HI))
    else:
        out["sf_price_volatility_60s"] = 0.0
        out["sf_price_return_60s"] = 0.0

    mask_180 = (bt >= t - 180) & (bt <= t)
    n_180 = int(mask_180.sum())
    n_60 = int(mask_60.sum())
    out["sf_swap_rate_180s"] = float(n_180 / 180.0)
    out["sf_swap_rate_late_60s"] = float(n_60 / 60.0)
    n_large_sell = int((mask_180 & (isb == 0) & (vol >= V5_5_LARGE_SELL_SOL)).sum())
    out["sf_large_sell_share_180s"] = float(n_large_sell / max(n_180, 1))
    n_sell_60 = int((mask_60 & (isb == 0)).sum())
    out["sf_sell_share_60s"] = float(n_sell_60 / max(n_60, 1))

    if n_180 >= 3:
        sub_px = px[mask_180]
        sub_vol = vol[mask_180]
        valid = (sub_px > 0) & np.isfinite(sub_px) & (sub_vol > 0)
        if valid.sum() >= 3:
            sub_px_v = sub_px[valid]
            sub_vol_v = sub_vol[valid]
            rets_abs = np.abs(np.diff(sub_px_v) / np.maximum(sub_px_v[:-1], 1e-12))
            # v5.5.3 FIX: per-swap clip before mean
            per_swap = rets_abs / np.maximum(sub_vol_v[1:], 1e-9)
            out["tb_amihud_illiq_180s"] = float(np.mean(np.clip(per_swap, 0, V5_5_3_AMIHUD_PER_SWAP_CAP)))
        else:
            out["tb_amihud_illiq_180s"] = 0.0
    else:
        out["tb_amihud_illiq_180s"] = 0.0

    for w_sec, key in [(60, "tb_vpin_60s"), (180, "tb_vpin_abs_180s")]:
        m = (bt >= t - w_sec) & (bt <= t)
        if m.sum() >= 3:
            buy_v = float(vol[m & (isb == 1)].sum())
            sell_v = float(vol[m & (isb == 0)].sum())
            out[key] = float(abs(buy_v - sell_v) / max(buy_v + sell_v, 1e-9))
        else:
            out[key] = 0.0

    if len(sub_px_60) >= 3:
        peak = np.maximum.accumulate(sub_px_60)
        dd = (peak - sub_px_60) / np.maximum(peak, 1e-12)
        out["price_drawdown_60s"] = float(dd.max())
    else:
        out["price_drawdown_60s"] = 0.0

    return out


def compute_v5_5_3_features(swaps: pd.DataFrame, t: float,
                             entry_time: float, entry_price: float) -> dict | None:
    """v5.5.3 features — applies sandwich filter (V5.5.2 same) before fragility-fixed compute."""
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    filt = _apply_v5_5_filter(swaps)
    return compute_v5_5_3_features_raw(filt, t, entry_time, entry_price)


def compute_v5_5_3_raw_cur_pnl(swaps: pd.DataFrame, t: float,
                                entry_time: float, entry_price: float) -> float | None:
    """v5.5.3 raw_cur_pnl — applies sandwich filter (same as v5.5.2)."""
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    filt = _apply_v5_5_filter(swaps)
    return compute_v5_5_raw_cur_pnl(filt, t, entry_time, entry_price)


def load_14y_v5_5_3():
    """v5.5.3 sandwich-filtered + fragility-clipped 12-feature exit model."""
    return _load_model("swap_exit_v5_5_3_12feature.pkl")


def predict_14y_v5_5_3(swaps: pd.DataFrame, t: float, grad_time: float,
                        entry_time: float, entry_price: float
                        ) -> dict | None:
    """Score with v5.5.3 sandwich-filtered + fragility-clipped model.
    API matches predict_14y_v5_5_2.

    Live gate: fire when (raw_cur_pnl ≥ V5_5_3_PROFIT_GATE) AND
               (p_dd_v5_5 ≥ V5_5_3_DEPLOY_CUTOFF).
    """
    if (entry_price <= 0 or not np.isfinite(entry_price) or t < entry_time):
        return None

    n_in = len(swaps)
    filt = _apply_v5_5_filter(swaps)
    n_out = len(filt)

    feats = compute_v5_5_3_features_raw(filt, t, entry_time, entry_price)
    if feats is None:
        return None

    raw_pnl = compute_v5_5_raw_cur_pnl(filt, t, entry_time, entry_price)
    if raw_pnl is None:
        return None

    try:
        payload = load_14y_v5_5_3()
    except Exception:
        return None

    # BUGFIX (Phase 15j 2026-05-11): apply training-time outlier clips
    feats = _apply_training_outlier_clips(feats, payload)

    feat_cols = payload.get("feature_cols", V5_5_FEATURE_ORDER)
    X = np.array([[feats[f] for f in feat_cols]], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model = payload["model"]
    p = float(model.predict_proba(X)[:, 1][0])
    return {
        "p_dd_v5_5": p,
        "raw_cur_pnl": float(raw_pnl),
        "features": feats,
        "n_swaps_in": int(n_in),
        "n_swaps_filtered": int(n_out),
    }
