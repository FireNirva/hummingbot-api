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

    pre_t = swaps.iloc[:pre_t_mask.sum()]  # sorted, so first N where bt ≤ t

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
    return _load_model("swap_exit_tier_b_model.pkl")


def load_f2a_hc():
    return _load_model("swap_exit_f2a_hc_model.pkl")


def load_f2a_hc_live_v1():
    """Live-schema retrain (2026-04-24). Trained on bot `swaps` table with
    identical feature extractor → zero training/deployment schema shift.
    Runs in parallel with Birdeye-trained model for 2-week A/B shadow;
    replacement decision in §9.9 of Phase_14y_Enhancement_FROZEN_2026-04-24.md.
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

    pre_t = swaps.iloc[:pre_t_mask.sum()]  # relies on pre-sorted
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
