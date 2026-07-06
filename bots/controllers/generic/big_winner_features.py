"""Phase 16.3 — big_winner v1 entry-score features.

Computes the 13 NEW features added in Phase 16:
  - Hawkes self-exciting (5)        — hk_*
  - Time-of-window (3)               — m_*
  - Tick-level distributional (5)    — td_*

Combined with 32 features from `meme_sniper_exit_models.compute_all_features`
+ `compute_v3_extra_features` (which big_winner reuses unchanged), gives the
full 38-feature input to the current `big_winner_v2_model.pkl` bundle.

Byte-parity with `scripts/phase16_1_expanded.py` is mandatory — any change
here MUST be mirrored in training pipeline (and vice versa). All formulas
ported verbatim from the training script.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Hawkes config (matches phase16_1_expanded.py)
HAWKES_DECAY_TAU_SEC = 20.0       # EWMA decay constant
HAWKES_INTENSITY_WIN_SEC = 60     # window for intensity calc
HAWKES_BRANCHING_WIN_SEC = 30     # window for branching ratio

# Tick distributional config
TICK_DIST_WIN_SEC = 180


def compute_hawkes_features(swaps_visible: pd.DataFrame, scan_t: int) -> dict:
    """5 Hawkes self-exciting intensity features.

    arXiv 2408.03594 (forecasting OFI via Hawkes). EWMA proxy with τ=20s,
    window 60s for intensity, 30s window for branching ratio.
    """
    win = swaps_visible[(swaps_visible['block_time'] >= scan_t - HAWKES_INTENSITY_WIN_SEC) &
                          (swaps_visible['block_time'] <= scan_t)]
    if len(win) == 0:
        return {f'hk_{k}': 0.0 for k in
                ['buy_intensity_60s', 'sell_intensity_60s', 'buy_sell_ratio',
                 'branching_buy', 'branching_sell']}

    bt = win['block_time'].to_numpy()
    is_buy = win['is_buy'].to_numpy().astype(bool)

    # EWMA-decayed intensities
    age = scan_t - bt
    decay = np.exp(-age / HAWKES_DECAY_TAU_SEC)
    buy_intensity = float(decay[is_buy].sum())
    sell_intensity = float(decay[~is_buy].sum())
    total = buy_intensity + sell_intensity
    ratio = buy_intensity / total if total > 0 else 0.5

    # Branching: mean number of subsequent same-side events within next 30s
    win30 = swaps_visible[(swaps_visible['block_time'] >= scan_t - HAWKES_BRANCHING_WIN_SEC) &
                            (swaps_visible['block_time'] <= scan_t)]
    bt30 = win30['block_time'].to_numpy()
    isb30 = win30['is_buy'].to_numpy().astype(bool)

    branching_buy = 0.0
    branching_sell = 0.0
    n_buys = int(isb30.sum())
    n_sells = int((~isb30).sum())
    if n_buys > 1:
        buy_times = bt30[isb30]
        counts = []
        for t_buy in buy_times:
            n_after = int(((buy_times > t_buy) &
                            (buy_times <= t_buy + HAWKES_BRANCHING_WIN_SEC)).sum())
            counts.append(n_after)
        branching_buy = float(np.mean(counts)) if counts else 0.0
    if n_sells > 1:
        sell_times = bt30[~isb30]
        counts = []
        for t_sell in sell_times:
            n_after = int(((sell_times > t_sell) &
                            (sell_times <= t_sell + HAWKES_BRANCHING_WIN_SEC)).sum())
            counts.append(n_after)
        branching_sell = float(np.mean(counts)) if counts else 0.0

    return {
        'hk_buy_intensity_60s': buy_intensity,
        'hk_sell_intensity_60s': sell_intensity,
        'hk_buy_sell_ratio': ratio,
        'hk_branching_buy': branching_buy,
        'hk_branching_sell': branching_sell,
    }


def compute_time_features(scan_t: int, grad_t: int) -> dict:
    """3 time-of-window features (QuickAdapter inspired).

    scan_t and grad_t are unix seconds. age_norm clipped [0, 1] over the
    T+0..T+10min window.
    """
    age_sec = scan_t - grad_t
    age_norm = age_sec / 600.0
    age_norm = float(np.clip(age_norm, 0.0, 1.0))
    return {
        'm_grad_age_norm': age_norm,
        'm_phase_sin': float(np.sin(2 * np.pi * age_norm)),
        'm_phase_cos': float(np.cos(2 * np.pi * age_norm)),
    }


def compute_tick_features(swaps_visible: pd.DataFrame, scan_t: int) -> dict:
    """5 tick distributional features (MemeTrans arXiv 2602.13480 inspired)."""
    win = swaps_visible[(swaps_visible['block_time'] >= scan_t - TICK_DIST_WIN_SEC) &
                          (swaps_visible['block_time'] <= scan_t)]
    if len(win) < 3:
        return {f'td_{k}': 0.0 for k in
                ['trade_size_p90', 'trade_size_skew',
                 'inter_arrival_p10', 'inter_arrival_p90',
                 'unique_trader_count']}

    sizes = win['sol_amount'].to_numpy()
    bt = np.sort(win['block_time'].to_numpy())
    inter = np.diff(bt) if len(bt) > 1 else np.array([0.0])
    inter = inter[inter > 0]

    if len(sizes) >= 3:
        s_mean = sizes.mean()
        s_std = sizes.std()
        skew = float(((sizes - s_mean) ** 3).mean() / (s_std ** 3)) if s_std > 0 else 0.0
    else:
        skew = 0.0

    return {
        'td_trade_size_p90': float(np.percentile(sizes, 90)) if len(sizes) > 0 else 0.0,
        'td_trade_size_skew': float(np.clip(skew, -10, 10)),
        'td_inter_arrival_p10': float(np.percentile(inter, 10)) if len(inter) > 0 else 0.0,
        'td_inter_arrival_p90': float(np.percentile(inter, 90)) if len(inter) > 0 else 0.0,
        'td_unique_trader_count': float(win['trader_address'].nunique()),
    }


def compute_phase16_entry_features(swaps_visible: pd.DataFrame, scan_t: int,
                                     grad_t: int) -> dict:
    """Compute the 13 NEW Phase 16 entry features (Hawkes + time + tick).

    Combine with 32 baseline features from production module to feed
    big_winner_v2_model.pkl.

    Args:
        swaps_visible: DataFrame with at least columns
            block_time, trader_address, is_buy, sol_amount.
            MUST be filtered to block_time <= scan_t (forward-only).
        scan_t: decision unix-time (T+3min..T+10min from graduation)
        grad_t: graduation unix-time

    Returns:
        dict with 13 features. Empty/all-zero if input insufficient.
    """
    return {
        **compute_hawkes_features(swaps_visible, scan_t),
        **compute_time_features(scan_t, grad_t),
        **compute_tick_features(swaps_visible, scan_t),
    }
