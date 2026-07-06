"""
VWMP — Volume-Weighted Median Price computation for phantom-resistant
price source.

Replaces raw gRPC pool snapshot as authoritative current_price source for
exit decisions (TP / SL / trail). Resistant to single-tick MEV/sandwich
phantoms that account for 0.00-0.03% of pool volume but 100% of pre-fix
phantom TP fires.

Spec: research_notebooks/meme_sniper/vwmp_price_defense/2026-05-15_vwmp_price_source_SPEC.md

Algorithm: take recent swaps in a rolling time window, drop entries below
a volume floor, sort by price, return the price at the 50% cumulative-
volume position. Median + volume weighting filters outlier prints
regardless of direction.

Default parameters (per spec §10):
    window_s   = 5.0      # rolling window
    vol_gate   = 0.01 SOL # min size to count (filters database-noise dust)
    min_swaps  = 3        # below this, returns None (caller falls back)
"""
from collections import deque
from typing import Iterable, Optional, Tuple
import time

# A swap entry: (timestamp_unix_sec, price_sol, volume_sol)
SwapEntry = Tuple[float, float, float]


def compute_vwmp(
    swap_buffer: Iterable[SwapEntry],
    vol_gate: float = 0.01,
    window_s: float = 5.0,
    min_swaps: int = 3,
    now: Optional[float] = None,
) -> Optional[float]:
    """
    Compute volume-weighted median price over a rolling window of swaps.

    Args:
        swap_buffer: iterable of (timestamp, price_sol, volume_sol).
            Order doesn't matter; function filters and sorts internally.
        vol_gate: minimum swap volume (SOL) to include. Filters absolute
            dust (zero-volume DB-noise rows). Phantom-defense itself
            comes from the median statistic, NOT this gate.
        window_s: rolling window in seconds; swaps older than
            (now - window_s) are ignored.
        min_swaps: minimum number of qualifying swaps required. If
            fewer survive the filters, returns None (caller should fall
            back to a legacy price source).
        now: reference timestamp; defaults to time.time(). Pass an
            explicit value for reproducible tests.

    Returns:
        VWMP price (float) if enough qualifying swaps; None otherwise.

    Invariants:
        - Returns are always strictly positive when not None.
        - Function is pure (no side effects on inputs).
        - O(N log N) in the number of qualifying swaps.
    """
    if now is None:
        now = time.time()
    cutoff = now - window_s

    # Snapshot first: producer thread may append to a deque concurrently.
    # In CPython, list(deque) is GIL-atomic for individual operations,
    # giving us a consistent iteration target. See spec §3.3.
    buffer_snapshot = list(swap_buffer)

    # Filter: time window + volume floor + strictly positive vol & price.
    # - `t is not None` and `v is not None` defends against stream bugs or
    #   DB rows where columns are NULL (would raise TypeError on numeric
    #   comparison in Python 3).
    # - `v > 0` clamp ensures negative volumes can't corrupt the median
    #   computation (cumulative sum would go non-monotonic). Also makes the
    #   function safe for vol_gate=0 (allow-all) callers.
    qualified = [
        (t, p, v) for t, p, v in buffer_snapshot
        if t is not None and p is not None and v is not None
        and t >= cutoff and v >= vol_gate and v > 0 and p > 0
    ]
    if len(qualified) < min_swaps:
        return None

    sorted_by_price = sorted(qualified, key=lambda x: x[1])
    total_vol = sum(v for _, _, v in sorted_by_price)
    if total_vol <= 0:
        return None

    half = total_vol / 2.0
    cum = 0.0
    for _, price, vol in sorted_by_price:
        cum += vol
        if cum >= half:
            return price
    # Numeric edge case fallback — should never hit given total_vol > 0
    return sorted_by_price[-1][1]


def compute_vwmp_with_diagnostics(
    swap_buffer: Iterable[SwapEntry],
    vol_gate: float = 0.01,
    window_s: float = 5.0,
    min_swaps: int = 3,
    now: Optional[float] = None,
) -> Tuple[Optional[float], dict]:
    """
    Same as compute_vwmp but returns diagnostic metadata alongside.

    Used for shadow-mode logging (compare VWMP vs legacy grpc_pool) and
    for the vwmp_shadow_log table during the first 24h post-deploy.

    Returns:
        (vwmp_price_or_None, diagnostics_dict)

    Diagnostics dict keys:
        buffer_size: total entries in the input iterable
        qualified_count: entries that survived all filters
        oldest_swap_age_s: age (sec) of oldest qualifying swap (None if no
            qualifying entries)
        newest_swap_age_s: age of newest qualifying swap
        total_qualified_vol: sum of vol across qualifying swaps
        price_min / price_max: bounds across qualifying swaps
        vwmp_status: one of {'ok', 'no_data', 'insufficient_swaps',
            'zero_volume', 'fallback_last'}
    """
    if now is None:
        now = time.time()
    cutoff = now - window_s

    buffer_list = list(swap_buffer)
    # See compute_vwmp() for None/clamp rationale.
    qualified = [
        (t, p, v) for t, p, v in buffer_list
        if t is not None and p is not None and v is not None
        and t >= cutoff and v >= vol_gate and v > 0 and p > 0
    ]

    diag = {
        'buffer_size': len(buffer_list),
        'qualified_count': len(qualified),
        'oldest_swap_age_s': None,
        'newest_swap_age_s': None,
        'total_qualified_vol': 0.0,
        'price_min': None,
        'price_max': None,
        'vwmp_status': 'no_data',
    }

    if not qualified:
        return None, diag

    timestamps = [t for t, _, _ in qualified]
    prices = [p for _, p, _ in qualified]
    diag['oldest_swap_age_s'] = now - min(timestamps)
    diag['newest_swap_age_s'] = now - max(timestamps)
    diag['total_qualified_vol'] = sum(v for _, _, v in qualified)
    diag['price_min'] = min(prices)
    diag['price_max'] = max(prices)

    if len(qualified) < min_swaps:
        diag['vwmp_status'] = 'insufficient_swaps'
        return None, diag

    sorted_by_price = sorted(qualified, key=lambda x: x[1])
    total_vol = diag['total_qualified_vol']
    if total_vol <= 0:
        diag['vwmp_status'] = 'zero_volume'
        return None, diag

    half = total_vol / 2.0
    cum = 0.0
    for _, price, vol in sorted_by_price:
        cum += vol
        if cum >= half:
            diag['vwmp_status'] = 'ok'
            return price, diag

    diag['vwmp_status'] = 'fallback_last'
    return sorted_by_price[-1][1], diag


def prune_old_swaps(
    buffer: deque,
    max_age_s: float = 30.0,
    now: Optional[float] = None,
) -> int:
    """
    Pop entries from the left of a deque that are older than max_age_s.

    Assumes the deque is roughly time-ordered (oldest on the left). This
    is the natural order produced by appending swaps as they arrive via
    the gRPC stream.

    Used in the per-position swap buffer cleanup; called every M4 tick
    to keep buffer memory bounded even when stream produces > maxlen
    entries within window_s.

    Args:
        buffer: collections.deque whose entries are (timestamp, ...).
        max_age_s: prune entries with timestamp < (now - max_age_s).
        now: reference timestamp; defaults to time.time().

    Returns:
        Number of entries pruned.
    """
    if now is None:
        now = time.time()
    cutoff = now - max_age_s
    pruned = 0
    while buffer and buffer[0][0] < cutoff:
        buffer.popleft()
        pruned += 1
    return pruned
