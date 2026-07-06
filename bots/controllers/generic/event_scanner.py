"""
Flow model — event-triggered entry scanner (Phase B research artifact).

The Flow model replaces the misnamed "V-shape event-triggered" concept.
Analysis of the top-5% first-trigger backtest showed the actual signal is
*early order flow*: 25% of predictive power comes from `m10_unique_buyers`,
and only 5% of triggers are genuine V-shape patterns (63% steady, 32% near).
"Flow" more accurately describes what the model detects.

Runs in parallel to the existing fixed T+10m V-shape gate. Does NOT influence
real trading decisions. Records hypothetical first-trigger entries to the
`shadow_event_evals` table so offline analysis can compare against live
outcomes.

Wired into MemeSniper via:
  - FlowModel — loads the regressor from Phase B Step 1
  - detect_pattern_at_t / compute_micro_at_t — time-parameterized versions
    of the live pattern / micro functions
  - _run_shadow_event (in meme_sniper.py) — tick-driven scanner

Turn on with `shadow_event_enabled: true` in yml. Default false so the live
bot is unchanged until explicitly opted in.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from meme_sniper.L4_Signal_and_Model_Inference import ShadowGuard

try:  # production: controllers.generic.ms.models (Hummingbot loads as controllers.generic.*)
    from controllers.generic.ms.models import registry as _model_registry
except ImportError:  # in-container test channel / direct file import
    from ms.models import registry as _model_registry

logger = logging.getLogger(__name__)


FEATURE_COLS_EVENT = [
    "vf_recovery_pct", "vf_entry_ret", "vf_trough_ret",
    "m10_growth_accel", "m10_unique_buyers", "m10_late_imbalance",
    "m10_buyer_hhi", "m10_seller_buyer_ratio", "m10_top1_buyer_share",
]


class FlowModel:
    """Flow model — order-flow-based time-invariant entry.

    Unlike VShapeModel (which treats pattern as the primary signal), Flow's
    dominant features are microstructure (m10_unique_buyers, growth_accel,
    imbalance). The underlying estimator is an XGBRegressor predicting
    `clip(trail_ret, -1.0, 2.0)`. Scores are continuous, not probabilities.
    Cutoffs stored in the pkl are negative values — still "top N%" semantics,
    just raw regression outputs.
    """

    def __init__(self, model_path: str, *, metrics: Optional[Any] = None):
        path = Path(model_path)
        self.guarded = _model_registry.get_or_load(path, metrics=metrics)
        artifact = self.guarded.unwrap()
        self.model = artifact["model"]
        # Patch sklearn imputer version incompatibility (same as VShapeModel)
        try:
            from sklearn.impute import SimpleImputer as _SI
            if hasattr(self.model, "steps"):
                for _, step in self.model.steps:
                    if isinstance(step, _SI) and not hasattr(step, "_fill_dtype"):
                        step._fill_dtype = step.statistics_.dtype
        except Exception:
            pass
        self.version = str(artifact.get("version", "flow_v1"))
        self.feature_names = list(artifact["feature_cols"])
        self.model_type = str(artifact.get("model_type", "regressor"))
        self.selection_cutoffs = {
            str(k): float(v) for k, v in dict(artifact.get("cutoffs", {})).items()
            if v is not None
        }
        self.scan_times_sec = list(artifact.get("scan_times_sec", [])) or list(range(180, 901, 60))
        self.swap_data_max_sec = int(artifact.get("swap_data_max_sec", 600))
        logger.info(
            "FlowModel: loaded %s (%d features, type=%s, bands=%s)",
            path, len(self.feature_names), self.model_type,
            ",".join(sorted(self.selection_cutoffs.keys())),
        )

    def predict_score(self, features: Dict[str, float]) -> float:
        X = pd.DataFrame(
            [{f: features.get(f, np.nan) for f in self.feature_names}],
            columns=self.feature_names,
        )
        if self.model_type == "regressor":
            return float(self.model.predict(X)[0])
        return float(self.model.predict_proba(X)[0][1])


def detect_pattern_at_t(
    kline_bars: List[Dict[str, Any]],
    graduation_time: float,
    eval_sec: int,
) -> Optional[Dict[str, float]]:
    """Parameterized variant of detect_vshape_live — evaluates at arbitrary
    eval_sec (seconds since graduation) instead of the hardcoded T+10m.

    Expects kline_bars from OnChainKlineBuilder.build_kline — each bar has
    keys: open, high, low, close, volume, time.
    """
    if not kline_bars or len(kline_bars) < 3:
        return None

    bars = []
    for b in kline_bars:
        bar_time = b.get("time", 0)
        if isinstance(bar_time, (int, float)) and bar_time > 1e12:
            bar_time = bar_time / 1000.0  # ms → sec
        offset = bar_time - graduation_time
        bars.append({
            "open": float(b.get("open", 0)),
            "high": float(b.get("high", 0)),
            "low": float(b.get("low", 0)),
            "close": float(b.get("close", 0)),
            "volume": float(b.get("volume", 0)),
            "offset": offset,
        })

    post = [b for b in bars if b["offset"] >= -30]
    if len(post) < 3:
        return None
    grad_price = post[0]["open"]
    if grad_price <= 0:
        return None

    entry_bars = [b for b in post if b["offset"] >= eval_sec]
    if not entry_bars:
        return None
    entry_price = entry_bars[0]["open"]
    if entry_price <= 0:
        return None

    pre_entry = [b for b in post if b["offset"] <= eval_sec]
    if len(pre_entry) < 3:
        return None

    closes = [b["close"] / grad_price - 1 for b in pre_entry]
    highs = [b["high"] / grad_price - 1 for b in pre_entry]
    lows = [b["low"] / grad_price - 1 for b in pre_entry]

    peak_ret = max(highs)
    trough_ret = min(lows)
    peak_idx = highs.index(max(highs))
    trough_idx = lows.index(min(lows))
    entry_ret = entry_price / grad_price - 1
    last_close_ret = closes[-1]

    range_total = peak_ret - trough_ret
    recovery = (last_close_ret - trough_ret) / range_total if range_total > 0.01 else 0
    dd_from_peak = last_close_ret - peak_ret

    is_vshape = (peak_ret >= 0.10 and trough_ret < peak_ret - 0.10 and
                 recovery >= 0.30 and peak_idx < trough_idx)
    is_steady = (entry_ret > 0.10 and dd_from_peak > -0.10)
    is_near = (not is_vshape and not is_steady and
               dd_from_peak > -0.20 and entry_ret > 0 and peak_ret >= 0.05)
    any_pattern = int(is_vshape or is_steady or is_near)

    return {
        "any_pattern": any_pattern,
        "pattern": ("vshape" if is_vshape else "steady" if is_steady
                    else "near" if is_near else "none"),
        "entry_price": entry_price,
        "eval_sec": eval_sec,
        "vf_entry_ret": entry_ret,
        "vf_trough_ret": trough_ret,
        "vf_recovery_pct": recovery,
    }


def compute_micro_at_t(
    swaps: List[Dict],
    graduation_time: float,
    eval_sec: int,
    swap_data_max_sec: int = 600,
) -> Dict[str, float]:
    """Parameterized micro features. Swaps beyond swap_data_max_sec are
    not used (research constraint). Returns empty dict if insufficient
    swaps in [grad_time, grad_time + min(eval_sec, max)).

    Swap dict key compat:
      - timestamp | block_time : seconds since epoch
      - volume_sol | sol_amount : SOL volume
      - trader_address | trader : buyer/seller wallet
      - is_buy : bool
    Live DB uses (block_time, sol_amount, trader_address).
    Training data (Birdeye) uses (block_time, sol_amount, trader_address).
    Both keys supported so a single code path works.
    """
    effective_sec = min(int(eval_sec), int(swap_data_max_sec))
    cutoff = graduation_time + effective_sec

    def _ts(s):
        t = s.get("timestamp")
        if t is None:
            t = s.get("block_time", 0)
        return float(t or 0)

    early = [s for s in swaps if graduation_time <= _ts(s) < cutoff]
    if len(early) < 5:
        return {}

    buys = [s for s in early if s.get("is_buy")]
    sells = [s for s in early if not s.get("is_buy")]
    n_buys = len(buys)

    def _trader(s):
        return s.get("trader_address") or s.get("trader") or ""

    def _volume(s):
        return float(s.get("volume_sol") or s.get("sol_amount") or 0.0)

    unique_buyers = len({_trader(s) for s in buys if _trader(s)})
    unique_sellers = len({_trader(s) for s in sells if _trader(s)})

    # Growth accel
    seen: set = set()
    n_windows = max(2, effective_sec // 120)
    window_size = effective_sec // n_windows
    growth_windows = []
    for lo in range(0, effective_sec, window_size):
        lo_t = graduation_time + lo
        hi_t = lo_t + window_size
        wb = [s for s in buys if lo_t <= _ts(s) < hi_t]
        new = len({_trader(s) for s in wb if _trader(s)} - seen)
        seen.update(_trader(s) for s in wb if _trader(s))
        growth_windows.append(new)
    mid = len(growth_windows) // 2
    early_g = sum(growth_windows[:mid]) if mid > 0 else 1
    late_g = sum(growth_windows[mid:])
    growth_accel = late_g / max(early_g, 1)

    # Concentration (HHI, top1)
    if n_buys > 0:
        buyer_vols: Dict[str, float] = {}
        for s in buys:
            t = _trader(s)
            if not t:
                continue
            buyer_vols[t] = buyer_vols.get(t, 0.0) + _volume(s)
        total_buyer_vol = sum(buyer_vols.values())
        if total_buyer_vol > 0 and buyer_vols:
            shares = [v / total_buyer_vol for v in buyer_vols.values()]
            hhi = sum(s * s for s in shares)
            top1 = max(shares)
        else:
            hhi = top1 = 1.0
    else:
        hhi = top1 = 1.0

    # Late imbalance (last 2 min of the window)
    late_start = max(0, effective_sec - 120)
    late_t = graduation_time + late_start
    l2_buy = sum(_volume(s) for s in buys if _ts(s) >= late_t)
    l2_sell = sum(_volume(s) for s in sells if _ts(s) >= late_t)
    late_imb = (l2_buy - l2_sell) / (l2_buy + l2_sell) if (l2_buy + l2_sell) > 0 else 0

    return {
        "m10_unique_buyers": unique_buyers,
        "m10_buyer_hhi": hhi,
        "m10_top1_buyer_share": top1,
        "m10_late_imbalance": late_imb,
        "m10_growth_accel": growth_accel,
        "m10_seller_buyer_ratio": unique_sellers / max(unique_buyers, 1),
    }
