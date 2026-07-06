"""Phase 24 BigWinner v2 model inference (live).

Loads ``big_winner_v2_model.pkl`` (XGBClassifier, 38 features) and provides
the scoring API for the live entry-side filter.

Usage from controller:
    from controllers.generic.big_winner_inference import (
        load_big_winner_v2, predict_big_winner, big_winner_passes,
    )

    p_big = predict_big_winner(swaps_df, scan_t, grad_t)
    if p_big is not None and p_big >= cutoff:
        # passing entry candidate
        ...

Phase 24 deploy model (2026-05-01): Birdeye 54-day corpus, y_profitable
target, top_1pct live cutoff 0.734191358089447.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from meme_sniper.L4_Signal_and_Model_Inference import ShadowGuard

try:  # production: controllers.generic.ms.models (Hummingbot loads as controllers.generic.*)
    from controllers.generic.ms.models import registry as _model_registry
except ImportError:  # in-container test channel / direct file import
    from ms.models import registry as _model_registry

try:  # production: absolute import via controllers.generic.* (Hummingbot loader)
    from controllers.generic.meme_sniper_exit_models import (
        compute_all_features, compute_v3_extra_features,
    )
    from controllers.generic.big_winner_features import compute_phase16_entry_features
except ImportError:  # ad-hoc / smoke test: file imported directly
    from meme_sniper_exit_models import (
        compute_all_features, compute_v3_extra_features,
    )
    from big_winner_features import compute_phase16_entry_features

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _MODULE_DIR / "models"
MODEL_VERSION = "big_winner_v2"
MODEL_FILENAME = f"{MODEL_VERSION}_model.pkl"

# Recommended deploy cutoff (top 1% chronological holdout, Phase 24).
DEFAULT_CUTOFF_TOP_1PCT = 0.734191358089447
DEFAULT_CUTOFF_TOP_5PCT = 0.6805275082588196
DEFAULT_CUTOFF = DEFAULT_CUTOFF_TOP_1PCT

_BUNDLE = None  # cached pkl payload (model + feature_cols + cutoffs)
_GUARDED = None  # cached ShadowGuard wrapper holding the bundle


def load_big_winner_v2(*, metrics: Optional[Any] = None):
    """Load model + feature_cols + cutoffs once. Cached."""
    global _BUNDLE, _GUARDED
    # Phase 4b: delegate to unified registry; module globals kept in sync so
    # get_big_winner_guarded() continues to return _GUARDED for .gate() callers.
    path = _MODELS_DIR / MODEL_FILENAME
    guarded = _model_registry.get_or_load(path, metrics=metrics)
    _GUARDED = guarded
    _BUNDLE = guarded.unwrap()
    return _BUNDLE


# Backward-compatible import alias for older controller/plugin bundles.
load_big_winner_v1 = load_big_winner_v2


def get_big_winner_guarded():
    """Return the cached ShadowGuard wrapper for the big-winner model.

    Phase 2.D accessor — callers in meme_sniper.py use this to route
    raw-score predictions through `.gate(score)` for mode-aware
    enforcement. Returns None if `load_big_winner_v2()` has not yet
    been called (loader populates `_GUARDED` lazily).
    """
    return _GUARDED


def predict_big_winner(swaps: pd.DataFrame, scan_t: int, grad_t: int
                       ) -> Optional[float]:
    """Score a token at scan_t. Returns p_big_winner in [0, 1] or None.

    Args:
        swaps: DataFrame with columns:
            block_time, trader_address, is_buy, sol_amount, effective_price_sol
            (must contain swaps with block_time <= scan_t for forward-only)
        scan_t: decision unix-time (typically grad_t + 180..600)
        grad_t: graduation unix-time

    Returns None if features can't be computed (insufficient swaps, etc.).
    """
    if swaps is None or len(swaps) < 8:
        return None
    if scan_t <= grad_t:
        return None

    # Hard rule: enforce forward-only; features see only swaps <= scan_t.
    visible = swaps[swaps["block_time"] <= scan_t]
    if len(visible) < 8:
        return None

    base = compute_all_features(visible, int(scan_t), int(grad_t))
    if base is None:
        return None
    extras = compute_v3_extra_features(visible, int(scan_t))
    if extras is None:
        return None

    try:
        new_feats = compute_phase16_entry_features(visible, int(scan_t), int(grad_t))
    except Exception as e:
        logger.debug("big_winner: phase16 features failed: %s", e)
        return None

    all_feats = {**base, **extras, **new_feats}
    bundle = load_big_winner_v2()
    feat_cols = bundle["feature_cols"]

    missing_feats = [f for f in feat_cols if f not in all_feats]
    if missing_feats:
        logger.warning(
            "big_winner: %d feature(s) missing, using 0.0 fallback "
            "(predict will be biased): %s%s",
            len(missing_feats),
            missing_feats[:5],
            "..." if len(missing_feats) > 5 else "",
        )
    try:
        x = np.array([[float(all_feats.get(f, 0.0)) for f in feat_cols]])
    except Exception as e:
        logger.debug("big_winner: assembly failed: %s", e)
        return None
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        p = float(bundle["model"].predict_proba(x)[0, 1])
        if not np.isfinite(p):
            return None
        return p
    except Exception as e:
        logger.debug("big_winner: predict_proba failed: %s", e)
        return None


def predict_big_winner_with_features(swaps: pd.DataFrame, scan_t: int,
                                      grad_t: int):
    """Same as predict_big_winner but ALSO returns the feature dict so
    callers can persist features alongside the score (Phase 25h+ fix
    for shadow eval reproducibility — without the dict, sim must
    re-derive features from drifted swap state).

    Returns (score, features_dict) or (None, None) on insufficient data.
    """
    if swaps is None or len(swaps) < 8:
        return None, None
    if scan_t <= grad_t:
        return None, None
    visible = swaps[swaps["block_time"] <= scan_t]
    if len(visible) < 8:
        return None, None
    base = compute_all_features(visible, int(scan_t), int(grad_t))
    if base is None:
        return None, None
    extras = compute_v3_extra_features(visible, int(scan_t))
    if extras is None:
        return None, None
    try:
        new_feats = compute_phase16_entry_features(
            visible, int(scan_t), int(grad_t))
    except Exception as e:
        logger.debug("big_winner: phase16 features failed: %s", e)
        return None, None
    all_feats = {**base, **extras, **new_feats}
    bundle = load_big_winner_v2()
    feat_cols = bundle["feature_cols"]
    try:
        x = np.array([[float(all_feats.get(f, 0.0)) for f in feat_cols]])
    except Exception:
        return None, None
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        p = float(bundle["model"].predict_proba(x)[0, 1])
        if not np.isfinite(p):
            return None, None
        # Coerce numpy scalars to plain float for JSON
        return p, {k: float(v) for k, v in all_feats.items()
                    if isinstance(v, (int, float, np.integer, np.floating))}
    except Exception as e:
        logger.debug("big_winner: predict_proba failed: %s", e)
        return None, None


def big_winner_passes(p: Optional[float], cutoff: float = DEFAULT_CUTOFF) -> bool:
    """Decision: does p_big pass the configured cutoff?"""
    return p is not None and p >= cutoff


def get_cutoff_for_band(band: str) -> Optional[float]:
    """Look up cutoff value by band name (top_1pct, top_2pct, top_5pct, ...)."""
    bundle = load_big_winner_v2()
    return bundle.get("cutoffs", {}).get(band)
