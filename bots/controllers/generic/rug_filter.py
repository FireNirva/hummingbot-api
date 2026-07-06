"""
Rug Filter model — t0 GMGN structural rug-risk classifier (shadow mode).

Model spec (matches 17b_rug_filter_train.py, rug_filter_v1):
  - XGBClassifier on 32 features flattened from gmgn_info_t0 + gmgn_security_t0
  - Target: min price in kline_60m (post-snapshot) <= 10% of grad price
  - Test AUC 0.7201 on chronological holdout (590 clean samples)

Live usage contract:
  1. Bot fetches gmgn_info_t0 + gmgn_security_t0 at M2 promotion (~T+2min).
     This is the SAME snapshot the model was trained on.
  2. flatten_gmgn_features() converts the two JSON blobs into a 32-feature
     row with the exact column order the model expects.
  3. RugFilterModel.predict_score() returns rug probability in [0, 1].
  4. RugFilterModel.would_reject(score, band) returns True if score >=
     cutoffs[band] — i.e. the token is in the predicted top-N% most
     rug-likely and should be skipped.

This module is IMPORT-SAFE even when rug filter is disabled — loading the
pkl or calling methods has no side effects on the live bot.
"""
from __future__ import annotations

import json
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


def _flatten_json(raw, prefix: str) -> dict:
    """Mirror of the flattening logic in 17a_rug_filter_data_audit.py.
    CRITICAL: must produce identical column names to training-time panel."""
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except Exception:
            return {}
    else:
        obj = raw
    if not isinstance(obj, dict):
        return {}
    out: dict = {}
    for k, v in obj.items():
        if isinstance(v, (int, float, bool)) or v is None:
            out[f"{prefix}{k}"] = v
        elif isinstance(v, str):
            try:
                out[f"{prefix}{k}"] = float(v)
            except ValueError:
                # non-numeric string: skip (same as training)
                pass
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float, bool)) or v2 is None:
                    out[f"{prefix}{k}__{k2}"] = v2
    return out


def flatten_gmgn_features(gmgn_info_t0, gmgn_security_t0) -> Dict[str, float]:
    """Produce one row of flattened features from the two snapshot blobs.
    Accepts either parsed dicts or JSON strings.
    """
    feats: dict = {}
    feats.update(_flatten_json(gmgn_info_t0, "info_"))
    feats.update(_flatten_json(gmgn_security_t0, "sec_"))
    return feats


class RugFilterModel:
    """Wrapper around the trained rug_filter_v1 pipeline."""

    def __init__(self, model_path: str, *, metrics: Optional[Any] = None):
        path = Path(model_path)
        self.guarded = _model_registry.get_or_load(path, metrics=metrics)
        art = self.guarded.unwrap()
        self.model = art["model"]
        # Patch sklearn imputer version incompatibility (same pattern as other models)
        try:
            from sklearn.impute import SimpleImputer as _SI
            if hasattr(self.model, "steps"):
                for _, step in self.model.steps:
                    if isinstance(step, _SI) and not hasattr(step, "_fill_dtype"):
                        step._fill_dtype = step.statistics_.dtype
        except Exception:
            pass

        self.version = str(art.get("version", "rug_filter_v1"))
        self.features: List[str] = list(art["features"])
        self.cutoffs: Dict[str, float] = {
            str(k): float(v) for k, v in dict(art.get("cutoffs", {})).items()
            if v is not None
        }
        self.train_auc = float(art.get("train_auc", 0.0))
        self.test_auc = float(art.get("test_auc", 0.0))

        logger.info(
            "RugFilterModel: loaded %s (%d features, test_auc=%.4f, bands=%s)",
            path, len(self.features), self.test_auc,
            ",".join(sorted(self.cutoffs.keys())),
        )

    def predict_score(self, features: Dict[str, float]) -> float:
        """Return rug probability in [0, 1]. Missing features → NaN → imputed."""
        X = pd.DataFrame(
            [{f: features.get(f, np.nan) for f in self.features}],
            columns=self.features,
        )
        return float(self.model.predict_proba(X)[0][1])

    def would_reject(self, score: float, band: str) -> bool:
        cut = self.cutoffs.get(band)
        if cut is None:
            return False
        return score >= cut
