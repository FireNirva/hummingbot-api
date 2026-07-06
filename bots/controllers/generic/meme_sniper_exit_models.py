"""
Facade: meme_sniper_exit_models — Phase 4c Strangler-Fig re-export.

All logic has moved to ms/models/exit_models.py. This module re-exports
everything dynamically so that:
  - 'from controllers.generic import meme_sniper_exit_models as mdl'
    and all mdl.X accesses (public, private, mutable state) resolve live.
  - 'from controllers.generic.meme_sniper_exit_models import name'
    works via PEP-562 __getattr__ (fires on missing names).
  - mdl.V3_LAST_WINDOW_USED reflects the live value after predict_14y_v3
    mutates it via 'global V3_LAST_WINDOW_USED' in _impl.

Do NOT use import-star or a one-time __dict__ copy — either would freeze
mutable module-level state (e.g. V3_LAST_WINDOW_USED) at import time.

Note on import-star: no production or test code uses 'from
meme_sniper_exit_models import *', so no __all__ forwarding is needed.
"""
from __future__ import annotations

try:  # production: controllers.generic.ms.models loaded by Hummingbot
    from controllers.generic.ms.models import exit_models as _impl
except ImportError:  # in-container test channel / direct file import
    from ms.models import exit_models as _impl


def __getattr__(name: str):
    """PEP-562: forward every attribute lookup to _impl live."""
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
