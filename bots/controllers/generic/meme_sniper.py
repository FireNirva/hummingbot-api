"""
MemeSniper Controller — Hybrid architecture for Solana meme token momentum trading.

Architecture: ControllerBase lifecycle + direct Gateway REST API swap.
Does NOT use PositionExecutor (dynamic tokens not compatible with fixed trading_pair).

Modules:
  M1: Token Discovery  — Tier 1 Chainstack on-chain + Tier 2 GMGN enrichment (fallback: GMGN Trenches)
  M2: Signal Pipeline   — On-chain kline (primary) / GMGN kline (fallback) + GBT survival model
  M3: Trade Execution   — Gateway REST (POST /tokens/, POST /trading/swap/execute)
  M4: Position Monitor  — Gateway REST (GET /trading/swap/quote) every 30s
  M5: Risk Manager      — stop-loss, time-limit, daily-loss, consecutive-loss
"""
import asyncio
import json
import logging
import math
import os
import random  # Phase B.2: random.random() for cs_divergence_log_sample_rate
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np  # Phase B.4 (2026-05-20): np.array / np.nan_to_num for cs inference path
from prometheus_client import Gauge, start_http_server
from pydantic import Field

from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction

from meme_sniper.L6_Risk_and_Monitoring import (
    AlertDispatchError,
    AlertSender,
    BotState,
    Killswitch,
    KillswitchConfig,
    PrometheusExporter,
)

from controllers.generic.meme_sniper_utils import (
    ConfirmedContinuationEVShadowModel,
    SNAPSHOT_SCHEDULE,
    GatewayTrader,
    GraduatedToken,
    ObservationEntry,
    OnChainKlineBuilder,
    Position,
    RiskManager,
    SignalPipeline,
    TokenDiscovery,
    TradeCandidate,
    TradeDB,
    TradeRecord,
    VShapeModel,
    compute_confirmed_entry_live_features,
    compute_micro_10m_live,
    compute_micro_live,
    compute_micro_live_full,
    detect_vshape_live,
    evaluate_super_winner_event_shadow_view,
)

try:
    from controllers.generic.event_scanner import (
        FlowModel, detect_pattern_at_t, compute_micro_at_t,
    )
    HAS_EVENT_SCANNER = True
except Exception:  # pragma: no cover
    FlowModel = None  # type: ignore
    detect_pattern_at_t = None  # type: ignore
    compute_micro_at_t = None  # type: ignore
    HAS_EVENT_SCANNER = False

try:
    from controllers.generic.rug_filter import (
        RugFilterModel, flatten_gmgn_features,
    )
    HAS_RUG_FILTER = True
except Exception:  # pragma: no cover
    RugFilterModel = None  # type: ignore
    flatten_gmgn_features = None  # type: ignore
    HAS_RUG_FILTER = False

# RugFilter v3 entry-gate REMOVED 2026-05-06 (audit found AUC ≈ 0.50 on
# real-trade cohort — non-monotonic with rug rate). Replaced by v4 (Hopeless
# Trajectory T+5min shadow entry gate) imported below.
# NOTE: `predict_14y_v3` (used by exit-warn) is a SEPARATE model (Phase 14y
# event-anchored LogReg) — that lives in meme_sniper_exit_models.py and
# remains active for `p_rug_v3` columns in shadow_exit_warn_evals.

try:
    from controllers.generic.rug_filter_v4 import RugFilterV4
    HAS_RUG_FILTER_V4 = True
except Exception:  # pragma: no cover
    RugFilterV4 = None  # type: ignore
    HAS_RUG_FILTER_V4 = False

# v4.2 v0.9i hybrid (drawdown_2min < -55%, raw XGB probability, no LR cal).
# Spec: model_specs/2026-05-08_rug_filter_v4_2_SPEC.md v0.9f section
try:
    from controllers.generic.rug_filter_v4_2 import RugFilterV4_2
    HAS_RUG_FILTER_V4_2 = True
except Exception:  # pragma: no cover
    RugFilterV4_2 = None  # type: ignore
    HAS_RUG_FILTER_V4_2 = False

# v4.3 — cross-source-aligned retrain on Chainstack data (22 features after G_micro).
# Spec: model_specs/2026-05-13_rug_filter_v4_3_SPEC.md §14
try:
    from controllers.generic.rug_filter_v4_3 import RugFilterV4_3
    HAS_RUG_FILTER_V4_3 = True
except Exception:  # pragma: no cover
    RugFilterV4_3 = None  # type: ignore
    HAS_RUG_FILTER_V4_3 = False

# v4.4 — cs L1+L2 cleaning A/B vs v4.3 (2026-05-22). Same 22 features, same label,
# retrained on cs-cleaned panel. Subclass of RugFilterV4_3 with cs hook + two-stage
# fallback. Spec: research_notebooks/meme_sniper/data_cleaning/
#                 PHASE_E2_V4_4_CS_CLEANING_SPEC.md v0.6
try:
    from controllers.generic.rug_filter_v4_4 import RugFilterV4_4
    HAS_RUG_FILTER_V4_4 = True
except Exception:  # pragma: no cover
    RugFilterV4_4 = None  # type: ignore
    HAS_RUG_FILTER_V4_4 = False

# v3b Tier-1 — F0+F7 logistic on cap_hit cohort (post-entry filter at grad+600s)
# Spec: quants-lab/research_notebooks/meme_sniper/rug_funder_graph/
#       data/factors/tier1_deploy_spec.json (v3b_F0only_F7filter)
try:
    from controllers.generic.rug_filter_v3b_tier1 import RugFilterV3B_Tier1
    HAS_RUG_FILTER_V3B = True
except Exception:  # pragma: no cover
    RugFilterV3B_Tier1 = None  # type: ignore
    HAS_RUG_FILTER_V3B = False

# Creator cap_hit async resolver — closes v3b snapshot gap (2026-05-17)
try:
    from controllers.generic.creator_resolver import CreatorResolver
    HAS_CREATOR_RESOLVER = True
except Exception:  # pragma: no cover
    CreatorResolver = None  # type: ignore
    HAS_CREATOR_RESOLVER = False

# Funder resolver — 1-hop SOL trace per creator (Phase 1A 2026-05-19)
try:
    from controllers.generic.funder_resolver import FunderResolver
    HAS_FUNDER_RESOLVER = True
except Exception:  # pragma: no cover
    FunderResolver = None  # type: ignore
    HAS_FUNDER_RESOLVER = False

# Funder-graph rug filter (F15/F16 rule, Phase 1A 2026-05-19)
try:
    from controllers.generic.rug_filter_funder_v1 import FunderRugFilter
    HAS_FUNDER_RUG_FILTER = True
except Exception:  # pragma: no cover
    FunderRugFilter = None  # type: ignore
    HAS_FUNDER_RUG_FILTER = False

try:
    from controllers.generic.geyser_stream import GeyserPumpSwapStream
    HAS_GEYSER = True
except ImportError:
    HAS_GEYSER = False

# Phase B.1 (2026-05-18) — cleaning_service bridge (Week 3 scaffold, shadow only).
try:
    from controllers.generic.cleaning_service_bridge import CleaningServiceBridge
    HAS_CLEANING_SERVICE = True
except Exception:  # pragma: no cover — defensive (lib missing in older images)
    CleaningServiceBridge = None  # type: ignore
    HAS_CLEANING_SERVICE = False

# VWMP — Volume-Weighted Median Price for phantom-resistant current_price.
# Spec: research_notebooks/meme_sniper/vwmp_price_defense/2026-05-15_vwmp_price_source_SPEC.md
try:
    from controllers.generic.vwmp import compute_vwmp_with_diagnostics
    HAS_VWMP = True
except Exception:  # pragma: no cover
    compute_vwmp_with_diagnostics = None  # type: ignore
    HAS_VWMP = False

# Phase 4e-1 (2026-06-01): shadow rug-filter runner extracted to ms/shadow/
try:
    from controllers.generic.ms.shadow import rug_filter_runner
except ImportError:
    from ms.shadow import rug_filter_runner  # type: ignore[no-redef]

# Phase 4f-1 (2026-06-01): exit model engine extracted to ms/shadow/
try:
    from controllers.generic.ms.shadow import exit_model_engine
except ImportError:
    from ms.shadow import exit_model_engine  # type: ignore[no-redef]

# Phase 4g (2026-06-01): entry shadow runners extracted to ms/shadow/
try:
    from controllers.generic.ms.entry import xoscross_runner as _xoscross
except ImportError:
    from ms.entry import xoscross_runner as _xoscross
try:
    from controllers.generic.ms.shadow import entry_scanner
except ImportError:
    from ms.shadow import entry_scanner  # type: ignore[no-redef]

# Phase 4h (2026-06-01): sw_v2 shadow runner extracted to ms/shadow/
try:
    from controllers.generic.ms.shadow import sw_v2_shadow
except ImportError:
    from ms.shadow import sw_v2_shadow  # type: ignore[no-redef]

# 2026-06-28: pre-grad bonding-curve factor shadow (L1 universe rug-screen).
# OBSERVE-ONLY; flag pregrad_factors_shadow_enabled defaults OFF.
try:
    from controllers.generic.ms.shadow import pregrad_factors_shadow
except ImportError:
    from ms.shadow import pregrad_factors_shadow  # type: ignore[no-redef]

# Phase 4i (2026-06-01): cs divergence summary emitter extracted to ms/telemetry/
try:
    from controllers.generic.ms.telemetry import cs_divergence_monitor
except ImportError:
    from ms.telemetry import cs_divergence_monitor  # type: ignore[no-redef]

# Phase 4j (2026-06-01): shadow scheduling harness extracted to ms/shadow/orchestrator
try:
    from controllers.generic.ms.shadow import orchestrator as shadow_orchestrator
except ImportError:
    from ms.shadow import orchestrator as shadow_orchestrator  # type: ignore[no-redef]

# Phase 5-1 (2026-06-01): exit state holder + shared emergency-exit thresholds.
# Production-first dual-context per ms_extraction_dual_context_import memory.
# P5-1 imports ExitStateTracker for the __init__ alias block (zero caller change
# to existing self._X sites).  The threshold constants are imported here so
# P5-2/P5-8 (ExitDecisionEngine) can consume them; the in-path locals at lines
# 6581-6584 / 8725 / 8779-8780 are NOT yet rewired (that is P5-8).
try:
    from controllers.generic.ms.monitor.exit_state import ExitStateTracker
    from controllers.generic.ms.monitor.exit_thresholds import (  # noqa: F401
        SL_EMERGENCY_BYPASS_PNL,
        EC_EMERGENCY_BYPASS_PNL,
        SL_TIME_ESCAPE_SEC as _SL_TIME_ESCAPE_SEC_CONST,
    )
except ImportError:
    from ms.monitor.exit_state import ExitStateTracker  # type: ignore[no-redef]
    from ms.monitor.exit_thresholds import (  # type: ignore[no-redef]  # noqa: F401
        SL_EMERGENCY_BYPASS_PNL,
        EC_EMERGENCY_BYPASS_PNL,
        SL_TIME_ESCAPE_SEC as _SL_TIME_ESCAPE_SEC_CONST,
    )

# Phase 5-3 (2026-06-01): position sizing extracted to ms/execution/sizing.py.
# Production-first dual-context per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.execution import sizing
except ImportError:
    from ms.execution import sizing  # type: ignore[no-redef]

# Phase 5-4 (2026-06-01): _preflight_buy 7-check buy gate extracted to
# ms/execution/preflight.py.  Production-first dual-context per
# ms_extraction_dual_context_import memory.  Controller method is a 1-line
# async delegate; call site in _execute_buy is UNCHANGED.
try:
    from controllers.generic.ms.execution.preflight import preflight_buy
except ImportError:
    from ms.execution.preflight import preflight_buy  # type: ignore[no-redef]

# Phase 5-3 (2026-06-01): killswitch heartbeat check extracted to ms/risk/killswitch_handler.py.
# Production-first dual-context per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.risk import killswitch_handler
except ImportError:
    from ms.risk import killswitch_handler  # type: ignore[no-redef]

# T1.4 (2026-06-17): config validator boot-gate — pure-function, zero framework deps.
# dual-context import per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.config.validator import validate_config as _validate_config
except ImportError:
    from ms.config.validator import validate_config as _validate_config  # type: ignore[no-redef]

# Phase 5-5 (2026-06-01): reconciliation cluster extracted to ms/execution/reconciliation.py.
# Production-first dual-context per ms_extraction_dual_context_import memory.
# Methods: _reconcile_pending_buy/sell, _spawn_*_reconciliation, _recover_phantom_sell,
# _scan_wallet_for_orphans. Byte-identical logic; self.->ctrl in module fns.
try:
    from controllers.generic.ms.execution.reconciliation import (
        reconcile_pending_buy as _reconcile_pending_buy_fn,
        spawn_buy_reconciliation as _spawn_buy_reconciliation_fn,
        reconcile_pending_sell as _reconcile_pending_sell_fn,
        spawn_sell_reconciliation as _spawn_sell_reconciliation_fn,
        recover_phantom_sell as _recover_phantom_sell_fn,
        scan_wallet_for_orphans as _scan_wallet_for_orphans_fn,
    )
except ImportError:
    from ms.execution.reconciliation import (  # type: ignore[no-redef]
        reconcile_pending_buy as _reconcile_pending_buy_fn,
        spawn_buy_reconciliation as _spawn_buy_reconciliation_fn,
        reconcile_pending_sell as _reconcile_pending_sell_fn,
        spawn_sell_reconciliation as _spawn_sell_reconciliation_fn,
        recover_phantom_sell as _recover_phantom_sell_fn,
        scan_wallet_for_orphans as _scan_wallet_for_orphans_fn,
    )

# Phase 5-6 (2026-06-01): _execute_buy SOL-spend path extracted to
# ms/execution/buy.py.  Production-first dual-context per
# ms_extraction_dual_context_import memory.  Controller method is a 1-line
# async delegate; call site in control_task is UNCHANGED.
try:
    from controllers.generic.ms.execution.buy import execute_buy as _execute_buy_fn
except ImportError:
    from ms.execution.buy import execute_buy as _execute_buy_fn  # type: ignore[no-redef]

# Phase 5-7 (2026-06-01): _execute_sell SOL-receive path + position-ops
# helpers extracted to ms/execution/sell.py + position_ops.py.
# Production-first dual-context per ms_extraction_dual_context_import memory.
# Controller methods become 1-line delegates; call sites are UNCHANGED.
try:
    from controllers.generic.ms.execution.sell import execute_sell as _execute_sell_fn
    from controllers.generic.ms.execution.position_ops import (
        get_wallet_token_balance_ui as _get_wallet_token_balance_ui_fn,
        prune_stale_position as _prune_stale_position_fn,
    )
except ImportError:
    from ms.execution.sell import execute_sell as _execute_sell_fn  # type: ignore[no-redef]
    from ms.execution.position_ops import (  # type: ignore[no-redef]
        get_wallet_token_balance_ui as _get_wallet_token_balance_ui_fn,
        prune_stale_position as _prune_stale_position_fn,
    )

# Phase 5-8 (2026-06-01): _monitor_positions poll loop extracted to
# ms/monitor/position_monitor.py::run_monitor_positions.
# Production-first dual-context per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.monitor.position_monitor import run_monitor_positions as _run_monitor_positions_fn
except ImportError:
    from ms.monitor.position_monitor import run_monitor_positions as _run_monitor_positions_fn  # type: ignore[no-redef]

# Phase 5-8 (2026-06-01): gRPC fast-path SL/EC watcher extracted to
# ms/monitor/grpc_exit_watcher.py::run_price_event_watcher.
# Production-first dual-context per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.monitor.grpc_exit_watcher import run_price_event_watcher as _run_price_event_watcher_fn
except ImportError:
    from ms.monitor.grpc_exit_watcher import run_price_event_watcher as _run_price_event_watcher_fn  # type: ignore[no-redef]

# Phase 5-8 (2026-06-01): rug-event watcher (phantom-defense layer 2)
# extracted to ms/monitor/phantom_guard.py::run_rug_event_watcher.
# Production-first dual-context per ms_extraction_dual_context_import memory.
try:
    from controllers.generic.ms.monitor.phantom_guard import run_rug_event_watcher as _run_rug_event_watcher_fn
except ImportError:
    from ms.monitor.phantom_guard import run_rug_event_watcher as _run_rug_event_watcher_fn  # type: ignore[no-redef]

# T2.4 (2026-06-18): MemeSniperConfig extracted to ms/config/schema.py.
try:
    from controllers.generic.ms.config.schema import MemeSniperConfig
except ImportError:
    from ms.config.schema import MemeSniperConfig  # type: ignore[no-redef]
# T2.4 (2026-06-18): config resolver/normalizer helpers extracted to ms/config/aliases.py.
try:
    from controllers.generic.ms.config import aliases as _config_aliases
except ImportError:
    from ms.config import aliases as _config_aliases  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# Load GMGN API key from env or config file
_GMGN_ENV_PATH = os.path.expanduser("~/.config/gmgn/.env")


def _load_gmgn_api_key() -> str:
    """Load GMGN_API_KEY from ~/.config/gmgn/.env"""
    key = os.environ.get("GMGN_API_KEY", "")
    if key:
        return key
    if os.path.exists(_GMGN_ENV_PATH):
        with open(_GMGN_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GMGN_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class MemeSniper(ControllerBase):
    """Hybrid architecture controller: ControllerBase lifecycle + direct Gateway REST API."""

    # Config resolver/normalizer helpers — canonical impl in ms/config/aliases.py (T2.4).
    # Bound as staticmethods so self._cfg_value(...) / ctrl._cfg_value(...) /
    # cls._coalesce_config_alias(...) / self._normalize_model_config_aliases(...) all work.
    _set_config_value = staticmethod(_config_aliases._set_config_value)
    _coalesce_config_alias = staticmethod(_config_aliases._coalesce_config_alias)
    _normalize_model_config_aliases = staticmethod(_config_aliases._normalize_model_config_aliases)
    _enabled_any = staticmethod(_config_aliases._enabled_any)
    _cfg_value = staticmethod(_config_aliases._cfg_value)

    def _is_big_winner_entry_source(self, source: str) -> bool:
        return str(source or "").strip() in self._big_winner_entry_source_aliases

    def _record_buy_rejection(
        self, mint: str, reason: str,
        gate: str = "m3_preflight", strategy: str = "meme_sniper",
    ) -> None:
        """Telemetry helper for the 18 ``M3: [BUY BLOCKED]`` sites.

        Records both the buy=rejected counter and the per-gate rejection
        counter. The ``gate`` label is intentionally coarse-grained (single
        ``m3_preflight`` value across all 18 sites) to keep Prometheus
        cardinality bounded; the fine-grained reason is preserved in the
        existing ``logger.warning(...)`` adjacent to each callsite. Per-gate
        breakdown is deferred to a follow-up that maps reason→gate labels.
        """
        self.prom.record_buy(mint=mint, strategy=strategy, outcome="rejected")
        self.prom.record_rejection(strategy=strategy, gate=gate)

    def _record_shadow_fire(self, model_id: str) -> None:
        """Telemetry helper for the 14 ``would_fire_*`` shadow-eval sites.

        Each shadow model (v5_3 / v5_5_1 / v5_5_2 / v5_5_3 / v5_5_6 / v5_6 /
        ev3m / sw_v2 / rug v3b/v4_4 shadow / etc.) calls this with its own
        model_id so the Prometheus dashboard can slice fires by model.
        """
        self.prom.record_shadow_fire(model=model_id)

    def _killswitch_check(self) -> None:
        return killswitch_handler.check(self)

    def __init__(self, config: MemeSniperConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: MemeSniperConfig = config
        self._normalize_model_config_aliases(config)

        # L6 monitoring wire-up (Phase 2 W2.2 F2-M3a). Six standard
        # framework metrics + two deploy-adapter auxiliary gauges per
        # framework/L6/PHASE2_HANDOFF.md §1-2. start_http_server binds
        # 9090 for Prometheus scrape; alertmanager rules live at
        # framework/L6/alert_rules_v1.yml on the host stack.
        self.prom = PrometheusExporter()
        try:
            start_http_server(9090, registry=self.prom.registry)
        except OSError as exc:
            # Port-in-use on a host running multiple bot instances — log
            # loud + continue. Metrics still record on the registry; the
            # scrape endpoint is just inaccessible for this process.
            logger.error(
                f"L6: prometheus scrape endpoint 9090 unavailable ({exc}); "
                f"metrics will record but cannot be scraped from this process"
            )
        self.killswitch_tripped_gauge = Gauge(
            "meme_sniper_killswitch_tripped",
            "1 if killswitch tripped, 0 otherwise",
            registry=self.prom.registry,
        )
        # NOTE: `meme_sniper_outcome_resolved_last_run_ts` is intentionally NOT
        # declared here even though L6 PHASE2_HANDOFF §2 lists it as an
        # auxiliary gauge. The resolver runs in a SEPARATE cron container
        # (scripts/resolve_shadow_outcomes.py) and writes the gauge via
        # node_exporter textfile collector (see F2-M3a §scripts edit). If
        # the bot ALSO declared this gauge here, the bot's registry would
        # expose it as `0` (default, never set) — and the alert rule
        # `(time() - meme_sniper_outcome_resolved_last_run_ts) > 86400`
        # in framework/L6/alert_rules_v1.yml would fire constantly
        # false-positive against the bot's scrape endpoint. Single
        # source-of-truth (textfile) avoids that.

        # Killswitch — checks drawdown / consecutive-loss / P0-alerts in the
        # heartbeat block (~30s cadence). Trip = additive halt on top of the
        # existing daily-loss / consecutive-loss RiskManager halt; the bot
        # design (line 7560: exits also gated by self.risk.halted) already
        # makes halted = Option C "halt everything including exit attempts,
        # let exchange-side SL/EC fire". Killswitch preserves that semantic.
        self.killswitch = Killswitch(
            KillswitchConfig(
                max_drawdown_pct=float(config.kill_max_drawdown_pct),
                max_consecutive_losses=int(config.kill_max_consecutive_losses),
                max_p0_alerts_in_1h=int(config.kill_max_p0_alerts_in_1h),
            )
        )
        # AlertSender — reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env at
        # construction. Telegram has rate limits (~30 msg/sec global, 1
        # msg/sec per chat); we throttle to 1 alert per 60s + edge-only
        # (False→True transition) to avoid spam during a flapping killswitch.
        self.alerts = AlertSender()
        self._last_killswitch_alert_ts: Optional[float] = None
        self._killswitch_was_tripped: bool = False  # edge-detection state

        self._big_winner_model_version = str(
            getattr(config, "big_winner_model_version", "") or "big_winner_v2")
        self._big_winner_entry_source = str(
            getattr(config, "big_winner_entry_source", "") or "big_winner_v2")
        self._big_winner_entry_source_aliases = {
            self._big_winner_entry_source,
            self._big_winner_model_version,
            "big_winner_v2",
            "big_winner_v1",
        }

        gmgn_key = _load_gmgn_api_key()
        if not gmgn_key:
            logger.error("GMGN_API_KEY not found — M1/M2 will not work")

        # M1: Token Discovery (three-tier: Chainstack → GMGN enrichment → GMGN kline)
        chainstack_url = config.chainstack_rpc_url or os.environ.get("CHAINSTACK_SOLANA_RPC", "")
        # v0.7.17: public RPC URL (default = official mainnet) for indexing-lag fallback.
        # Env var PUBLIC_SOLANA_RPC can override or set to empty to disable.
        public_rpc_url = (
            os.environ.get("PUBLIC_SOLANA_RPC")
            if "PUBLIC_SOLANA_RPC" in os.environ
            else config.public_rpc_url
        )
        self.discovery = TokenDiscovery(
            api_key=gmgn_key,
            base_url=config.gmgn_base_url,
            min_liquidity_usd=config.min_liquidity_usd,
            chainstack_rpc_url=chainstack_url,
            chainstack_batch_size=config.chainstack_batch_size,
            chainstack_tx_concurrency=config.chainstack_tx_concurrency,
            instant_buy_mode=config.instant_buy_mode,
            public_rpc_url=public_rpc_url or "",
        )
        if chainstack_url:
            logger.info(f"M1: Chainstack on-chain discovery enabled (batch={config.chainstack_batch_size})")
        else:
            logger.warning("M1: No Chainstack RPC configured — using GMGN Trenches fallback only")
        if public_rpc_url:
            logger.info(f"v0.7.17: public RPC fallback enabled ({public_rpc_url}) — "
                        f"applied to BUY reconcile + orphan_scanner only; M4 hot path unaffected")
        else:
            logger.warning("v0.7.17: public RPC fallback DISABLED — Chainstack indexing lag will silently strand BUYs")

        # M2: Signal Pipeline
        self.signal: Optional[SignalPipeline] = None
        if config.model_path and os.path.exists(config.model_path):
            self.signal = SignalPipeline(
                model_path=config.model_path,
                api_key=gmgn_key,
                base_url=config.gmgn_base_url,
                metrics=self.prom,
            )
            n_bars = max(3, config.feature_delay_sec // 60)
            self.signal.set_min_bars(n_bars)
        else:
            logger.error(f"Model not found at {config.model_path} — M2 will not predict")
        self.shadow_ev3m_model: Optional[ConfirmedContinuationEVShadowModel] = None
        if config.shadow_ev3m_enabled:
            if config.shadow_ev3m_model_path and os.path.exists(config.shadow_ev3m_model_path):
                try:
                    self.shadow_ev3m_model = ConfirmedContinuationEVShadowModel(
                        config.shadow_ev3m_model_path,
                        metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"EV3m shadow model failed to load from {config.shadow_ev3m_model_path}: {e}")
            else:
                logger.error(f"EV3m shadow model not found at {config.shadow_ev3m_model_path}")

        self.ev3m_live_model: Optional[ConfirmedContinuationEVShadowModel] = None
        if config.ev3m_live_enabled:
            if config.ev3m_live_model_path and os.path.exists(config.ev3m_live_model_path):
                try:
                    self.ev3m_live_model = ConfirmedContinuationEVShadowModel(
                        config.ev3m_live_model_path,
                        metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"EV3m live model failed to load from {config.ev3m_live_model_path}: {e}")
            else:
                logger.error(f"EV3m live model not found at {config.ev3m_live_model_path}")
        self.shadow_super_winner_event_model: Optional[ConfirmedContinuationEVShadowModel] = None
        if config.shadow_super_winner_event_enabled:
            if config.shadow_super_winner_event_model_path and os.path.exists(config.shadow_super_winner_event_model_path):
                try:
                    self.shadow_super_winner_event_model = ConfirmedContinuationEVShadowModel(
                        config.shadow_super_winner_event_model_path,
                        metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(
                        f"Super winner event shadow model failed to load from "
                        f"{config.shadow_super_winner_event_model_path}: {e}"
                    )
            else:
                logger.error(
                    f"Super winner event shadow model not found at "
                    f"{config.shadow_super_winner_event_model_path}"
                )

        # Rug filter v1 shadow model
        self.rug_filter_model = None
        if config.shadow_rug_enabled and HAS_RUG_FILTER:
            if config.shadow_rug_model_path and os.path.exists(config.shadow_rug_model_path):
                try:
                    self.rug_filter_model = RugFilterModel(
                        config.shadow_rug_model_path, metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"RugFilterModel load failed: {e}")
            else:
                logger.warning(
                    f"RugFilterModel path not found: {config.shadow_rug_model_path} "
                    "— shadow rug scorer disabled")

        # Rug filter v4 shadow model (T+5m, 5 features, R1+R4 ensemble)
        # Spec: model_specs/2026-05-06_rug_filter_v4_CTA_FROZEN_SPEC.md
        self.rug_filter_v4_model = None
        if config.shadow_rug_v4_enabled and HAS_RUG_FILTER_V4:
            if config.shadow_rug_v4_model_path and os.path.exists(config.shadow_rug_v4_model_path):
                try:
                    self.rug_filter_v4_model = RugFilterV4(
                        model_path=config.shadow_rug_v4_model_path,
                        cutoff=config.shadow_rug_v4_cutoff,
                        window_s=config.shadow_rug_v4_window_sec,
                        min_swaps=config.shadow_rug_v4_min_swaps,
                        metrics=self.prom,
                    )
                    logger.info(
                        f"RugFilterV4 loaded: cutoff={config.shadow_rug_v4_cutoff} "
                        f"window={config.shadow_rug_v4_window_sec}s "
                        f"min_swaps={config.shadow_rug_v4_min_swaps}"
                    )
                except Exception as e:
                    logger.error(f"RugFilterV4 load failed: {e}")
            else:
                logger.warning(
                    f"RugFilterV4 path not found: {config.shadow_rug_v4_model_path} "
                    "— v4 shadow scorer disabled"
                )

        # Rug filter v4.2 v0.9i hybrid (T+5m, 18 features, raw XGB probability)
        # Spec: model_specs/2026-05-08_rug_filter_v4_2_SPEC.md v0.9f
        self.rug_filter_v4_2_model = None
        if config.shadow_rug_v4_2_enabled and HAS_RUG_FILTER_V4_2:
            if config.shadow_rug_v4_2_model_path and os.path.exists(config.shadow_rug_v4_2_model_path):
                try:
                    self.rug_filter_v4_2_model = RugFilterV4_2(
                        model_path=config.shadow_rug_v4_2_model_path,
                        cutoff=config.shadow_rug_v4_2_cutoff,
                        window_s=config.shadow_rug_v4_2_window_sec,
                        min_swaps=config.shadow_rug_v4_2_min_swaps,
                        metrics=self.prom,
                    )
                    logger.info(
                        f"RugFilterV4_2 loaded (v0.9i hybrid raw): "
                        f"cutoff={config.shadow_rug_v4_2_cutoff:.4f} "
                        f"window={config.shadow_rug_v4_2_window_sec}s "
                        f"min_swaps={config.shadow_rug_v4_2_min_swaps}"
                    )
                except Exception as e:
                    logger.error(f"RugFilterV4_2 load failed: {e}")
                    self.rug_filter_v4_2_model = None
            else:
                logger.warning(
                    f"RugFilterV4_2 path not found: {config.shadow_rug_v4_2_model_path} "
                    "— v4.2 shadow scorer disabled"
                )

        # Rug filter v4.3 — cross-source-aligned (22 features after G_micro)
        # Spec: model_specs/2026-05-13_rug_filter_v4_3_SPEC.md §14
        self.rug_filter_v4_3_model = None
        if config.shadow_rug_v4_3_enabled and HAS_RUG_FILTER_V4_3:
            if config.shadow_rug_v4_3_model_path and os.path.exists(config.shadow_rug_v4_3_model_path):
                try:
                    self.rug_filter_v4_3_model = RugFilterV4_3(
                        model_path=config.shadow_rug_v4_3_model_path,
                        cutoff=config.shadow_rug_v4_3_cutoff,
                        window_s=config.shadow_rug_v4_3_window_sec,
                        min_swaps=config.shadow_rug_v4_3_min_swaps,
                        metrics=self.prom,
                    )
                    logger.info(
                        f"RugFilterV4_3 loaded (depth4_balanced, 22 features): "
                        f"cutoff={config.shadow_rug_v4_3_cutoff:.4f} "
                        f"window={config.shadow_rug_v4_3_window_sec}s "
                        f"min_swaps={config.shadow_rug_v4_3_min_swaps} "
                        f"shadow_only={config.shadow_rug_v4_3_shadow_only}"
                    )
                except Exception as e:
                    logger.error(f"RugFilterV4_3 load failed: {e}")
                    self.rug_filter_v4_3_model = None
            else:
                logger.warning(
                    f"RugFilterV4_3 path not found: {config.shadow_rug_v4_3_model_path} "
                    "— v4.3 shadow scorer disabled"
                )

        # Rug filter v4.4 — cs L1+L2 cleaning A/B vs v4.3 (2026-05-22, spec v0.6)
        # Inherits __init__/score_features/_predict from v4.3; overrides only
        # score_from_sqlite to insert cs.apply_l0_l1_l2 + two-stage fallback.
        # Shares CleaningService instance via cs_service injection if bridge available.
        self.rug_filter_v4_4_model = None
        if config.shadow_rug_v4_4_enabled and HAS_RUG_FILTER_V4_4:
            if config.shadow_rug_v4_4_model_path and os.path.exists(config.shadow_rug_v4_4_model_path):
                try:
                    # cleaning_service_bridge is initialized later in __init__
                    # (line ~1854). Use getattr for safety; if bridge not yet
                    # available, v4.4 uses a standalone CleaningService instance.
                    # apply_l0_l1_l2 is stateless per-call, so this is functionally
                    # equivalent to shared bridge (no buffer needed for batch path).
                    bridge = getattr(self, '_cleaning_service_bridge', None)
                    cs_for_v44 = bridge.service if bridge is not None else None
                    self.rug_filter_v4_4_model = RugFilterV4_4(
                        model_path=config.shadow_rug_v4_4_model_path,
                        cutoff=config.shadow_rug_v4_4_cutoff,
                        window_s=config.shadow_rug_v4_4_window_sec,
                        min_swaps=config.shadow_rug_v4_4_min_swaps,
                        cs_service=cs_for_v44,
                        metrics=self.prom,
                    )
                    logger.info(
                        f"RugFilterV4_4 loaded (cs L1+L2 cleaning, 22 features): "
                        f"cutoff={config.shadow_rug_v4_4_cutoff:.4f} "
                        f"window={config.shadow_rug_v4_4_window_sec}s "
                        f"min_swaps={config.shadow_rug_v4_4_min_swaps} "
                        f"shadow_only={config.shadow_rug_v4_4_shadow_only} "
                        f"cs_service={'shared_bridge' if cs_for_v44 else 'standalone'}"
                    )
                except Exception as e:
                    logger.error(f"RugFilterV4_4 load failed: {e}")
                    self.rug_filter_v4_4_model = None
            else:
                logger.warning(
                    f"RugFilterV4_4 path not found: {config.shadow_rug_v4_4_model_path} "
                    "— v4.4 shadow scorer disabled"
                )

        # Rug filter v3b Tier-1 — F0+F7 logistic on cap_hit cohort
        # Post-entry filter (eval at grad_t + 600s). Loads 3 snapshot files.
        # Spec: tier1_deploy_spec.json (v3b_F0only_F7filter, 2026-05-17)
        self.rug_filter_v3b_model = None
        if config.shadow_rug_v3b_enabled and HAS_RUG_FILTER_V3B:
            if (config.shadow_rug_v3b_model_path
                and os.path.exists(config.shadow_rug_v3b_model_path)):
                try:
                    self.rug_filter_v3b_model = RugFilterV3B_Tier1(
                        model_path=config.shadow_rug_v3b_model_path,
                        trader_stats_path=config.shadow_rug_v3b_trader_stats_path,
                        cap_hit_path=config.shadow_rug_v3b_cap_hit_path,
                        metrics=self.prom,
                    )
                    logger.info(
                        f"RugFilterV3B_Tier1 loaded (F0+F7, cap_hit cohort): "
                        f"mode={config.shadow_rug_v3b_mode} "
                        f"eval_delay={config.shadow_rug_v3b_eval_delay_sec}s "
                        f"cutoffs=top1={self.rug_filter_v3b_model.cutoffs['top_1_pct']:.3f} "
                        f"top5={self.rug_filter_v3b_model.cutoffs['top_5_pct']:.3f} "
                        f"top10={self.rug_filter_v3b_model.cutoffs['top_10_pct']:.3f} "
                        f"trader_stats_n={len(self.rug_filter_v3b_model._trader_stats):,} "
                        f"cap_hit_map_n={len(self.rug_filter_v3b_model._cap_hit_map):,} "
                        f"model_date={self.rug_filter_v3b_model.model_date}"
                    )
                except Exception as e:
                    logger.error(f"RugFilterV3B_Tier1 load failed: {e}")
                    self.rug_filter_v3b_model = None
            else:
                logger.warning(
                    f"RugFilterV3B_Tier1 path not found: {config.shadow_rug_v3b_model_path} "
                    "— v3b shadow scorer disabled"
                )

        # Async creator cap_hit resolver — populates DB cache as new mints
        # graduate. Solves v3b snapshot gap window (mints grad'd post-snapshot
        # are otherwise SKIP_CAP_HIT_PENDING forever).
        # Started in on_start(), stopped in on_stop().
        self.creator_resolver: Optional[CreatorResolver] = None
        if (config.shadow_rug_v3b_enabled and HAS_CREATOR_RESOLVER
            and self.rug_filter_v3b_model is not None):
            # Created here; started in on_start() (needs async context).
            # rpc is borrowed from TokenDiscovery._rpc after it's initialized.
            self._creator_resolver_pending = True
        else:
            self._creator_resolver_pending = False

        # Funder resolver — Phase 1A 2026-05-19. Same lifecycle as
        # creator_resolver; started in on_start() right after.
        self.funder_resolver: Optional[FunderResolver] = None

        # Funder-graph rug filter — F15/F16 rule (shadow mode initially).
        # Spec: 2026-05-18_funder_only_rug_filter_SPEC.md §6.1
        self.rug_filter_funder: Optional[FunderRugFilter] = None
        if HAS_FUNDER_RUG_FILTER and config.shadow_rug_funder_v1_enabled:
            try:
                self.rug_filter_funder = FunderRugFilter(
                    config.shadow_rug_funder_v1_rrw_path)
                logger.info(
                    f"FunderRugFilter loaded: {self.rug_filter_funder.stats()}")
            except Exception as e:
                logger.warning(
                    f"FunderRugFilter init failed (will return PASS-all): {e}")
                self.rug_filter_funder = None

        # Phase B event-triggered scanner (shadow only)
        self.flow_model = None
        if config.shadow_event_enabled and HAS_EVENT_SCANNER:
            if config.shadow_event_model_path and os.path.exists(config.shadow_event_model_path):
                try:
                    self.flow_model = FlowModel(
                        config.shadow_event_model_path, metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"FlowModel load failed: {e}")
            else:
                logger.warning(
                    f"FlowModel path not found: {config.shadow_event_model_path} "
                    "— shadow event scanner disabled")
        self._shadow_event_pending: Dict[str, Dict] = {}

        # Phase 24 BigWinner v2 soft entry filter.
        self._big_winner_pending: Dict[str, Dict] = {}
        self._big_winner_cooldown: Dict[str, float] = {}
        self._big_winner_trade_count: int = 0  # canary counter (restored from DB below)
        self.big_winner_loaded = False
        if config.big_winner_enabled:
            try:
                from controllers.generic.big_winner_inference import load_big_winner_v2
                bundle = load_big_winner_v2(metrics=self.prom)
                self.big_winner_loaded = True
                logger.info(
                    f"BigWinnerV2: loaded ({len(bundle['feature_cols'])} features, "
                    f"cutoff={config.big_winner_cutoff:.4f}, "
                    f"shadow_only={config.big_winner_shadow_only}, "
                    f"size=${config.big_winner_position_size_usd:.2f}, "
                    f"entry_source={self._big_winner_entry_source})")
            except Exception as e:
                logger.error(f"BigWinnerV2 load failed: {e}")

        # Phase B Event Invariant v1 Model B (shadow only, T+3-5m only)
        self.event_invariant_model = None
        if config.shadow_event_invariant_enabled and HAS_EVENT_SCANNER:
            if (config.shadow_event_invariant_model_path
                    and os.path.exists(config.shadow_event_invariant_model_path)):
                try:
                    # FlowModel class works for any 9-feature regressor
                    # with stored cutoffs — same structural contract.
                    self.event_invariant_model = FlowModel(
                        config.shadow_event_invariant_model_path,
                        metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"EventInvariantModel load failed: {e}")
            else:
                logger.warning(
                    f"EventInvariantModel path not found: "
                    f"{config.shadow_event_invariant_model_path} — shadow disabled")
        self._shadow_event_invariant_pending: Dict[str, Dict] = {}

        if config.vshape_entry_enabled and config.vshape_live_enabled:
            raise ValueError(
                "vshape_entry_enabled=true is incompatible with "
                "vshape_live_enabled=true. Disable the legacy v1.6.1 live "
                "gate so the rolling v3.4 gate is the only live V-shape "
                "entry decision."
            )

        # V-shape T+10m shadow model
        self.shadow_vshape_model: Optional[VShapeModel] = None
        if config.shadow_vshape_enabled:
            if config.shadow_vshape_model_path and os.path.exists(config.shadow_vshape_model_path):
                try:
                    self.shadow_vshape_model = VShapeModel(
                        config.shadow_vshape_model_path, metrics=self.prom,
                    )
                except Exception as e:
                    logger.error(f"VShape model failed to load from {config.shadow_vshape_model_path}: {e}")
            else:
                logger.warning(f"VShape model not found at {config.shadow_vshape_model_path}")
        self._shadow_vshape_pending: Dict[str, Dict] = {}

        # Phase 22 v3.1 Variant D — rolling first-passage scan replaces v1.6.1
        self.vshape_entry_model: Optional[VShapeModel] = None
        self.v3_1_model: Optional[VShapeModel] = None  # legacy alias for older methods
        if config.vshape_entry_enabled:
            if os.path.exists(config.vshape_entry_model_path):
                try:
                    self.vshape_entry_model = VShapeModel(
                        config.vshape_entry_model_path, metrics=self.prom,
                    )
                    self.v3_1_model = self.vshape_entry_model
                    _override = float(getattr(config, "vshape_entry_cutoff_override", -1.0) or -1.0)
                    _effective_cutoff = (
                        _override if _override > 0
                        else self.vshape_entry_model.selection_cutoffs.get(config.vshape_entry_selection_band)
                    )
                    _cutoff_source = (
                        f"override={_override:.4f}" if _override > 0
                        else f"band={config.vshape_entry_selection_band}"
                    )
                    logger.info(
                        f"VShapeEntry {config.vshape_entry_model_version} loaded: "
                        f"{len(self.vshape_entry_model.feature_names)} features, "
                        f"scan [{config.vshape_entry_scan_min_sec}s, {config.vshape_entry_scan_max_sec}s] "
                        f"step {config.vshape_entry_scan_step_sec}s, "
                        f"{_cutoff_source}, cutoff={_effective_cutoff}"
                    )
                except Exception as e:
                    logger.error(f"VShapeEntry model failed to load from {config.vshape_entry_model_path}: {e}")
                    raise ValueError(f"vshape_entry_enabled=true but model load failed: {e}")
            else:
                raise ValueError(
                    f"vshape_entry_enabled=true but model not found at "
                    f"{config.vshape_entry_model_path} - refusing to start")
            band = str(config.vshape_entry_selection_band or "").strip()
            if band not in (self.vshape_entry_model.selection_cutoffs or {}):
                raise ValueError(
                    f"vshape_entry_selection_band={band!r} not in model cutoffs "
                    f"{sorted((self.vshape_entry_model.selection_cutoffs or {}).keys())}"
                )
        # Per-token last V-shape entry scan timestamp for throttling.
        self._vshape_entry_last_scan_ts: Dict[str, float] = {}
        self._v3_1_last_scan_ts = self._vshape_entry_last_scan_ts

        # Live-mode invariants: fail loud at startup rather than silently
        # mis-gating trades.
        if config.vshape_live_enabled:
            if config.instant_buy_mode:
                raise ValueError(
                    "vshape_live_enabled=true is incompatible with "
                    "instant_buy_mode=true — instant_buy skips the V-shape "
                    "gate entirely. Disable one.")
            if self.shadow_vshape_model is None:
                raise ValueError(
                    "vshape_live_enabled=true but V-shape model failed to "
                    "load — refusing to start (would reject every candidate).")
            band = str(config.vshape_live_selection_band or "").strip()
            cutoffs = self.shadow_vshape_model.selection_cutoffs or {}
            if band not in cutoffs:
                raise ValueError(
                    f"vshape_live_selection_band={band!r} not in model "
                    f"cutoffs {sorted(cutoffs.keys())} — typo? "
                    "Refusing to start with a silent 0.5 fallback.")

        # M3: Gateway Trader
        # Prefer env so the repo can keep conf/controllers/conf_meme_sniper.yml
        # free of host-specific wallet identities. The yml field remains as a
        # fallback for older deployments.
        wallet_address = (
            os.environ.get("SOLANA_WALLET_ADDRESS", "").strip()
            or config.wallet_address.strip()
        )
        if not wallet_address:
            raise ValueError(
                "wallet_address is empty — set it in conf/controllers/conf_meme_sniper.yml "
                "or set SOLANA_WALLET_ADDRESS in the container environment. "
                "Default is intentionally empty so the public Docker image does not "
                "ship anyone's wallet identity.")
        if wallet_address != config.wallet_address.strip():
            logger.info("wallet_address loaded from SOLANA_WALLET_ADDRESS env")
            try:
                config.wallet_address = wallet_address
            except Exception:
                object.__setattr__(config, "wallet_address", wallet_address)
        self.trader = GatewayTrader(
            gateway_url=config.gateway_url,
            wallet_address=wallet_address,
            connector=config.connector,
            chain_network=config.chain_network,
            slippage_pct=config.slippage_pct,
            jupiter_api_key=os.environ.get("JUPITER_API_KEY", ""),
        )

        # M5: Risk Manager
        self.risk = RiskManager(
            daily_loss_limit_usd=float(config.daily_loss_limit_usd),
            max_consecutive_losses=config.max_consecutive_losses,
            max_total_trades=config.max_total_trades,
            max_positions=config.max_positions,
            cooldown_sec=config.cooldown_sec,
        )

        # On-chain kline builder (builds kline from pool swap txs — zero GMGN dependency)
        self.kline_builder: Optional[OnChainKlineBuilder] = None
        if chainstack_url:
            self.kline_builder = OnChainKlineBuilder(
                rpc_url=chainstack_url,
                tx_concurrency=config.chainstack_tx_concurrency,
            )
            logger.info("M2: OnChainKlineBuilder enabled (will build kline from pool swap data)")

        # Yellowstone gRPC Geyser stream (replaces WS + HTTP for M1 + M2)
        # Rug event queue: gRPC stream emits here when a large sell hits a
        # watched pool. `_rug_event_watcher` consumes and fires panic sell.
        self._rug_event_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._rug_event_watcher_task: Optional[asyncio.Task] = None

        # Phase 22.S.P1 — price event queue for fast SL/EC trigger
        self._price_event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._price_event_watcher_task: Optional[asyncio.Task] = None
        # Phase 5-1: exit/sell mutable-container state collected into ExitStateTracker.
        # Aliases below keep every existing self._X access site byte-identical.
        # See ms/monitor/exit_state.py for field docs + alias-pattern rationale.
        # _stoploss_blacklist stays on controller (reassigned in heartbeat + on_start).
        self._exit_state = ExitStateTracker()
        self._exit_in_progress        = self._exit_state.exit_in_progress        # Set[str]

        # Phase B.1: cleaning_service bridge (shadow only — no consumer wired yet).
        # Instantiated BEFORE the gRPC stream so the stream can hold a reference.
        # If disabled or lib missing, bridge stays None and zero overhead.
        self._cleaning_service_bridge: Optional["CleaningServiceBridge"] = None
        if config.cleaning_service_enabled and HAS_CLEANING_SERVICE:
            try:
                wal_path = (
                    Path(config.cleaning_service_wal_path)
                    if config.cleaning_service_wal_path else None
                )
                self._cleaning_service_bridge = CleaningServiceBridge(
                    max_buffer_per_mint=config.cleaning_service_max_buffer_per_mint,
                    wal_path=wal_path,
                    diagnostic_log_interval_sec=config.cleaning_service_diagnostic_interval_sec,
                )
                logger.info(
                    "Phase B.1: cleaning_service bridge enabled "
                    "(buffer=%d, wal=%s, diag=%.0fs)",
                    config.cleaning_service_max_buffer_per_mint,
                    config.cleaning_service_wal_path or "off",
                    config.cleaning_service_diagnostic_interval_sec,
                )
            except Exception as e:
                # Defensive — never let bridge init kill bot startup
                logger.warning(
                    "Phase B.1: cleaning_service bridge init failed (%s) — "
                    "continuing without it",
                    e,
                )
                self._cleaning_service_bridge = None
        elif config.cleaning_service_enabled and not HAS_CLEANING_SERVICE:
            logger.warning(
                "Phase B.1: cleaning_service_enabled=True but "
                "cleaning_service_bridge not importable (restart container?)"
            )

        # Phase B.2 divergence-shadow state (per v3 plan — no batch buffer).
        # Cumulative (session-total): never reset, used in CUMULATIVE summary
        # log line + analyze-script DB queries reference.
        self._cs_divergence_total_count = 0
        self._cs_divergence_flagged_5pct = 0
        self._cs_divergence_flagged_2pct = 0
        self._cs_divergence_flagged_1pct = 0
        self._cs_divergence_inline_null_count = 0
        self._cs_divergence_cs_null_count = 0
        self._cs_divergence_query_count = 0  # all queries (including null results)
        # Audit fix (P1): per-window counters reset each summary emit so flag
        # rates align with windowed percentile samples (no slow-moving-average
        # blur as session lengthens).
        self._cs_divergence_total_count_window = 0
        self._cs_divergence_flagged_5pct_window = 0
        self._cs_divergence_flagged_2pct_window = 0
        self._cs_divergence_flagged_1pct_window = 0
        self._cs_divergence_inline_null_count_window = 0
        self._cs_divergence_cs_null_count_window = 0
        self._cs_divergence_query_count_window = 0
        # Errors stay cumulative (v3 acceptance #5: 0 over 24h)
        self._cs_divergence_error_count = 0
        self._cs_divergence_pct_samples: List[float] = []   # rolling for percentiles
        self._cs_latency_ms_samples: List[float] = []       # rolling for percentiles
        self._cs_div_last_summary_t = time.monotonic()

        # Phase B.2.2 promote: counters track how often cs actually drove
        # current_price decisions vs falling back to inline VWMP or pool.
        # Observability for the canary rollout (operator hot-toggles config).
        # Audit fix P0.1: separate _invalid_count from None-fallback (cs <= 0).
        # Audit fix P0.2: per-window versions reset at each summary emit,
        # matching the divergence counter pattern from v3 audit-fix.
        self._cs_promoted_used_count = 0           # cs supplied current_price
        self._cs_promoted_fallback_inline_count = 0   # cs=None, used inline VWMP
        self._cs_promoted_fallback_pool_count = 0     # cs+inline None, used pool
        self._cs_promoted_invalid_count = 0        # cs returned <= 0 / NaN / Inf
        self._cs_promoted_used_count_window = 0
        self._cs_promoted_fallback_inline_count_window = 0
        self._cs_promoted_fallback_pool_count_window = 0
        self._cs_promoted_invalid_count_window = 0
        # Audit fix P1.3: separate inline-VWMP exception count from legit
        # no-data null count. Tracks regression bugs in compute_vwmp path.
        self._cs_divergence_inline_error_count = 0
        # Audit fix (throttle): per-mint last-shadow time so observing path
        # samples at most once per cs_divergence_observing_per_mint_interval_sec.
        self._cs_observing_last_t: Dict[str, float] = {}

        # Phase B.3 (2026-05-19) — PHANTOM-KILL peak-gate counters.
        # Tracks per-tier: peak update attempts, layer1 (Phantom Guard) rejects,
        # layer2 (B.3 cs gate) rejects, combined rejects (either layer), and
        # errors. Cumulative + window pattern per Week 4 audit lessons.
        self._cs_peak_gate_attempt_count = 0           # all peak update attempts
        self._cs_peak_gate_layer1_reject_count = 0     # Phantom Guard rejected
        self._cs_peak_gate_layer2_reject_count = 0     # B.3 cs gate would reject
        self._cs_peak_gate_combined_reject_count = 0   # either gate rejected
        self._cs_peak_gate_b3_only_reject_count = 0    # B.3 caught what Phantom Guard missed
        self._cs_peak_gate_warmup_skip_count = 0       # gate skipped due to warmup
        self._cs_peak_gate_error_count = 0
        self._cs_peak_gate_attempt_count_window = 0
        self._cs_peak_gate_layer1_reject_count_window = 0
        self._cs_peak_gate_layer2_reject_count_window = 0
        self._cs_peak_gate_combined_reject_count_window = 0
        self._cs_peak_gate_b3_only_reject_count_window = 0
        self._cs_peak_gate_warmup_skip_count_window = 0

        # Phase B.4 (2026-05-20) — per-model inference divergence counters.
        # Cumulative + window pattern per Week 4 audit. Tracks: attempts,
        # decision agreement vs disagreement, new-path errors / nulls,
        # latency samples.
        self._cs_b4_v5_5_3_attempt_count = 0
        self._cs_b4_v5_5_3_agreement_count = 0
        self._cs_b4_v5_5_3_disagreement_count = 0
        self._cs_b4_v5_5_3_new_null_count = 0
        self._cs_b4_v5_5_2_attempt_count = 0
        self._cs_b4_v5_5_2_agreement_count = 0
        self._cs_b4_v5_5_2_disagreement_count = 0
        self._cs_b4_v5_5_2_new_null_count = 0
        self._cs_b4_old_compute_error_count = 0
        self._cs_b4_new_compute_error_count = 0
        self._cs_b4_log_error_count = 0
        # Window counters (reset on emit)
        self._cs_b4_v5_5_3_attempt_count_window = 0
        self._cs_b4_v5_5_3_agreement_count_window = 0
        self._cs_b4_v5_5_3_disagreement_count_window = 0
        self._cs_b4_v5_5_3_new_null_count_window = 0
        self._cs_b4_v5_5_2_attempt_count_window = 0
        self._cs_b4_v5_5_2_agreement_count_window = 0
        self._cs_b4_v5_5_2_disagreement_count_window = 0
        self._cs_b4_v5_5_2_new_null_count_window = 0
        # Phase B.4 V5.5.6+ extension counters (2026-05-22)
        # 8 counters/model × 3 models (V5.5.6 / V5.5.7 v2.1 / V5.6) = 24 total
        # Cumulative + window pattern per V5.5.3 B.4 precedent.
        self._cs_b4_v5_5_6_attempt_count = 0
        self._cs_b4_v5_5_6_agreement_count = 0
        self._cs_b4_v5_5_6_disagreement_count = 0
        self._cs_b4_v5_5_6_new_null_count = 0
        self._cs_b4_v5_5_6_attempt_count_window = 0
        self._cs_b4_v5_5_6_agreement_count_window = 0
        self._cs_b4_v5_5_6_disagreement_count_window = 0
        self._cs_b4_v5_5_6_new_null_count_window = 0
        self._cs_b4_v5_5_7_v2_1_attempt_count = 0
        self._cs_b4_v5_5_7_v2_1_agreement_count = 0
        self._cs_b4_v5_5_7_v2_1_disagreement_count = 0
        self._cs_b4_v5_5_7_v2_1_new_null_count = 0
        self._cs_b4_v5_5_7_v2_1_attempt_count_window = 0
        self._cs_b4_v5_5_7_v2_1_agreement_count_window = 0
        self._cs_b4_v5_5_7_v2_1_disagreement_count_window = 0
        self._cs_b4_v5_5_7_v2_1_new_null_count_window = 0
        self._cs_b4_v5_6_attempt_count = 0
        self._cs_b4_v5_6_agreement_count = 0
        self._cs_b4_v5_6_disagreement_count = 0
        self._cs_b4_v5_6_new_null_count = 0
        self._cs_b4_v5_6_attempt_count_window = 0
        self._cs_b4_v5_6_agreement_count_window = 0
        self._cs_b4_v5_6_disagreement_count_window = 0
        self._cs_b4_v5_6_new_null_count_window = 0

        self._grpc_stream = None
        grpc_url = config.grpc_url or os.environ.get("CHAINSTACK_GRPC_URL", "")
        grpc_token = config.grpc_token or os.environ.get("CHAINSTACK_GRPC_TOKEN", "")
        if grpc_url and HAS_GEYSER:
            self._grpc_stream = GeyserPumpSwapStream(
                grpc_url=grpc_url,
                grpc_token=grpc_token,
                graduation_queue=self.discovery._ws_queue,
                kline_builder=self.kline_builder,
                seen_mints=self.discovery._seen_mints,
                rug_event_queue=self._rug_event_queue,
                rug_sell_threshold_sol=config.rug_event_sol_threshold,
                price_event_queue=self._price_event_queue,
                price_event_min_delta=float(config.grpc_event_min_price_delta),
                cleaning_service_bridge=self._cleaning_service_bridge,
            )
            self.discovery.set_grpc_stream(self._grpc_stream)
            logger.info(f"M1/M2: Yellowstone gRPC stream configured ({grpc_url})")
        elif grpc_url and not HAS_GEYSER:
            logger.warning("M1/M2: GRPC_URL set but geyser_stream not importable "
                           "(proto stubs missing? restart container)")

        # Trade database
        self.db = TradeDB()

        # Restore BigWinner canary trade count from
        # completed trades (survives restart). Must happen after self.db is
        # initialized. Counter is bumped after each successful big_winner buy.
        if self.big_winner_loaded:
            try:
                sources = {
                    self._big_winner_entry_source,
                    self._big_winner_model_version,
                    "big_winner_v2",
                    "big_winner_v1",
                }
                self._big_winner_trade_count = sum(
                    self.db.count_trades_by_entry_source(source)
                    for source in sources if source
                )
                cap = (config.big_winner_canary_trade_limit
                       if config.big_winner_canary_trade_limit > 0 else "∞")
                logger.info(
                    f"BigWinnerV2: canary counter restored {self._big_winner_trade_count}/{cap}")
            except Exception as e:
                logger.warning(f"BigWinnerV2: canary count restore failed: {e}")
                self._big_winner_trade_count = 0

        # State
        self._positions: List[Position] = []
        self._pending: List[GraduatedToken] = []  # waiting for feature_delay_sec
        self._candidate_queue: List[TradeCandidate] = []  # PASS candidates waiting for cooldown
        self._trade_log: List[TradeRecord] = []  # capped at 500 entries; all trades persisted in DB
        self._trade_log_max: int = 500
        self._candidates_seen: int = 0
        self._stoploss_blacklist: Dict[str, float] = {}  # mint_address -> blacklist_until timestamp
        self._pool_discovery_attempts: Dict[str, int] = {}  # mint -> GMGN pool discovery attempt count
        self._observation_pool: Dict[str, ObservationEntry] = {}  # mint -> observation for 30min data collection
        # Mints whose swap collector should keep running past M2 evaluation
        # (because the token is heading to a real trade and we want full
        # hold-period swap data for P12-D training).
        # Cleared on trade exit OR on preflight/buy failure.
        self._swap_keepalive_mints: set = set()
        # mint -> deadline_ts. Default-collect raw swaps for all graduated tokens
        # for swap_collection_window_sec (30min) regardless of trade decision.
        # Heartbeat sweeps expired entries → _flush_swaps_to_db(unregister=True).
        # If mint enters _swap_keepalive_mints (real trade), the keepalive set
        # takes precedence and the deadline is ignored until trade exit.
        self._swap_collection_until: Dict[str, float] = {}
        # Phase 16.3 Step 3 audit fix — sell-in-progress mutex set.
        # rug_event_watcher and M4 monitor can both reach _execute_sell
        # concurrently. The mutex serializes them to prevent double-sell.
        # Phase 5-1: remaining ExitStateTracker aliases (same object as tracker).
        # All .add() / .discard() / [k]= / .pop() / `in` sites work unchanged.
        self._sell_in_progress        = self._exit_state.sell_in_progress        # Set[str]
        self._sell_retry_after        = self._exit_state.sell_retry_after        # Dict[str, float] mint -> earliest retry ts
        self._sell_retry_count        = self._exit_state.sell_retry_count        # Dict[str, int]   mint -> consecutive fail count
        self._sell_retry_notice_after = self._exit_state.sell_retry_notice_after # Dict[str, float] mint -> next retry-wait log ts
        self._exit_signal_reason      = self._exit_state.exit_signal_reason      # Dict[str, str]   mint -> last emitted trigger reason
        self._price_miss_count        = self._exit_state.price_miss_count        # Dict[str, int]   mint -> consecutive price fetch failures
        self._shadow_pending: Dict[str, Dict] = {}  # mint -> shadow eval payload
        self._shadow_ev3m_pending: Dict[str, Dict] = {}  # mint -> EV3m shadow payload
        self._shadow_super_winner_event_pending: Dict[str, Dict] = {}  # mint -> event shadow payload
        self._ev3m_live_audit: Dict[str, Dict] = {}  # mint -> latest live EV3m decision context
        self._first_price_seen: set = set()  # mints with first live price seen after buy
        self._sol_price_usd: float = 0.0
        self._last_sol_price_update: float = 0.0
        self._tick_count: int = 0
        # Q2 fix (2026-05-06): time-based checkpoint flush. Tick rate in
        # production is ~1 tick / 3.6s, so the old `tick_count % 180 == 1`
        # logic flushed every ~10.8 min instead of the documented 3 min.
        # Time-based gives consistent cadence regardless of heartbeat speed.
        self._last_checkpoint_flush_ts: float = 0.0
        self._last_poll_time: float = 0.0  # rate-limit M1 GMGN polling
        self._last_swap_poll_time: float = 0.0  # rate-limit on-chain swap polling
        self._last_hot_rank_check_time: float = 0.0  # rate-limit M1 coverage cross-check
        self._last_monitor_time: float = 0.0  # rate-limit M4 quote polling
        # 14y shadow-exit-warn — per-mint last-tick timestamp (rate-limit
        # inference to `shadow_exit_warn_tick_interval` seconds).
        self._last_shadow_exit_warn_ts: Dict[str, float] = {}
        self._shadow_exit_warn_resolver_task: Optional[asyncio.Task] = None
        self._shadow_exit_warn_last_resolve: float = 0.0
        # Lazy-imported inference module (None if shadow_exit_warn disabled)
        self._exit_models_mod = None
        # 19c sw_v2 shadow: dedup per mint (one F1 score per graduation).
        self._sw_v2_scored: Set[str] = set()
        self._sw_v2_mod = None
        # 2026-06-28 pre-grad factor shadow: dedup per mint (one L1 row / grad).
        self._pregrad_factors_scored: Set[str] = set()
        self._shadow_sw_v2_resolver_task: Optional[asyncio.Task] = None
        # Phase 25o (2026-05-04): Simple T+5m entry — dedup per mint
        self._simple_t5m_seen: Set[str] = set()

    # ──────────────────────────────────────────
    # Framework overrides
    # ──────────────────────────────────────────

    async def update_processed_data(self):
        """M1 + M2: discover new tokens and generate trade candidates."""
        now = time.time()

        # M1: poll new graduations (rate-limited by graduation_poll_interval)
        if now - self._last_poll_time >= self.config.graduation_poll_interval:
            self._last_poll_time = now
            try:
                new_tokens = await self.discovery.poll_new_graduations()
                if new_tokens:
                    logger.info(f"M1: discovered {len(new_tokens)} new token(s), adding to pending")
                    # Register tokens with on-chain kline builder.
                    # Accept up to feature_delay_sec + 300s old (P10 backfill can recover history).
                    # Beyond that, the token has been moving for too long for our model's
                    # training distribution to apply.
                    max_age = self.config.feature_delay_sec + 300  # 660s = 11min
                    for token in new_tokens:
                        age = now - token.graduation_time
                        # Async enqueue v3b creator_resolver — closes snapshot
                        # gap window so v3b can predict on this new mint
                        # (instead of SKIP_CAP_HIT_PENDING). Idempotent.
                        if self.creator_resolver is not None:
                            self.creator_resolver.enqueue(token.mint_address)
                        if token.pool_address and self.kline_builder and age < max_age:
                            self.kline_builder.register(token.mint_address, token.pool_address)
                            self._swap_collection_until.setdefault(
                                token.mint_address,
                                token.graduation_time + self.config.swap_collection_window_sec)
                            # Phase 22.D Route X RC2 (2026-05-01): backfill if
                            # age > 60s OR if gRPC pre-register failed (very few
                            # swaps in buffer). Previous gate (age>60 only)
                            # missed the case where gRPC pre-register raised an
                            # exception silently, leaving tokens with no swap
                            # history when M1 finally registered them.
                            existing_count = self.kline_builder.get_swap_count(
                                token.mint_address)
                            should_backfill = (age > 60) or (
                                age > 15 and existing_count < 5)
                            if should_backfill:
                                try:
                                    target_span = int(age) + 30  # cover from grad to now
                                    count = await self.kline_builder.backfill_swaps(
                                        token.mint_address, target_span_sec=target_span, max_pages=15)
                                    # Log span coverage to detect partial backfill
                                    swaps = self.kline_builder.get_swaps(token.mint_address)
                                    if swaps:
                                        actual_span = swaps[-1].timestamp - swaps[0].timestamp
                                        coverage = (actual_span / target_span * 100) if target_span > 0 else 0
                                        logger.info(f"M1: {token.symbol} late-detect (age={age:.0f}s) "
                                                    f"backfilled {count} swaps, span={actual_span:.0f}s "
                                                    f"({coverage:.0f}% of target)")
                                    else:
                                        logger.warning(f"M1: {token.symbol} late-detect (age={age:.0f}s) "
                                                       f"backfill returned {count} swaps but get_swaps empty")
                                except Exception as e:
                                    logger.warning(f"M1: {token.symbol} backfill failed: {e}")
                    self._pending.extend(new_tokens)
                    # Phase 25o (2026-05-04 fix): when V-shape disabled,
                    # tokens never reach observation pool via M2 reject paths,
                    # so RugFilter v3 (which only scores tokens in obs pool)
                    # never fires → SIMPLE-T5M at T+5m fails closed on
                    # "no rug_v3 score". Eagerly add to obs pool here so
                    # RugFilter v3 scoring fires at T+60s as designed.
                    if self.config.simple_t5m_enabled:
                        for _t in new_tokens:
                            self._add_to_observation_pool(
                                _t, None, False, "simple_t5m_pending")
            except Exception as e:
                logger.error(f"M1 poll failed: {e}", exc_info=True)

        # M1b: trending discovery REMOVED 2026-04-20.
        # Retired per CLAUDE.md: 6/6 historical trending picks were rugs,
        # and chainstack graduation detection has ~100% coverage of the
        # tokens worth trading. Keeping the trending feature added dead
        # code paths and GMGN API surface without any PnL upside.

        # M1c: hot rank cross-check (every 60s)
        # Compares GMGN hot rank against M1 discoveries to measure coverage gaps.
        # NEW: auto-promotes m1_missed tokens that pass bonding curve + safety validation.
        if now - self._last_hot_rank_check_time >= 60:
            self._last_hot_rank_check_time = now
            try:
                hot_items = await self.discovery.fetch_hot_rank_raw(max_age_sec=1800)
                if hot_items:
                    matched = 0
                    missed = 0
                    promoted = 0
                    active_mints = {t.mint_address for t in self._pending}
                    active_mints.update(p.token.mint_address for p in self._positions)
                    for rank_pos, item in enumerate(hot_items, start=1):
                        mint = item.get("address", "")
                        if not mint:
                            continue
                        symbol = str(item.get("symbol", ""))[:32]
                        ots = float(item.get("open_timestamp") or item.get("creation_timestamp") or 0)
                        age = int(now - ots) if ots > 0 else -1
                        disc = self.db.lookup_discovery(mint)
                        if disc:
                            match_status = "matched"
                            m1_first_seen, m1_passed = disc[0], bool(disc[1])
                            matched += 1
                        else:
                            match_status = "m1_missed"
                            m1_first_seen, m1_passed = None, None
                            missed += 1
                        try:
                            self.db.record_hot_rank_observation(
                                mint=mint, symbol=symbol, age_sec=age,
                                liquidity_usd=float(item.get("liquidity", 0) or 0),
                                volume_5m=float(item.get("volume", 0) or 0),
                                swaps_5m=int(item.get("swaps", 0) or 0),
                                smart_degen_count=int(item.get("smart_degen_count", 0) or 0),
                                rank_position=rank_pos, match_status=match_status,
                                m1_first_seen=m1_first_seen, m1_passed=m1_passed,
                            )
                        except Exception as e:
                            logger.debug(f"M1c: record failed for {mint[:12]}: {e}")

                        # Auto-promote m1_missed tokens (max 3 per cycle)
                        if match_status != "m1_missed" or promoted >= 3:
                            continue
                        if age < 0 or age > 600:
                            continue
                        liq = float(item.get("liquidity", 0) or 0)
                        if liq < self.config.min_liquidity_usd:
                            continue
                        if mint in active_mints:
                            continue

                        # Validate via token info API (same as M1b trending)
                        validation = await self.discovery.validate_trending_token(
                            mint, min_liq_usd=self.config.min_liquidity_usd)
                        if not validation:
                            continue
                        pool = validation["pool_address"]

                        token = GraduatedToken(
                            mint_address=mint, symbol=symbol,
                            name=item.get("name", symbol), decimals=6,
                            graduation_time=ots, liquidity_usd=validation["real_liquidity"],
                            price_usd=float(item.get("price", 0) or 0),
                            pool_address=pool, source="hot_rank",
                            _gmgn_info_raw=item,
                        )

                        # Async enqueue v3b creator_resolver — M1c hot_rank
                        # promotion path needs the same enqueue as M1 main loop
                        # (line 1583) so v3b can predict on this mint at +600s.
                        # Idempotent — safe if resolver already saw this mint.
                        if self.creator_resolver is not None:
                            self.creator_resolver.enqueue(mint)

                        # Register kline builder + backfill (same as M1b)
                        if self.kline_builder:
                            self.kline_builder.register(mint, pool)
                            self._swap_collection_until.setdefault(
                                mint, ots + self.config.swap_collection_window_sec)
                            try:
                                count = await self.kline_builder.backfill_swaps(
                                    mint, target_span_sec=300, max_pages=10)
                                logger.info(f"M1c: {symbol} backfilled {count} swaps")
                            except Exception as e:
                                logger.warning(f"M1c: {symbol} backfill failed: {e}")
                                continue

                        token_age = now - token.graduation_time
                        if token_age >= self.config.feature_delay_sec:
                            token.graduation_time = now - self.config.feature_delay_sec - 1
                        self._pending.append(token)
                        # Phase 25o (2026-05-04 fix): see same rationale at
                        # M1 poll site. Add to obs pool so RugFilter v3 fires.
                        if self.config.simple_t5m_enabled:
                            self._add_to_observation_pool(
                                token, None, False, "simple_t5m_pending")
                        self.discovery._seen_mints.setdefault(mint)
                        active_mints.add(mint)
                        promoted += 1
                        logger.info(
                            f"M1c: [PROMOTED] {symbol} | mint={mint} | "
                            f"liq=${validation['real_liquidity']:.0f} | age={age}s | "
                            f"pool={pool[:12]}... | source=hot_rank"
                        )

                    logger.info(f"M1c: hot_rank — {len(hot_items)} candidates, "
                                f"matched={matched}, m1_missed={missed}, promoted={promoted}")
            except Exception as e:
                logger.warning(f"M1c: hot_rank check failed: {e}")

        # Poll on-chain swap data for pending tokens AND tokens in active
        # trade lifecycle (M2-passed → buy → hold → sell). Rate-limited by
        # swap_poll_interval.
        #
        # Bug fix (2026-04-08): the swap collector used to stop after M2
        # evaluation, leaving 0 swap data for the actual trade hold period.
        # P12-D calibration showed 0/46 trades had any local swap data
        # covering their hold window. Now we keep collecting through entry,
        # hold, and exit so the local DB has the full price path.
        if (self.kline_builder
                and (self._pending or self._swap_keepalive_mints)
                and now - self._last_swap_poll_time >= self.config.swap_poll_interval):
            self._last_swap_poll_time = now

            # Build the union of mints we want to poll. Use a dict so each
            # mint maps to a display label for logs (symbol if available).
            poll_targets: Dict[str, str] = {}
            for token in self._pending:
                poll_targets[token.mint_address] = token.symbol
            for pos in self._positions:
                if pos.token.mint_address in self._swap_keepalive_mints:
                    poll_targets[pos.token.mint_address] = pos.token.symbol
            # Mints that passed M2 but haven't entered position yet (queued)
            # — also keep collecting for them.
            for cand in self._candidate_queue:
                if cand.token.mint_address in self._swap_keepalive_mints:
                    poll_targets.setdefault(cand.token.mint_address, cand.token.symbol)

            for mint, symbol in poll_targets.items():
                if not self.kline_builder.is_registered(mint):
                    continue
                # Discover additional pools (e.g. Meteora) via GMGN biggest_pool_address
                # Retry up to 3 times (GMGN may have indexing delay for new pools)
                if self.kline_builder.get_pool_count(mint) == 1 \
                        and self._pool_discovery_attempts.get(mint, 0) < 3:
                    try:
                        self._pool_discovery_attempts[mint] = \
                            self._pool_discovery_attempts.get(mint, 0) + 1
                        biggest = await self.discovery.discover_additional_pools(mint)
                        if biggest and biggest != self.kline_builder.get_pool_address(mint):
                            self.kline_builder.register(mint, biggest)
                            self._pool_discovery_attempts[mint] = 99  # stop retrying
                            logger.info(
                                f"KlineBuilder: {symbol} added pool {biggest[:12]}... "
                                f"(multi-DEX, now {self.kline_builder.get_pool_count(mint)} pools)"
                            )
                    except Exception as e:
                        logger.debug(f"KlineBuilder: pool discovery failed for {symbol}: {e}")
                # Skip RPC polling when gRPC Geyser is pushing swaps in real-time
                grpc_active = self._grpc_stream and self._grpc_stream.connected
                if not grpc_active:
                    try:
                        count = await self.kline_builder.poll_swaps(mint)
                        if count > 0:
                            total = self.kline_builder.get_swap_count(mint)
                            pools = self.kline_builder.get_pool_count(mint)
                            logger.debug(f"KlineBuilder: {symbol} +{count} swaps (total={total}, pools={pools})")
                    except Exception as e:
                        logger.debug(f"KlineBuilder: poll failed for {symbol}: {e}")

        # M2: process pending tokens
        candidates = []
        still_pending = []
        evaluated_this_tick = False
        now = time.time()
        ready_delay_sec = int(self.config.feature_delay_sec)
        if self.config.ev3m_live_enabled:
            ready_delay_sec = max(ready_delay_sec, int(self.config.ev3m_live_entry_delay_sec))
        # v3.1 D scan starts at scan_min_sec (T+7min default).
        # When v3.1 enabled, ensure first M2 evaluation doesn't fire before
        # scan_min_sec, and pending tokens stay alive through scan_max_sec.
        # Defensive: check both canonical + legacy (Phase 25g 2026-05-03).
        if self._enabled_any(self.config, "vshape_entry_enabled", "vshape_v3_1_enabled"):
            ready_delay_sec = max(ready_delay_sec, int(self._cfg_value(self.config, "vshape_entry_scan_min_sec", "vshape_v3_1_scan_min_sec", default=420)))

        for token in self._pending:
            age = now - token.graduation_time

            # ── 19c sw_v2 (F1) shadow score at T+180s ──
            # OBSERVE_ONLY: once per graduation, when the token reaches age
            # >= shadow_sw_v2_entry_offset_sec, compute the F1 entry score
            # from the recent swap buffer and log it to shadow_sw_v2_evals.
            # Non-blocking: any failure is swallowed.
            if (self.config.shadow_sw_v2_enabled
                    and age >= self.config.shadow_sw_v2_entry_offset_sec
                    and token.mint_address not in self._sw_v2_scored):
                self._run_sw_v2_shadow_once(token, now)
                self._sw_v2_scored.add(token.mint_address)

            # ── pre-grad bonding-curve factor shadow (L1 universe rug-screen) ──
            # OBSERVE_ONLY: once per graduation, RPC-backfill the pre-grad
            # bonding-curve PDA swap tape, compute 3 framework-parity pre-grad
            # factors + frozen median-vote, log the verdict. ASYNC RPC backfill
            # → fire-and-forget so the main loop never blocks on it. Default OFF;
            # never gates a buy. Dedup so the backfill runs once per mint even
            # before the ensure_future task has finished.
            if (self.config.pregrad_factors_shadow_enabled
                    and token.mint_address not in self._pregrad_factors_scored):
                self._pregrad_factors_scored.add(token.mint_address)
                asyncio.ensure_future(
                    self._run_pregrad_factors_shadow_once(token, now))

            # ── P13 INSTANT BUY MODE ──
            # Skip feature_delay entirely. Create a TradeCandidate with no
            # features/model score and push straight to the buy queue.
            # M2 kline collection continues in background for observation logging.
            #
            # Age filter (2026-04-10): latency decomposition on 24 live trades
            # showed a bimodal distribution — 15 "fast" tokens (WS real-time,
            # age 4-13s) and 9 "slow" tokens (WS backlog / M1 queue, age 40-210s).
            # The slow group is bot-restart backlog and M1 enrichment queuing,
            # NOT fresh graduations. Buying these at age > 30s means entering
            # well past the T+0 momentum window. Filter them out.
            if self.config.instant_buy_mode:
                # P13 Choice A: only trade WS-detected fresh tokens (< 15s).
                # Hot_rank tokens (age 40-200s) consistently have peak=0% because
                # momentum is gone by the time they arrive. 14/14 hot_rank instant
                # buys lost money. Only WS real-time detections have the T+0 edge.
                if age > 15:
                    logger.debug(f"M2-instant: {token.symbol} SKIP — too old "
                                 f"(age={age:.0f}s > 15s)")
                    self._add_to_observation_pool(
                        token, None, False, f"instant_buy_too_old_age={age:.0f}s")
                    continue

                # Still check stop-loss blacklist
                bl_until = self._stoploss_blacklist.get(token.mint_address)
                if bl_until and time.time() < bl_until:
                    logger.info(f"M2-instant: {token.symbol} SKIP — stoploss blacklisted")
                    self.db.record_discovery(token, None, False, "stoploss_blacklisted")
                    continue

                logger.info(f"M2-instant: {token.symbol} — INSTANT BUY mode, "
                            f"age={age:.1f}s, skipping feature_delay+M2 eval")
                self._record_latency_event(token, "instant_candidate_queued", age_at_queue_sec=age)
                candidate = TradeCandidate(
                    token=token,
                    model_score=0.0,
                    features={"instant_buy": True, "age_at_queue_sec": age},
                )
                candidates.append(candidate)
                self._candidates_seen += 1
                # Add to observation pool for M6 logging (runs in background)
                self._add_to_observation_pool(
                    token, 0.0, True, "instant_buy_mode",
                    features={"instant_buy": True})
                self.db.record_discovery(token, 0.0, True, "instant_buy_mode")
                continue

            # ── xoscross 穿越锚 7 臂(DEPLOY_SPEC_xoscross,2026-07-05;默认 OFF)──
            # shadow_only=True 阶段:仅 7 臂决策日志,零真金;live admit + shadow_only=False
            # 才走下方 M2 安全链(safety 硬保护不可跳)。fail-safe:异常决不阻塞主循环。
            # source gate(2026-07-05 对抗审计 #3):M1b/M1c 的 GMGN/hot_rank 锚
            # 是合成 graduation_time(M1c 甚至改写 grad=now−300−1)→ 桶边界/warmup
            # 全走样且研究宇宙(链上毕业检测)不含它们 —— xoscross 只吃主路 cohort。
            if (self.config.xoscross_enabled
                    and getattr(token, "source", "") == "chainstack"
                    and age >= 60
                    and age <= self.config.xoscross_max_age_sec):
                if getattr(self, "_xoscross_state", None) is None:
                    self._xoscross_state = _xoscross.XoscrossState(
                        top_frac=self.config.xoscross_bottomk_top_frac,
                        window=self.config.xoscross_bottomk_window,
                        cold_start_min=self.config.xoscross_bottomk_cold_start)
                    try:  # 修复①(2026-07-06):重启回灌 bottom-k 窗口,冷启动进度不清零
                        _seeded = _xoscross.warm_load_gates(
                            self._xoscross_state, self.db)
                        logger.info(
                            f"XOSCROSS: bottom-k warm-loaded {_seeded} "
                            f"(cold_start_min="
                            f"{self.config.xoscross_bottomk_cold_start})")
                    except Exception as _we:
                        logger.debug(f"xoscross warm-load skipped: {_we}")
                try:
                    _l1 = getattr(self, "_pregrad_l1_verdicts", {}).get(
                        token.mint_address)
                    _xos_admit = _xoscross.on_tick(self, self._xoscross_state,
                                                   token, now, l1_pass=_l1)
                except Exception as _xe:
                    logger.debug(f"xoscross tick {token.symbol}: {_xe}")
                    _xos_admit = False
                if _xos_admit and self.config.xoscross_shadow_only:
                    # 修复②:shadow_only 阶段 admit 的终判归因(不走真金链)
                    _xoscross.mark_live_action(self, token.mint_address,
                                               "shadow_only")
                if _xos_admit and not self.config.xoscross_shadow_only:
                    # SL 黑名单(其他入场路径都查;2026-07-05 审计补齐)
                    if token.mint_address in self._stoploss_blacklist and \
                            self._stoploss_blacklist[token.mint_address] > now:
                        logger.info(f"XOSCROSS: {token.symbol} REJECT — SL blacklist")
                        _xoscross.mark_live_action(self, token.mint_address,
                                                   "sl_blacklist")
                        continue
                    try:
                        kline_bars, kline_source, m2_features = (
                            await self._build_m2_kline_for_safety(token))
                        safety_ok, reject_reason = self._apply_m2_safety_filters(
                            token, m2_features, kline_source, source="xoscross")
                    except Exception as _xe:
                        safety_ok, reject_reason = False, f"m2_err_{_xe}"
                    if safety_ok:
                        logger.info(f"XOSCROSS: {token.symbol} ADMIT → candidate")
                        # 2026-07-05 审计 CRITICAL 修复:必须入队 TradeCandidate
                        # (裸 tuple 会让 control_task 的 TTL/排序消费端每 tick
                        # AttributeError,买单永不执行 —— 消费端契约见 :2545+)。
                        from controllers.generic.meme_sniper_utils import TradeCandidate
                        self._candidate_queue.append(TradeCandidate(
                            token=token,
                            model_score=0.0,
                            features={
                                "xoscross": True,
                                "kline_source": kline_source,
                                "age_at_entry_sec": int(age),
                            },
                            queued_at=now,
                            last_swap_price_sol=0.0,
                            entry_source="xoscross",
                            position_size_usd=float(
                                self.config.simple_t5m_position_size_usd),
                        ))
                        _xoscross.mark_live_action(self, token.mint_address,
                                                   "enqueued")
                    else:
                        logger.info(f"XOSCROSS: {token.symbol} REJECT — M2: {reject_reason}")
                        self._add_to_observation_pool(token, 0.0, False,
                                                      f"xoscross_m2_{reject_reason}")
                        _xoscross.mark_live_action(self, token.mint_address,
                                                   f"m2_reject:{reject_reason}")
                # xoscross 不消耗 pending(可 re-entry);与 simple_t5m 并行由 operator
                # 经 config 互斥使用(两者同开会双路入队,SPEC §1 已注明)。

            # ── Phase 25o (2026-05-04): Simple T+5m entry path ──
            # Bypass model gates entirely. At T+offset_sec, if RugFilter PASS
            # + M2 safety PASS → enqueue. V-shape and big_winner reduced to
            # shadow log status when this is enabled.
            if (self.config.simple_t5m_enabled
                    and token.mint_address not in self._simple_t5m_seen
                    and age >= self.config.simple_t5m_offset_sec
                    and age <= (self.config.simple_t5m_offset_sec
                                + self.config.simple_t5m_offset_window_sec)):
                self._simple_t5m_seen.add(token.mint_address)

                # RugFilter v3 entry-gate REMOVED 2026-05-06.
                # v4 shadow scoring runs in heartbeat (`_run_shadow_rug_filter_v4`)
                # but NOT as a gate during shadow phase. Promote to gate only
                # after 7-14 day shadow validation per CTA spec §5.2.

                # M2 hard safety filters (honeypot, top10, -85% crash, etc)
                try:
                    kline_bars, kline_source, m2_features = (
                        await self._build_m2_kline_for_safety(token))
                except Exception as e:
                    logger.info(
                        f"SIMPLE-T5M: {token.symbol} M2 kline build failed: "
                        f"{e} — skip")
                    self._add_to_observation_pool(
                        token, 0.0, False, "simple_t5m_kline_failed")
                    continue
                safety_ok, reject_reason = self._apply_m2_safety_filters(
                    token, m2_features, kline_source,
                    source=self.config.simple_t5m_entry_source)
                if not safety_ok:
                    logger.info(
                        f"SIMPLE-T5M: {token.symbol} REJECT — "
                        f"M2 safety: {reject_reason}")
                    self._add_to_observation_pool(
                        token, 0.0, False, f"simple_t5m_m2_{reject_reason}")
                    continue

                # B7 fix (2026-05-09): non-negative slope pre-filter.
                # Live audit found 53% of trades had peak_pnl_pct ≤ 0
                # (never went up). Reject entries where price slope in last
                # 60s is < -5% — these tokens are already in active dump.
                stream = getattr(self, "_grpc_stream", None)
                slope_ok = True
                slope_pct = None
                if (self.config.simple_t5m_min_slope_enabled
                        and stream is not None):
                    try:
                        records = stream.get_recent_swap_records(
                            token.mint_address, max_age_sec=60.0)
                        if records and len(records) >= 5:
                            # Sort by ts to be safe
                            recs_sorted = sorted(records, key=lambda r: r.timestamp)
                            # Use median of first 3 vs median of last 3 to avoid
                            # single-tick spike/dip distortion
                            from statistics import median
                            first_med = median(r.price_sol for r in recs_sorted[:3])
                            last_med = median(r.price_sol for r in recs_sorted[-3:])
                            if first_med > 0:
                                slope_pct = (last_med - first_med) / first_med
                                lo = float(self.config.simple_t5m_min_slope_pct)
                                hi = float(self.config.simple_t5m_max_slope_pct)
                                if slope_pct < lo:
                                    slope_ok = False
                                    slope_reject_reason = "active_dump"
                                elif slope_pct > hi:
                                    # B8 fix: pump too steep → likely pump-dump trap
                                    slope_ok = False
                                    slope_reject_reason = "pump_top"
                    except Exception as e_slope:
                        logger.debug(
                            f"SIMPLE-T5M: {token.symbol} slope check failed: {e_slope} — allow")
                if not slope_ok:
                    reason = locals().get("slope_reject_reason", "active_dump")
                    if reason == "active_dump":
                        msg = (f"slope_60s={slope_pct*100:+.1f}% < "
                               f"{self.config.simple_t5m_min_slope_pct*100:+.1f}% "
                               f"(token actively dumping)")
                    else:  # pump_top
                        msg = (f"slope_60s={slope_pct*100:+.1f}% > "
                               f"+{self.config.simple_t5m_max_slope_pct*100:.0f}% "
                               f"(pump-dump trap)")
                    logger.info(f"SIMPLE-T5M: {token.symbol} REJECT — {msg}")
                    self._add_to_observation_pool(
                        token, 0.0, False,
                        f"simple_t5m_slope_{reason}_{slope_pct*100:+.0f}pct")
                    continue

                # ── RugFilter v4.3 hard-gate (2026-05-14) ──
                # Wired only when shadow_rug_v4_3_shadow_only=false. When
                # active, query the latest eval written by the T+305s heartbeat
                # in `_run_shadow_rug_filter_v4_3`. If no eval yet (SIMPLE-T5M
                # fires at age 300-304s before heartbeat catches up — observed
                # in 100% of post-deploy gate firings), score ON-DEMAND so the
                # gate is never silently bypassed.
                v4_3_gate_active = (
                    getattr(self.config, "shadow_rug_v4_3_enabled", False)
                    and not getattr(self.config, "shadow_rug_v4_3_shadow_only", True)
                    and self.rug_filter_v4_3_model is not None
                )
                v4_3_decision: Optional[str] = None
                v4_3_score_s: str = "NaN"
                v4_3_cutoff_s: str = "?"
                if v4_3_gate_active:
                    try:
                        eval_row = self.db.get_latest_shadow_rug_v4_3_eval(
                            token.mint_address)
                    except Exception as _e_v43:
                        logger.debug(
                            f"SIMPLE-T5M: {token.symbol} v4.3 DB lookup failed: "
                            f"{_e_v43}")
                        eval_row = None

                    if eval_row:
                        v4_3_decision = eval_row["decision"]
                        if eval_row.get("score") is not None:
                            v4_3_score_s = f"{eval_row['score']:.3f}"
                        v4_3_cutoff_s = f"{eval_row['cutoff']:.3f}"
                    else:
                        # ── On-demand fallback: score synchronously now ──
                        # Mirrors `_run_shadow_rug_filter_v4_3` body but skips
                        # the "if elapsed < window_s+5" gate (we already know
                        # age >= 300s, close enough). Writes to DB so the
                        # heartbeat won't re-score and offline analysis sees
                        # the same row regardless of which path triggered.
                        try:
                            db_path = self.db.db_path if hasattr(self.db, "db_path") else None
                            if db_path:
                                try:
                                    self._flush_swaps_to_db(
                                        token.mint_address, unregister=False)
                                except Exception as _flush_err:
                                    logger.debug(
                                        f"SIMPLE-T5M: {token.symbol} v4.3 pre-score "
                                        f"flush failed: {_flush_err}")
                                result = self.rug_filter_v4_3_model.score_from_sqlite(
                                    db_path=db_path,
                                    mint_address=token.mint_address,
                                    graduation_time=token.graduation_time,
                                )
                                window_s = int(self.config.shadow_rug_v4_3_window_sec)
                                elapsed = age
                                self.db.record_shadow_rug_v4_3_eval(
                                    mint_address=token.mint_address,
                                    symbol=token.symbol,
                                    graduation_time=token.graduation_time,
                                    scored_at_delay_s=elapsed,
                                    window_s=window_s,
                                    n_swaps=result.n_swaps,
                                    score=(None if result.score != result.score
                                           else result.score),
                                    cutoff=self.rug_filter_v4_3_model.cutoff,
                                    decision=result.decision,
                                    reason=result.reason,
                                    features=result.features or None,
                                    model_version=self.rug_filter_v4_3_model.VERSION,
                                )
                                v4_3_decision = result.decision
                                if result.score == result.score:
                                    v4_3_score_s = f"{result.score:.3f}"
                                v4_3_cutoff_s = f"{self.rug_filter_v4_3_model.cutoff:.3f}"
                                # Mark scored on observation pool to prevent
                                # heartbeat re-scoring later in this hold cycle.
                                obs = self._observation_pool.get(token.mint_address)
                                if obs is not None:
                                    obs._rug_v4_3_scored = True
                                logger.info(
                                    f"RUG-V4_3-ONDEMAND: {token.symbol} "
                                    f"score={v4_3_score_s} cutoff={v4_3_cutoff_s} "
                                    f"n_swaps={result.n_swaps} "
                                    f"decision={result.decision}"
                                )
                        except Exception as _e_score:
                            logger.warning(
                                f"SIMPLE-T5M: {token.symbol} v4.3 on-demand "
                                f"scoring failed: {_e_score} — allow")
                            v4_3_decision = None  # safe degrade → allow

                    if v4_3_decision == "REJECT":
                        logger.info(
                            f"SIMPLE-T5M: {token.symbol} REJECT — "
                            f"rug_v4.3 score={v4_3_score_s} >= "
                            f"cutoff={v4_3_cutoff_s}"
                        )
                        self._add_to_observation_pool(
                            token, 0.0, False,
                            f"simple_t5m_v4_3_reject_{v4_3_score_s}")
                        continue

                # PASS — enqueue
                from controllers.generic.meme_sniper_utils import TradeCandidate
                candidate = TradeCandidate(
                    token=token,
                    model_score=0.0,
                    features={
                        "simple_t5m": True,
                        "kline_source": kline_source,
                        "age_at_entry_sec": int(age),
                    },
                    queued_at=now,
                    last_swap_price_sol=0.0,
                    entry_source=self.config.simple_t5m_entry_source,
                    position_size_usd=float(
                        self.config.simple_t5m_position_size_usd),
                )
                # v4.3 tag distinguishes 3 states for offline analysis:
                #   "v4.3 PASS"     — scored, below cutoff
                #   "v4.3 SKIPPED"  — scoring failed or insufficient data
                #   ""              — gate inactive (shadow_only)
                if not v4_3_gate_active:
                    v4_3_tag = ""
                elif v4_3_decision == "PASS":
                    v4_3_tag = f" v4.3 PASS ({v4_3_score_s}<{v4_3_cutoff_s})"
                else:
                    v4_3_tag = f" v4.3 SKIPPED ({v4_3_decision})"
                logger.info(
                    f"SIMPLE-T5M: {token.symbol} PASS — entered at T+"
                    f"{int(age)}s (M2 safety ✓{v4_3_tag})")
                candidates.append(candidate)
                self._candidates_seen += 1
                self._add_to_observation_pool(
                    token, 0.0, True, "simple_t5m_entered",
                    features=candidate.features)
                continue

            # ── Normal mode: wait for feature_delay_sec then evaluate ──
            # Drop tokens that are too old — features are stale.
            # When v3.1 D enabled, keep token alive through full scan window
            # (scan_max_sec + 60s grace) so rolling first-passage can complete.
            max_pending_age = ready_delay_sec + 300
            if self._enabled_any(self.config, "vshape_entry_enabled", "vshape_v3_1_enabled"):
                max_pending_age = max(max_pending_age,
                                      int(self._cfg_value(self.config, "vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec", default=900)) + 60)
            if self.config.xoscross_enabled:
                max_pending_age = max(max_pending_age,
                                      int(self.config.xoscross_max_age_sec) + 60)
            if age > max_pending_age:
                logger.debug(f"M2: dropping expired token {token.symbol} (age={age:.0f}s)")
                self._add_to_observation_pool(token, None, False, "expired_pending")
                # Clean up v3.1 scan state to bound memory
                self._vshape_entry_last_scan_ts.pop(token.mint_address, None)
                continue
            if age >= ready_delay_sec:
                # ── v3.1 D rolling-scan throttle (Phase 22 v3.1, 2026-04-29) ──
                # When v3.1 enabled, only invoke heavy evaluation every
                # scan_step_sec. Between scans, re-queue silently so other
                # tokens get airtime.
                if (self._enabled_any(self.config, "vshape_entry_enabled", "vshape_v3_1_enabled")
                        and self.vshape_entry_model is not None
                        and (now - self._vshape_entry_last_scan_ts.get(token.mint_address, 0.0))
                            < int(self._cfg_value(self.config, "vshape_entry_scan_step_sec", "vshape_v3_1_scan_step_sec", default=30))):
                    still_pending.append(token)
                    continue
                if not evaluated_this_tick:
                    candidate = await self._evaluate_token(token)
                    evaluated_this_tick = True
                    if candidate:
                        # ── v3.1 D first-passage gate (Phase 22 v3.1, 2026-04-29) ──
                        # Replaces v1.6.1 (which had detect_vshape_live last-bar
                        # leakage bug). Rolling scan in [scan_min_sec, scan_max_sec]
                        # at scan_step_sec cadence. First-passage: fire on first
                        # cross above selection_band cutoff. Below cutoff →
                        # re-queue for next 30s anchor.
                        if (self._enabled_any(self.config, "vshape_entry_enabled", "vshape_v3_1_enabled")
                                and self.vshape_entry_model is not None):
                            try:
                                self._vshape_entry_last_scan_ts[token.mint_address] = now
                                # Quantize current age to scan_step boundary,
                                # clamp to [scan_min, scan_max]
                                step = int(self._cfg_value(self.config, "vshape_entry_scan_step_sec", "vshape_v3_1_scan_step_sec", default=30))
                                anchor_sec = int((age // step) * step)
                                anchor_sec = max(int(self._cfg_value(self.config, "vshape_entry_scan_min_sec", "vshape_v3_1_scan_min_sec", default=420)),
                                                  anchor_sec)
                                anchor_sec = min(int(self._cfg_value(self.config, "vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec", default=900)),
                                                  anchor_sec)
                                # Build kline up to anchor
                                window_min = max(1, anchor_sec // 60)
                                n_bars_v = max(window_min + 2, 7)
                                kline_bars_v3, _ = self.kline_builder.build_kline(
                                    mint=token.mint_address,
                                    start_ts=int(token.graduation_time),
                                    n_bars=n_bars_v, resolution=60,
                                    sol_price_usd=self._sol_price_usd
                                                  if self._sol_price_usd > 0 else 80.0,
                                )
                                v3_passed = False
                                v3_score = 0.0
                                v3_pattern = "none"
                                if kline_bars_v3 and len(kline_bars_v3) >= 5:
                                    vf3 = detect_vshape_live(
                                        kline_bars_v3, token.graduation_time,
                                        entry_offset_sec=anchor_sec)
                                    if vf3 and vf3.get("any_pattern", 0) == 1:
                                        swaps_raw_v3_unfiltered = self.db.get_swaps_for_token(
                                            token.mint_address)
                                        # Byte-parity with build_vshape_v3_1_panel.py
                                        # build_swap_dicts: drop tiny / non-positive
                                        # price swaps before feature compute.
                                        swaps_raw_v3 = [
                                            s for s in swaps_raw_v3_unfiltered
                                            if float(s.get("sol_amount", 0)) >= 0.001
                                               and float(s.get("price_sol", 0) or 0) > 0
                                        ]
                                        # Sliding 5-min m5_* window ending AT anchor.
                                        # Trick: pass anchor_window_start as fake
                                        # graduation_time so compute_micro_live's
                                        # [grad, grad+window] filter picks
                                        # [anchor-300, anchor]. MUST byte-match the
                                        # panel build (build_vshape_v3_1_panel.py).
                                        anchor_window_start = (
                                            float(token.graduation_time) + anchor_sec - 300)
                                        sliding_swaps = [
                                            s for s in swaps_raw_v3
                                            if anchor_window_start
                                                <= float(s.get("block_time", 0))
                                                <  float(token.graduation_time) + anchor_sec
                                        ]
                                        m5 = compute_micro_live(
                                            sliding_swaps, anchor_window_start,
                                            window_sec=300, feat_prefix="m5")
                                        # m_full normalization uses TRAINING anchor range
                                        # (300, 900), NOT scan range. This is byte-parity
                                        # with build_vshape_v3_1_panel.py which trained on
                                        # anchors {300, 360, ..., 900}. Even though live
                                        # only scans [scan_min_sec=420, scan_max_sec=900],
                                        # the m_full_anchor_age_norm feature must be
                                        # normalized identically to training.
                                        m_full = compute_micro_live_full(
                                            swaps_raw_v3, float(token.graduation_time),
                                            anchor_sec=anchor_sec,
                                            max_anchor_sec=900,
                                            min_anchor_sec=300,
                                        )
                                        if m5 and m_full:
                                            features_v3 = {**vf3, **m5, **m_full}
                                            # ===== v3.3 OOD-defense features (2026-04-30) =====
                                            # Byte-parity with v3_3_combined_panel.parquet:
                                            #   compute_all_features → sf_swap_density,
                                            #     hc_top3_share_t, tb_kyle_lambda_180s
                                            #   compute_phase16_entry_features → hk_branching_buy
                                            # Default FEATURE_WINDOW=180s matches training.
                                            try:
                                                from controllers.generic.meme_sniper_exit_models import (
                                                    compute_all_features as _cmp_all_v33)
                                                from controllers.generic.big_winner_features import (
                                                    compute_phase16_entry_features as _cmp_p16_v33)
                                            except ImportError:
                                                from meme_sniper_exit_models import (
                                                    compute_all_features as _cmp_all_v33)
                                                from big_winner_features import (
                                                    compute_phase16_entry_features as _cmp_p16_v33)
                                            anchor_t_abs = int(token.graduation_time) + int(anchor_sec)
                                            swaps_df_v33 = self.db.get_swaps_df_for_token(
                                                token.mint_address)
                                            ood_base = None
                                            ood_p16 = None
                                            if swaps_df_v33 is not None and len(swaps_df_v33) >= 8:
                                                visible_v33 = swaps_df_v33[
                                                    swaps_df_v33["block_time"] <= anchor_t_abs]
                                                if len(visible_v33) >= 8:
                                                    try:
                                                        ood_base = _cmp_all_v33(
                                                            visible_v33, anchor_t_abs,
                                                            int(token.graduation_time))
                                                        ood_p16 = _cmp_p16_v33(
                                                            visible_v33, anchor_t_abs,
                                                            int(token.graduation_time))
                                                    except Exception as _e_v33:
                                                        logger.warning(
                                                            f"M2-VSHAPE-ENTRY: {token.symbol} OOD compute err: {_e_v33}")
                                            if ood_base is None or ood_p16 is None:
                                                logger.info(
                                                    f"M2-VSHAPE-ENTRY: {token.symbol} anchor={anchor_sec}s "
                                                    f"OOD features unavailable (insufficient swaps), REJECT")
                                                candidate = None
                                                if age + step < int(self._cfg_value(self.config, "vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec", default=900)):
                                                    still_pending.append(token)
                                                continue
                                            features_v3["sf_swap_density"] = float(
                                                ood_base.get("sf_swap_density", 0.0))
                                            features_v3["hc_top3_share_t"] = float(
                                                ood_base.get("hc_top3_share_t", 0.0))
                                            features_v3["tb_kyle_lambda_180s"] = float(
                                                ood_base.get("tb_kyle_lambda_180s", 0.0))
                                            features_v3["hk_branching_buy"] = float(
                                                ood_p16.get("hk_branching_buy", 0.0))
                                            v3_score = float(
                                                self.vshape_entry_model.predict_score(features_v3))
                                            band_v3 = str(
                                                self.config.vshape_entry_selection_band
                                                or "top5_live").strip() or "top5_live"
                                            # Cutoff override (Phase 22 cutoff recalib 2026-04-30)
                                            override = float(
                                                getattr(self.config,
                                                        "vshape_entry_cutoff_override",
                                                        -1.0) or -1.0)
                                            if override > 0:
                                                cutoff_v3 = override
                                                band_v3 = f"override({override:.4f})"
                                            else:
                                                cutoff_v3 = float(
                                                    self.vshape_entry_model.selection_cutoffs.get(
                                                        band_v3, 0.7134))

                                            # ===== OOD pre-filter (2026-04-30) =====
                                            # v3.1 D was trained on Birdeye Tier 2 data which
                                            # excludes dead/low-volume tokens. On Chainstack live,
                                            # the model gives high scores to OOD low-volume tokens
                                            # (overfitting). humancoin (entered 2026-04-29 23:11)
                                            # had m5_unique_buyers=58 (training p10!) m5_buy_vol_total=35
                                            # SOL (training p10) yet scored 0.6781 - classic OOD.
                                            # Reject if features are below training p10 thresholds.
                                            ood_min_buyers = int(getattr(
                                                self.config,
                                                "vshape_entry_ood_min_buyers", 60))
                                            ood_min_buy_vol_sol = float(getattr(
                                                self.config,
                                                "vshape_entry_ood_min_buy_vol_sol", 30.0))
                                            ood_block = False
                                            ood_reason = ""
                                            m5_buyers = features_v3.get("m5_unique_buyers", 0) or 0
                                            m5_buy_vol = features_v3.get("m5_buy_vol_total", 0) or 0
                                            # Phase 22.S.OOD-cleanup (2026-05-01): removed
                                            # redundant `sf_swap_density < 30` check.
                                            # 7d Chainstack panel sensitivity sweep (n=955)
                                            # showed: m5_buyers≥60 AND m5_buy_vol≥30 catches
                                            # 100% of density<30 cases (no marginal samples
                                            # in density 15-29 range that pass m5 filters).
                                            # Removing the density check is byte-equivalent
                                            # in historical panel and unblocks rare LIVE
                                            # cases (e.g. MSGA: density=17 but m5 OK).
                                            # Source: §I/§J of /tmp/ood_sensitivity.py.
                                            if m5_buyers < ood_min_buyers:
                                                ood_block = True
                                                ood_reason = f"m5_unique_buyers={m5_buyers}<{ood_min_buyers}"
                                            elif m5_buy_vol < ood_min_buy_vol_sol:
                                                ood_block = True
                                                ood_reason = f"m5_buy_vol_total={m5_buy_vol:.1f}<{ood_min_buy_vol_sol:.0f}"
                                            # Phase 2.D — route raw score through ShadowGuard.gate() for
                                            # mode-aware enforcement. If meta.mode='shadow', decision.fired=False
                                            # blocks the fire regardless of cutoff. mode='canary'/'standard'
                                            # passes through (cutoff check remains binding). Closes the v4.3
                                            # 2026-05-14 footgun (feedback_shadow_only_flag_must_be_wired).
                                            _v3_decision = self.vshape_entry_model.guarded.gate(v3_score)
                                            v3_passed = (v3_score >= cutoff_v3) and (not ood_block) and _v3_decision.fired
                                            v3_pattern = (
                                                "vshape" if vf3.get("is_vshape") else
                                                "steady_up" if vf3.get("is_steady_up") else
                                                "near_high" if vf3.get("is_near_high") else
                                                "reversal" if vf3.get("is_reversal") else
                                                "unknown")
                                            ood_tag = f" OOD_BLOCK[{ood_reason}]" if ood_block else ""
                                            logger.info(
                                                f"M2-VSHAPE-ENTRY: {token.symbol} anchor={anchor_sec}s "
                                                f"pattern={v3_pattern} score={v3_score:.4f} "
                                                f"{'PASS' if v3_passed else 'REJECT'} "
                                                f"({band_v3}>={cutoff_v3:.4f}){ood_tag}")
                                            # Phase 25h (2026-05-03) — persist v3.4 shadow eval
                                            # for sim/live alignment audit. Captures EVERY scan
                                            # tick with score + features + decision so audit
                                            # can replay. Mirrors shadow_big_winner_evals.
                                            try:
                                                import json as _v3_json
                                                _v3_features_json = _v3_json.dumps(
                                                    {k: float(v) for k, v in features_v3.items()
                                                     if isinstance(v, (int, float))},
                                                    separators=(",", ":"))
                                            except Exception:
                                                _v3_features_json = None
                                            try:
                                                self.db.record_shadow_vshape_v3_4_eval(
                                                    mint_address=token.mint_address,
                                                    symbol=token.symbol,
                                                    graduation_time=token.graduation_time,
                                                    scan_t_sec=int(anchor_sec),
                                                    cutoff_value=float(cutoff_v3),
                                                    score=float(v3_score),
                                                    decision_pass=bool(v3_passed),
                                                    ood_pass=(not ood_block),
                                                    pattern_detected=True,
                                                    n_swaps=int(len(sliding_swaps)) if 'sliding_swaps' in locals() else 0,
                                                    entry_price_sol=None,
                                                    sol_price_usd=self._sol_price_usd,
                                                    model_version=str(getattr(
                                                        self.config, "vshape_entry_model_version",
                                                        "vshape_v3_4") or "vshape_v3_4"),
                                                    features_json=_v3_features_json,
                                                )
                                            except Exception as _v3log_e:
                                                logger.debug(
                                                    f"M2-VSHAPE-ENTRY: shadow log failed for "
                                                    f"{token.symbol}: {_v3log_e}")
                                        else:
                                            logger.info(
                                                f"M2-VSHAPE-ENTRY: {token.symbol} anchor={anchor_sec}s "
                                                f"insufficient micro features, REJECT")
                                    else:
                                        logger.info(
                                            f"M2-VSHAPE-ENTRY: {token.symbol} anchor={anchor_sec}s "
                                            f"no pattern detected, REJECT")
                                else:
                                    logger.info(
                                        f"M2-VSHAPE-ENTRY: {token.symbol} anchor={anchor_sec}s "
                                        f"insufficient bars, REJECT")

                                if v3_passed:
                                    candidate.model_score = v3_score
                                    # Phase 25k (2026-05-04): tag entry_source so the
                                    # require_entry_source fail-closed gate (Phase 25g)
                                    # at line ~2041 doesn't reject this candidate.
                                    # Without this, V-shape passes were silently dropped.
                                    candidate.entry_source = str(getattr(
                                        self.config, "vshape_entry_model_version",
                                        "vshape_v3_4") or "vshape_v3_4")
                                    if candidate.features is None:
                                        candidate.features = {}
                                    candidate.features["vshape_entry_score"] = v3_score
                                    candidate.features["vshape_entry_anchor_sec"] = anchor_sec
                                    candidate.features["vshape_entry_pattern"] = v3_pattern
                                    candidate.features["vshape_entry_band"] = band_v3
                                    candidate.features["vshape_entry_cutoff"] = cutoff_v3
                                    candidate.features["v3_1_score"] = v3_score
                                    candidate.features["v3_1_anchor_sec"] = anchor_sec
                                    candidate.features["v3_1_pattern"] = v3_pattern
                                    candidate.features["v3_1_band"] = band_v3
                                    candidate.features["v3_1_cutoff"] = cutoff_v3
                                else:
                                    self.db.record_discovery(
                                        token, v3_score, False,
                                        f"vshape_entry_reject_anchor={anchor_sec}_score={v3_score:.4f}",
                                        features=candidate.features)
                                    candidate = None
                                    # Re-queue for next 30s scan if still in window
                                    if age + step < int(self._cfg_value(self.config, "vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec", default=900)):
                                        still_pending.append(token)
                                        continue
                            except Exception as e:
                                logger.error(
                                    f"M2-VSHAPE-ENTRY: {token.symbol} gate error: {e}",
                                    exc_info=True)
                                candidate = None
                                if age + 60 < int(self._cfg_value(self.config, "vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec", default=900)):
                                    still_pending.append(token)
                                    continue

                    if candidate:
                        # ── EV3m live gate: frozen Phase 6.4 main contract ──
                        if self.ev3m_live_model is not None and self.config.ev3m_live_enabled:
                            try:
                                ev3m_features, ev3m_src = await self._collect_ev3m_features(
                                    token=token,
                                    delay_sec=int(self.config.ev3m_live_entry_delay_sec),
                                    model=self.ev3m_live_model,
                                )
                                if ev3m_features is not None:
                                    ev3m_score = float(self.ev3m_live_model.predict_score(ev3m_features))
                                    band = str(self.config.ev3m_live_selection_band or "top10").strip() or "top10"
                                    cutoff = self.ev3m_live_model.selection_cutoffs.get(band)
                                    if cutoff is None:
                                        logger.warning(
                                            f"M2: {token.symbol} EV3m live band {band} missing in artifact; "
                                            f"falling back to top10"
                                        )
                                        band = "top10"
                                        cutoff = self.ev3m_live_model.selection_cutoffs.get("top10", 0.67)
                                    self._cache_ev3m_live_audit(
                                        token,
                                        status="ok",
                                        feature_source=ev3m_src,
                                        age_sec=age,
                                        score=ev3m_score,
                                        band=band,
                                        cutoff=cutoff,
                                        passed=ev3m_score >= cutoff,
                                        reason=None,
                                    )
                                    if ev3m_score < cutoff:
                                        logger.info(
                                            f"M2: {token.symbol} REJECT — EV3m score "
                                            f"{ev3m_score:.3f} < {band} cutoff {cutoff:.3f} "
                                            f"(src={ev3m_src} age={age:.1f}s)")
                                        self.db.record_discovery(
                                            token, ev3m_score, False,
                                            f"ev3m_live_{band}={ev3m_score:.3f}<{cutoff:.3f}",
                                            features=candidate.features)
                                        self._add_to_observation_pool(
                                            token, ev3m_score, False,
                                            f"ev3m_live_below_{band}",
                                            features=candidate.features)
                                        candidate = None
                                    else:
                                        logger.info(
                                            f"M2: {token.symbol} PASS — EV3m score "
                                            f"{ev3m_score:.3f} >= {band} cutoff {cutoff:.3f} "
                                            f"(src={ev3m_src} age={age:.1f}s)")
                                        candidate.model_score = ev3m_score
                                        if candidate.features is None:
                                            candidate.features = {}
                                        candidate.features["ev3m_live_score"] = ev3m_score
                                        candidate.features["ev3m_live_cutoff"] = cutoff
                                        candidate.features["ev3m_live_feature_source"] = ev3m_src
                                else:
                                    self._cache_ev3m_live_audit(
                                        token,
                                        status="insufficient_features",
                                        feature_source=ev3m_src,
                                        age_sec=age,
                                        score=None,
                                        band=str(self.config.ev3m_live_selection_band or "top10").strip() or "top10",
                                        cutoff=self.ev3m_live_model.selection_cutoffs.get(
                                            str(self.config.ev3m_live_selection_band or "top10").strip() or "top10",
                                            self.ev3m_live_model.selection_cutoffs.get("top10", 0.67),
                                        ),
                                        passed=False,
                                        reason=ev3m_src,
                                    )
                                    logger.debug(
                                        f"M2: {token.symbol} EV3m features unavailable "
                                        f"(reason={ev3m_src}, age={age:.1f}s), rejecting live EV3m gate"
                                    )
                                    candidate = None
                            except Exception as e:
                                self._cache_ev3m_live_audit(
                                    token,
                                    status="error",
                                    feature_source="live_gate_exception",
                                    age_sec=age,
                                    score=None,
                                    band=str(self.config.ev3m_live_selection_band or "top10").strip() or "top10",
                                    cutoff=self.ev3m_live_model.selection_cutoffs.get("top10", 0.67),
                                    passed=False,
                                    reason=str(e),
                                )
                                logger.warning(f"M2: {token.symbol} EV3m live gate error: {e} — rejecting")
                                candidate = None

                        # ── V-shape live gate: Phase 8 main contract ──
                        if (candidate is not None
                                and self.shadow_vshape_model is not None
                                and self.config.vshape_live_enabled):
                            try:
                                # Phase 15a (2026-04-26): n_bars sourced from model's
                                # entry_delay_sec to match training byte-parity:
                                #   v1.6 (300s entry):  n_bars = max(5+2, 7) = 7
                                #     (training uses build_1min_bars_from_swaps n_bars=7)
                                #   v1.4r (600s entry): n_bars = max(10+2, 7) = 12
                                #     (training also uses 12)
                                # Old code used hardcoded n_bars=12 which over-built
                                # 5 placeholder bars for v1.6 (token only ~5min old at
                                # M2 fire time). Shadow v-shape (line 2635) already
                                # uses this formula — now M2 matches.
                                entry_offset = int(self.shadow_vshape_model.entry_delay_sec)
                                window_min = max(1, entry_offset // 60)
                                n_bars_v = max(window_min + 2, 7)
                                kline_bars, kline_shifted = self.kline_builder.build_kline(
                                    mint=token.mint_address,
                                    start_ts=int(token.graduation_time),
                                    n_bars=n_bars_v, resolution=60,
                                    sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
                                )
                                vshape_passed = False
                                vshape_score = 0.0
                                pattern = "none"
                                if kline_bars and len(kline_bars) >= 3:
                                    vf = detect_vshape_live(
                                        kline_bars, token.graduation_time,
                                        entry_offset_sec=entry_offset)
                                    if vf and vf.get("any_pattern", 0) == 1:
                                        # Phase 15d 2026-04-27: re-enabled reversal pattern.
                                        # Was hardcoded REJECT since v1.4r (2026-04-24)
                                        # citing 28-trade PnL −$15.42 — but that was BEFORE
                                        # v5.2 profit-protect + rug_event_exit deployed.
                                        # Reversal had highest historical win rate (55.6%
                                        # vs vshape 35.7%) so re-enabling lets the model
                                        # score gate it like other patterns and provides
                                        # additional cohort to validate the new exit stack.
                                        swaps_raw = self.db.get_swaps_for_token(token.mint_address)
                                        micro = compute_micro_live(
                                            swaps_raw, token.graduation_time,
                                            window_sec=entry_offset,
                                            feat_prefix=f"m{window_min}")
                                        sm = self.db.compute_smart_money_features(token.mint_address)
                                        features_vshape = {}
                                        features_vshape.update(vf)
                                        features_vshape.update(micro)
                                        features_vshape.update(sm)
                                        vshape_score = self.shadow_vshape_model.predict_score(features_vshape)
                                        band = str(self.config.vshape_live_selection_band or "top10").strip()
                                        cutoff = self.shadow_vshape_model.selection_cutoffs.get(band, 0.5)
                                        vshape_passed = vshape_score >= cutoff
                                        pattern = (
                                            "vshape" if vf.get("is_vshape") else
                                            "steady_up" if vf.get("is_steady_up") else
                                            "near_high" if vf.get("is_near_high") else
                                            "reversal" if vf.get("is_reversal") else
                                            "unknown")
                                        logger.info(
                                            f"M2-VSHAPE: {token.symbol} pattern={pattern} "
                                            f"score={vshape_score:.3f} {'PASS' if vshape_passed else 'REJECT'} "
                                            f"({band}>={cutoff:.3f})")
                                    else:
                                        logger.info(f"M2-VSHAPE: {token.symbol} no pattern detected, REJECT")
                                else:
                                    logger.info(f"M2-VSHAPE: {token.symbol} insufficient bars, REJECT")

                                if not vshape_passed:
                                    self.db.record_discovery(
                                        token, vshape_score, False,
                                        f"vshape_reject_score={vshape_score:.3f}",
                                        features=candidate.features)
                                    self._add_to_observation_pool(
                                        token, vshape_score, False, "vshape_below_cutoff",
                                        features=candidate.features)
                                    candidate = None
                                else:
                                    candidate.model_score = vshape_score
                                    # Phase 25k: tag entry_source for legacy v1.6.1 path
                                    candidate.entry_source = "vshape_v1_6_1"
                                    if candidate.features is None:
                                        candidate.features = {}
                                    candidate.features["vshape_score"] = vshape_score
                                    candidate.features["vshape_pattern"] = pattern
                            except Exception as e:
                                # Fail-closed: error during gate eval rejects
                                # the token. Logged at ERROR with stack trace
                                # so latent bugs (swap data, model IO, feature
                                # dict) surface instead of silently rejecting
                                # every candidate.
                                logger.error(
                                    f"M2-VSHAPE: {token.symbol} gate error: {e}",
                                    exc_info=True)
                                candidate = None

                        # RugFilter v3 entry-gate REMOVED 2026-05-06.
                        # v4 shadow-only at this stage; will become entry gate
                        # once 7-14 day shadow validates precision/recall on live data.

                    if candidate:
                        # Phase 25g (2026-05-03) — fail-closed safety net.
                        # If no entry-model gate set entry_source, REJECT.
                        # Prevents the post-codex regression where v3.4 +
                        # big_winner gates silently skip and candidates flow
                        # through with empty entry_source (FISHER -$28.85).
                        if (self.config.require_entry_source
                                and not str(getattr(candidate, "entry_source", "") or "").strip()):
                            logger.warning(
                                f"M2: {token.symbol} REJECT — no entry_source "
                                f"set after gate sequence (require_entry_source=true). "
                                f"model_score={candidate.model_score:.3f}")
                            self.db.record_discovery(
                                token, candidate.model_score, False,
                                "no_entry_source_fail_closed",
                                features=candidate.features)
                            self._add_to_observation_pool(
                                token, candidate.model_score, False,
                                "no_entry_source_fail_closed",
                                features=candidate.features)
                            candidate = None
                        else:
                            candidates.append(candidate)
                            self._candidates_seen += 1
                else:
                    # defer to next tick
                    still_pending.append(token)
            else:
                # not yet ready — keep waiting for feature delay
                still_pending.append(token)

        self._pending = still_pending
        self.processed_data["candidates"] = candidates

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """Not using PositionExecutor — return empty list."""
        return []

    async def control_task(self):
        """Override control_task to run all 5 modules in a single loop.

        Default control_task gates on market_data_provider.ready and executors_update_event.
        We bypass both since we don't use hummingbot connectors or executors.
        """
        self._tick_count += 1

        # Periodic heartbeat log (every 30 ticks ≈ 30s at 1s tick)
        if self._tick_count % 30 == 1:
            # Cleanup expired blacklist entries
            now_bl = time.time()
            self._stoploss_blacklist = {m: t for m, t in self._stoploss_blacklist.items() if t > now_bl}

            # Force RiskManager date reset check (bug fix 2026-05-04):
            # _check_daily_reset() only fires inside can_trade()/is_cooldown_only().
            # Without a periodic call here, halt persists past UTC midnight when
            # no signal triggers can_trade() (observed 2026-05-03 halt @ 18:28
            # remained active for 6+ hours after midnight rollover).
            self.risk._check_daily_reset()

            # L6 killswitch heartbeat check (W2.2 F2-M3b). Single placement
            # site — heartbeat tick, NOT per-trade — to avoid evaluation cost
            # on every buy candidate. Edge-only firing + 60s alert throttle
            # documented inside _killswitch_check.
            try:
                self._killswitch_check()
            except Exception as exc:  # noqa: BLE001 — killswitch eval must never crash heartbeat
                logger.exception(f"killswitch eval failed: {exc}")

            # Sweep expired swap collectors. Tokens that have hit
            # swap_collection_window_sec deadline AND aren't in keepalive set
            # (real trade in flight) get final-flushed and unregistered.
            self._sweep_expired_swap_collectors(now_bl)

            # Phase 25l (2026-05-04): periodic checkpoint flush of ALL active
            # swap collectors (without unregistering). Birdeye-vs-Chainstack
            # alignment audit found Chainstack data ends ~15min before
            # Birdeye on every mint — caused by bot restarts losing
            # in-memory swap buffers. on_stop's flush only runs on graceful
            # SIGTERM with enough grace period; docker restart often SIGKILLs
            # before flush completes. db.save_swaps is idempotent (tx_hash +
            # tuple dedup), so periodic flush is safe.
            #
            # Q2 fix (2026-05-06): time-based instead of tick-based.
            # Production tick rate is ~1 tick / 3.6s (heartbeat is not 1Hz),
            # so the old `tick_count % 180 == 1` flushed every 10.8 min, not
            # the 3 min the comment claimed. D2 lag audit (V5_6_PAIN_POINT_REPORT
            # appendix) showed v4 read DB up to ~10 min stale → 80% of swaps
            # missing at score time. Time-based 30s cadence keeps DB
            # synchronized within ~30s of memory state for all downstream
            # readers (v4 shadow now uses Q1 forced-flush; this protects
            # any future model that reads from DB without forcing first).
            _now_chk = time.time()
            if _now_chk - self._last_checkpoint_flush_ts >= 30:
                self._last_checkpoint_flush_ts = _now_chk
                _checkpoint_n = 0
                for _mint in list(self._swap_collection_until.keys()):
                    try:
                        self._flush_swaps_to_db(_mint, unregister=False)
                        _checkpoint_n += 1
                    except Exception as _e:
                        logger.debug(
                            f"checkpoint-flush: {_mint[:12]}... err: {_e}")
                # Also checkpoint keepalive (open-position) collectors.
                for _mint in list(self._swap_keepalive_mints):
                    try:
                        self._flush_swaps_to_db(_mint, unregister=False)
                        _checkpoint_n += 1
                    except Exception as _e:
                        logger.debug(
                            f"checkpoint-flush keepalive: {_mint[:12]}... err: {_e}")
                if _checkpoint_n:
                    logger.info(
                        f"checkpoint-flush: {_checkpoint_n} collector(s) "
                        f"persisted (no unregister)")

            logger.info(f"[tick={self._tick_count}] positions={len(self._positions)}, "
                        f"pending={len(self._pending)}, queue={len(self._candidate_queue)}, "
                        f"observing={len(self._observation_pool)}, "
                        f"swap_collecting={len(self._swap_collection_until)}, "
                        f"blacklist={len(self._stoploss_blacklist)}, "
                        f"trades={self.risk.total_trades}, "
                        f"pnl=${sum(t.pnl_usd for t in self._trade_log):.2f}, "
                        f"halted={self.risk.halted}")

            # Phase B.2 divergence summary (own cadence, default 5min)
            now_mono = time.monotonic()
            if (self.config.cs_divergence_shadow_enabled
                    and (now_mono - self._cs_div_last_summary_t)
                        >= self.config.cs_divergence_summary_interval_sec):
                self._emit_cs_divergence_summary()
                # Phase B.4 piggy-backs on same 5min cadence
                self._emit_b4_inference_summary()
                self._cs_div_last_summary_t = now_mono

        # Phase B.2 v4 (2026-05-19): observing-mint shadow path — collects
        # divergence data even when bot has 0 open positions. Solves the
        # data-scarcity issue blocking B.2 promote (24h with 0 positions →
        # 0 rows from the position-side path).
        try:
            await self._run_b2_shadow_for_observing_mints()
        except Exception as _e:
            logger.warning("Phase B.2 v4 observing shadow tick failed: %s", _e)

        # M1 + M2: discover and evaluate
        await self.update_processed_data()

        # Flush expired observations (GMGN kline fetch) + intermediate snapshots
        await self._flush_expired_observations()
        await self._collect_observation_snapshots()
        await shadow_orchestrator.run_all(self)

        # Update SOL price for USD conversion (every 60s)
        await self._refresh_sol_price()

        # M3: execute buys — merge new candidates into queue, prioritize by model_score
        now = time.time()

        # Add new candidates to queue (stamp queued_at)
        for candidate in self.processed_data.get("candidates", []):
            candidate.queued_at = now
            self._candidate_queue.append(candidate)
            logger.info(f"M3: [QUEUED] {candidate.token.symbol} model_score={candidate.model_score:.3f} "
                        f"— added to candidate queue (size={len(self._candidate_queue)})")

        # Expire stale candidates (older than candidate_queue_ttl_sec)
        # 2026-07-06 修复②:过期具名化(原匿名计数 → 逐 symbol 可查)+ xoscross
        # 候选终判归因落库。
        ttl = self.config.candidate_queue_ttl_sec
        _fresh, _stale = [], []
        for c in self._candidate_queue:
            (_fresh if now - c.queued_at < ttl else _stale).append(c)
        self._candidate_queue = _fresh
        if _stale:
            logger.info(
                f"M3: expired {len(_stale)} stale candidates from queue "
                f"(ttl={ttl:.0f}s): "
                f"{', '.join(c.token.symbol for c in _stale[:8])}")
            for c in _stale:
                if getattr(c, "entry_source", "") == "xoscross":
                    _xoscross.mark_live_action(self, c.token.mint_address,
                                               "queue_expired")

        self._candidate_queue.sort(key=lambda c: c.model_score, reverse=True)

        # Try to buy from queue
        bought = []
        max_buy_retries = 5
        for candidate in self._candidate_queue:
            # Phase 16.3 Step 3 (audit fix) — recompute the open-mints set
            # per-iteration. A successful buy earlier in this loop mutates
            # `self._positions`; a stale snapshot would let a duplicate slip
            # through if `break` after success were ever removed.
            open_mints = {p.token.mint_address for p in self._positions}
            if candidate.token.mint_address in open_mints:
                bought.append(candidate)
                logger.info(
                    f"M3: {candidate.token.symbol} skipped — already has open "
                    f"position (entry_source={getattr(candidate,'entry_source','')})")
                if getattr(candidate, "entry_source", "") == "xoscross":
                    _xoscross.mark_live_action(
                        self, candidate.token.mint_address, "dup_open_position")
                continue
            if candidate.buy_fail_count >= max_buy_retries:
                bought.append(candidate)
                logger.info(f"M3: {candidate.token.symbol} evicted — {candidate.buy_fail_count} "
                            f"consecutive buy failures")
                # Token will never enter trade, free the swap collector
                self._cleanup_swap_keepalive(
                    candidate.token.mint_address, reason="buy retries exhausted")
                if getattr(candidate, "entry_source", "") == "xoscross":
                    _xoscross.mark_live_action(
                        self, candidate.token.mint_address,
                        f"evicted:max_retries:{candidate.buy_last_fail_reason}")
                continue
            # 2026-05-14: class-aware retry backoff. Original blanket
            # `10*2^N` (20s/40s/80s/120s) wasted entry-timing alpha when the
            # failure was a transient AMM jitter (Pump.fun bonding curve +
            # Jupiter conservative estimator → price_impact estimate flips
            # within 1-2 ticks even though pool state is essentially flat).
            #   soft     = 0s (next-tick retry); covers price_impact / chase
            #   hard     = exp backoff (upstream API down); covers snapshot /
            #              quote_error / not_wsol — exp protects the dead API
            #   terminal = evict immediately (pool too small, etc.)
            fail_class = self._classify_preflight_failure(
                candidate.buy_last_fail_reason)
            if fail_class == "terminal" and candidate.buy_fail_count > 0:
                bought.append(candidate)
                logger.info(
                    f"M3: {candidate.token.symbol} evicted — TERMINAL "
                    f"failure (reason={candidate.buy_last_fail_reason})")
                self._cleanup_swap_keepalive(
                    candidate.token.mint_address,
                    reason=f"terminal preflight fail: {candidate.buy_last_fail_reason}")
                if getattr(candidate, "entry_source", "") == "xoscross":
                    _xoscross.mark_live_action(
                        self, candidate.token.mint_address,
                        f"evicted:{candidate.buy_last_fail_reason}")
                continue
            if fail_class == "soft":
                backoff = 0.0          # 1-tick retry
            elif candidate.buy_fail_count > 0:
                backoff = min(10.0 * (2 ** candidate.buy_fail_count), 120.0)
            else:
                backoff = 0.0
            if candidate.buy_last_fail_time > 0 and now - candidate.buy_last_fail_time < backoff:
                continue
            if self.risk.can_trade(len(self._positions)):
                success = await self._execute_buy(candidate)
                if success:
                    bought.append(candidate)
                    if getattr(candidate, "entry_source", "") == "xoscross":
                        _xoscross.mark_live_action(
                            self, candidate.token.mint_address, "entered")
                    break
                else:
                    candidate.buy_fail_count += 1
                    candidate.buy_last_fail_time = now
                    # Re-classify with the *new* reason set by _execute_buy
                    new_class = self._classify_preflight_failure(
                        candidate.buy_last_fail_reason)
                    next_backoff = (0.0 if new_class == "soft"
                                    else min(10.0 * (2 ** candidate.buy_fail_count),
                                             120.0))
                    logger.info(
                        f"M3: {candidate.token.symbol} buy failed (attempt "
                        f"#{candidate.buy_fail_count}, class={new_class}, "
                        f"reason={candidate.buy_last_fail_reason}), "
                        f"retry in {next_backoff:.0f}s")
                    break
            elif self.risk.is_cooldown_only(len(self._positions)):
                # Cooldown blocking — keep in queue, will retry next tick
                secs_left = self.config.cooldown_sec - (now - self.risk._last_trade_time)
                logger.debug(f"M3: {candidate.token.symbol} queued, cooldown {secs_left:.0f}s remaining "
                             f"(queue={len(self._candidate_queue)})")
                break  # no point trying others, they'll all be blocked by cooldown
            else:
                # Hard block (halted, max_positions, etc.) — discard all
                logger.debug(f"M3: discarding queue — hard block: halted={self.risk.halted}, "
                             f"positions={len(self._positions)}/{self.risk.max_positions}")
                # Free swap collectors for every dropped candidate
                for c in self._candidate_queue:
                    self._cleanup_swap_keepalive(
                        c.token.mint_address, reason="queue cleared (hard block)")
                    if getattr(c, "entry_source", "") == "xoscross":
                        _xoscross.mark_live_action(
                            self, c.token.mint_address,
                            f"discarded:hard_block:halted={self.risk.halted}"
                            f":pos={len(self._positions)}")
                self._candidate_queue.clear()
                break

        # Remove bought candidates from queue
        for c in bought:
            self._candidate_queue.remove(c)

        # M4 + M5: monitor positions and check exits
        await self._monitor_positions()

    # ── T1.4 config validator boot-gate ─────────────────────────────────────
    # Spec §3.B.3 / §8.2 / §8.4: validate_config() is pure-function, fast,
    # no DB/network.  Critical findings (R14: max_buy_price_impact_pct < 0.20)
    # BLOCK startup via RuntimeError.  Warnings (R15 cs_b4_promoted, R16
    # cross-field, R1 missing keys) are logged + alerted but NEVER block.
    # Defensive outer try/except ensures an unexpected validator-internal crash
    # (not a deliberate critical finding) lets boot continue.
    # Extracted to a named helper so tests can exercise it directly without
    # constructing the full asyncio controller stack.
    def _run_config_boot_gate(self) -> None:
        """Run validate_config() on self.config and enforce the boot policy.

        - warnings  → logger.warning + record_event + Telegram (best-effort)
        - criticals → logger.error + record_event + Telegram (best-effort),
                      then raise RuntimeError to block startup.
        - validator crash → logger.warning, boot continues (validator-internal
                      crash must never cause a self-inflicted outage).

        The RuntimeError for criticals propagates out of this method and is
        NOT caught by the outer defensive try/except — only genuine validator
        crashes are swallowed.
        """
        # Step 1: convert Pydantic v2 config to plain dict.
        cfg_dict: dict
        try:
            cfg_dict = self.config.model_dump()
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"T1.4 config boot-gate: model_dump() failed ({exc}); "
                "skipping config validation."
            )
            return

        # Step 2: call validator (defensive wrapper for unexpected crashes).
        issues = None
        try:
            issues = _validate_config(cfg_dict)
        except Exception as exc:  # pragma: no cover — validator-internal crash
            logger.warning(
                f"T1.4 config boot-gate: validate_config() raised unexpectedly "
                f"({exc}); skipping config validation to avoid self-outage."
            )
            return

        # Step 3: partition and handle by severity.
        warnings_found = [i for i in issues if i.severity == "warning"]
        criticals_found = [i for i in issues if i.severity == "critical"]

        for issue in warnings_found:
            logger.warning(f"CONFIG WARN: {issue.code}: {issue.message}")
            try:
                self.db.record_event(
                    "WARN", "config",
                    f"T1.4 boot-gate warning: {issue.code}: {issue.message[:200]}"
                )
            except Exception:  # pragma: no cover
                pass
            try:
                self.alerts.send(
                    "config_warn",
                    extra_text=f"{issue.code}: {issue.message[:200]}"
                )
            except Exception:  # pragma: no cover — alert failure must not block boot
                pass

        for issue in criticals_found:
            logger.error(f"CONFIG REJECT: {issue.code}: {issue.message}")
            try:
                self.db.record_event(
                    "ERROR", "config",
                    f"T1.4 boot-gate CRITICAL: {issue.code}: {issue.message[:200]}"
                )
            except Exception:  # pragma: no cover
                pass
            try:
                self.alerts.send(
                    "config_critical",
                    extra_text=f"BOOT BLOCKED — {issue.code}: {issue.message[:200]}"
                )
            except Exception:  # pragma: no cover — alert failure must not block boot
                pass

        # Step 4: hard-fail on any critical.  This raise is OUTSIDE the
        # defensive try/except above so it propagates as intended.
        if criticals_found:
            n = len(criticals_found)
            codes = ", ".join(i.code for i in criticals_found)
            raise RuntimeError(
                f"Config validation failed: {n} critical issue(s) [{codes}] — "
                "see CONFIG REJECT log lines above. Fix the config before restarting."
            )

    async def on_start(self):
        """Called once when controller starts."""
        # ── T1.4: config validator boot-gate (EARLY — pure in-memory, no DB/net) ──
        # validate_config() checks R14/R15/R16/R1.  Critical findings block startup;
        # warnings are logged+alerted but never block.  Validator crashes are swallowed
        # (never self-outage).  Must run before any network call so a bad config is
        # caught before we touch Gateway or DB.
        self._run_config_boot_gate()

        # ── L1 wallet guard (Phase 15a deployment audit F2, 2026-04-26) ──
        # Verify Gateway has private keys for the wallet_address in config.
        # Prior incident (2026-04-09): yml drift caused 4 silent buy-500 trades
        # because gateway signed with a different wallet than config expected.
        try:
            loaded = await self.trader.get_loaded_wallets(chain="solana")
            cfg_addr = self.config.wallet_address.lower()
            if loaded and cfg_addr not in loaded:
                raise RuntimeError(
                    f"L1 wallet mismatch: config.wallet_address={self.config.wallet_address} "
                    f"NOT in gateway-loaded wallets={loaded}. Reconcile "
                    f"gateway/conf/wallets/solana/<addr>.json with conf yml."
                )
            if not loaded:
                logger.warning(
                    "L1 wallet check: gateway returned no loaded wallets — "
                    "endpoint may be unsupported. Proceeding without verification.")
            else:
                logger.info(f"L1 wallet check: {self.config.wallet_address} confirmed loaded by gateway")
        except RuntimeError:
            raise  # propagate explicit mismatch
        except Exception as e:
            logger.warning(f"L1 wallet check skipped (non-fatal): {e}")

        logger.info("=" * 60)
        logger.info("MemeSniper controller STARTING")
        logger.info(f"  gateway_url    = {self.config.gateway_url}")
        logger.info(f"  wallet         = {self.config.wallet_address}")
        logger.info(f"  connector      = {self.config.connector}")
        logger.info(f"  model_path     = {self.config.model_path}")
        logger.info(f"  position_size  = ${self.config.position_size_usd}")
        logger.info(f"  stop_loss      = {self.config.stop_loss_pct*100:.0f}%")
        logger.info(f"  time_limit     = {self.config.time_limit_sec}s")
        logger.info(f"  max_buy_impact = {self.config.max_buy_price_impact_pct:.1%}")
        logger.info(f"  max_buy_prem   = {self.config.max_buy_vs_mid_premium_pct:.1%}")
        logger.info(f"  max_entry_chase= {self.config.max_entry_chase_pct:.1%}")
        logger.info(f"  jupiter_wait   = {self.config.buy_jupiter_wait_sec:.2f}s "
                    f"({'enabled' if self.config.buy_jupiter_wait_sec > 0 else 'DISABLED — instant'})")
        _gp1 = ("LIVE" if self.config.grpc_event_exit_enabled
                  and not self.config.grpc_event_exit_shadow_only
                  else "SHADOW" if self.config.grpc_event_exit_enabled
                  else "OFF")
        logger.info(
            f"  grpc_exit_p1   = {_gp1} "
            f"(SL={'on' if self.config.grpc_event_exit_sl_enabled else 'off'}, "
            f"EC={'on' if self.config.grpc_event_exit_ec_enabled else 'off'}, "
            f"min_delta={self.config.grpc_event_min_price_delta:.2%})")
        logger.info(f"  live_snapshot  = {self.config.require_live_market_snapshot}")
        logger.info(f"  route_biggest  = {self.config.require_route_uses_biggest_pool}")
        logger.info(f"  trailing_act   = {self.config.trailing_activation_pct*100:.0f}%")
        logger.info(f"  trailing_drop  = {self.config.trailing_drop_pct*100:.0f}%")
        logger.info(f"  max_positions  = {self.config.max_positions}")
        logger.info(f"  queue_ttl      = {self.config.candidate_queue_ttl_sec}s")
        logger.info(f"  max_trades     = {self.config.max_total_trades}")
        logger.info(f"  daily_loss_lim = ${self.config.daily_loss_limit_usd}")
        # Phase 22.G.2 sizing config dump
        _sm = str(getattr(self.config, "sizing_mode", "fixed"))
        if _sm == "slippage_budget":
            logger.info(
                f"  sizing_mode    = {_sm} "
                f"(β={self.config.sizing_slippage_budget:.4f} "
                f"min=${self.config.sizing_min_usd:.0f} "
                f"max=${self.config.sizing_max_usd:.0f} "
                f"src_cap={'on' if self.config.sizing_respect_source_override_as_cap else 'off'})")
            # Cross-check daily_loss_limit. If loss limit < 5× max, warn —
            # one or two SLs would halt the bot.
            _ll = float(self.config.daily_loss_limit_usd)
            _max = float(self.config.sizing_max_usd)
            if _ll < 5 * _max:
                logger.warning(
                    f"  ⚠️ sizing_max_usd=${_max:.0f} but daily_loss_limit=${_ll:.0f}. "
                    f"At max sizing 1-2 SLs trigger halt. Recommend "
                    f"daily_loss_limit ≥ ${5*_max:.0f}.")
            # Sanity-check β
            _b = float(self.config.sizing_slippage_budget)
            if _b > 0.01:
                logger.warning(
                    f"  ⚠️ sizing_slippage_budget={_b:.4f} > 0.01 (1% one-way). "
                    f"Round-trip friction will be ≥2%. Phase 22.G THIN edge "
                    f"is +4.13% — leaves <2pp margin.")
            if self.config.sizing_min_usd > self.config.sizing_max_usd:
                logger.error(
                    f"  ❌ sizing_min_usd=${self.config.sizing_min_usd:.0f} > "
                    f"sizing_max_usd=${self.config.sizing_max_usd:.0f} — "
                    f"compute will clamp at runtime, but config is invalid.")
        else:
            logger.info(
                f"  sizing_mode    = {_sm} (legacy: position_size_usd=${self.config.position_size_usd})")
        logger.info(f"  sl_blacklist   = {self.config.token_stoploss_cooldown_sec:.0f}s")
        logger.info(f"  max_5min_ret   = {self.config.max_5min_return:.0%}")
        logger.info(f"  max_vol_liq    = {self.config.max_vol_liq_ratio:.1f}")
        logger.info(f"  min_liquidity  = ${self.config.min_liquidity_usd:,.0f}")
        logger.info(f"  feature_delay  = {self.config.feature_delay_sec}s")
        logger.info(f"  swap_poll_int  = {self.config.swap_poll_interval}s")
        logger.info(f"  shadow_rule    = {'enabled' if self.config.shadow_rule_first_enabled else 'disabled'}")
        if self.config.shadow_rule_first_enabled:
            logger.info(f"  shadow_stage2  = {self.config.shadow_rule_stage2_name}")
            logger.info(f"  shadow_dd_min  = {self.config.shadow_rule_drawdown_1to3m_min:+.6f}")
            logger.info(f"  shadow_rng_max = {self.config.shadow_rule_range_1to3m_max:.6f} (legacy fallback)")
        logger.info(f"  ev3m_live      = {'enabled' if self.config.ev3m_live_enabled else 'disabled'}")
        if self.config.ev3m_live_enabled:
            logger.info(f"  ev3m_live_model = {self.config.ev3m_live_model_path}")
            logger.info(f"  ev3m_live_delay = {self.config.ev3m_live_entry_delay_sec}s")
            logger.info(f"  ev3m_live_band  = {self.config.ev3m_live_selection_band}")
            logger.info(f"  ev3m_live_no_trailing = {self.config.ev3m_live_disable_trailing}")
            logger.info(f"  ev3m_live_loaded = {self.ev3m_live_model is not None}")
            if int(self.config.feature_delay_sec) != int(self.config.ev3m_live_entry_delay_sec):
                logger.warning(
                    f"  ev3m_live_delay mismatch: feature_delay={self.config.feature_delay_sec}s "
                    f"vs ev3m_live_delay={self.config.ev3m_live_entry_delay_sec}s"
                )
        logger.info(f"  shadow_ev3m    = {'enabled' if self.config.shadow_ev3m_enabled else 'disabled'}")
        if self.config.shadow_ev3m_enabled:
            logger.info(f"  shadow_ev3m_model = {self.config.shadow_ev3m_model_path}")
            logger.info(f"  shadow_ev3m_delay = {self.config.shadow_ev3m_entry_delay_sec}s")
            logger.info(f"  shadow_ev3m_bands = {self.config.shadow_ev3m_selection_bands}")
            logger.info(f"  shadow_ev3m_loaded = {self.shadow_ev3m_model is not None}")
        logger.info(
            f"  sw_event_shadow = {'enabled' if self.config.shadow_super_winner_event_enabled else 'disabled'}"
        )
        if self.config.shadow_super_winner_event_enabled:
            logger.info(f"  sw_event_model = {self.config.shadow_super_winner_event_model_path}")
            logger.info(
                f"  sw_event_scan  = {self.config.shadow_super_winner_event_scan_start_sec}s"
                f"->{self.config.shadow_super_winner_event_scan_end_sec}s "
                f"step={self.config.shadow_super_winner_event_scan_step_sec}s"
            )
            logger.info(
                f"  sw_event_bands = {self.config.shadow_super_winner_event_default_band},"
                f"{self.config.shadow_super_winner_event_fallback_band}"
            )
            logger.info(
                f"  sw_event_pols  = {self.config.shadow_super_winner_event_default_policy},"
                f"{self.config.shadow_super_winner_event_fallback_policy}"
            )
            logger.info(
                f"  sw_event_rule  = {self.config.shadow_super_winner_event_arbitration_rule}"
            )
            logger.info(
                f"  sw_event_loaded = {self.shadow_super_winner_event_model is not None}"
            )
        logger.info(f"  latency_events = {'enabled' if self.config.latency_events_enabled else 'disabled'}")
        _xos_mode = ("LIVE" if not self.config.xoscross_shadow_only else "shadow") \
            if self.config.xoscross_enabled else "disabled"
        logger.info(f"  xoscross       = {_xos_mode} (x{self.config.xoscross_bottomk_top_frac:.2f}, "
                    f"simple_t5m={'on' if self.config.simple_t5m_enabled else 'off'})")
        if (self.config.xoscross_enabled and not self.config.xoscross_shadow_only
                and not self.config.pregrad_factors_shadow_enabled):
            logger.error("xoscross LIVE 但 pregrad_factors_shadow_enabled=false —— "
                         "L1 verdict 无来源,fail-closed 会拒掉全部真金入场(零单)!"
                         "请打开 pregrad_factors_shadow_enabled 或回退 shadow_only。")
        logger.info(f"  kline_builder  = {'enabled' if self.kline_builder else 'disabled'}")
        logger.info(f"  grpc_stream    = {'enabled' if self._grpc_stream else 'disabled'}")
        logger.info(f"  model loaded   = {self.signal is not None}")
        logger.info(f"  gmgn_api_key   = {'set' if self.discovery.api_key else 'MISSING'}")
        logger.info("=" * 60)
        self.db.record_event("INFO", "on_start", "MemeSniper controller started")
        await self._refresh_sol_price()
        logger.info(f"SOL price: ${self._sol_price_usd:.2f}")

        # Rebuild risk state from DB (daily PnL, consecutive losses, total trades)
        self.risk.rebuild_from_db(self.db)

        # Rebuild trade log from DB so heartbeat/PnL display survives restarts
        self._trade_log = self.db.load_all_trades()
        if self._trade_log:
            total_pnl = sum(t.pnl_usd for t in self._trade_log)
            logger.info(f"Rebuilt trade log from DB: {len(self._trade_log)} trades, "
                        f"cumulative PnL=${total_pnl:+.2f}")

        # Recover stoploss blacklist from DB
        self._stoploss_blacklist = self.db.get_recent_stoploss_mints(
            self.config.token_stoploss_cooldown_sec)
        if self._stoploss_blacklist:
            logger.info(f"Recovered {len(self._stoploss_blacklist)} stoploss blacklist entries from DB")

        # Recover open positions from DB
        recovered = self.db.load_open_positions()

        # BUG 3 FIX (2026-04-30) — phantom position janitor.
        # Sweep open_positions rows whose wallet token balance is empty: these
        # are phantom records left behind when a sell succeeded on-chain but
        # bot didn't update DB (e.g., Gateway 400 false-error race, fomo case).
        if recovered:
            phantom_mints = []
            for pos in recovered:
                try:
                    bal = await self._get_wallet_token_balance_ui(pos.token.mint_address)
                    if bal is not None and pos.token_amount > 0:
                        if bal < pos.token_amount * 0.01:  # <1% remaining = phantom
                            phantom_mints.append((pos, bal))
                except Exception as e:
                    logger.debug(f"janitor check failed for {pos.token.symbol}: {e}")
            for pos, bal in phantom_mints:
                logger.warning(
                    f"[STARTUP JANITOR] Pruning phantom open_position "
                    f"{pos.token.symbol} ({pos.token.mint_address[:12]}...) — "
                    f"wallet balance {bal:.4f} << expected {pos.token_amount:.0f}")
                self.db.remove_position(pos.token.mint_address)
                self.db.record_event(
                    "WARNING", "startup",
                    f"JANITOR pruned phantom {pos.token.symbol} "
                    f"wallet={bal:.4f} expected={pos.token_amount:.0f}")
            if phantom_mints:
                # Reload after pruning
                recovered = self.db.load_open_positions()

        if recovered:
            self._positions = recovered
            # Add recovered mints to seen set so M1 doesn't re-discover them
            for pos in recovered:
                self.discovery._seen_mints.setdefault(pos.token.mint_address)
            logger.info(f"Recovered {len(recovered)} open positions from DB:")
            for pos in recovered:
                hold = pos.hold_seconds()
                logger.info(f"  {pos.token.symbol} ({pos.token.mint_address[:8]}...) "
                            f"— {pos.token_amount:.0f} tokens, invested={pos.sol_invested:.4f} SOL, "
                            f"held={hold:.0f}s")
                # Re-register the swap collector for the recovered position so
                # we keep collecting hold-period swap data through the rest of
                # this trade. Without this, restarted bots would not record any
                # swap data for in-flight positions — the same bug as before
                # the fix, just triggered by a different code path.
                if self.kline_builder and pos.token.pool_address:
                    self.kline_builder.register(pos.token.mint_address, pos.token.pool_address)
                    self._swap_keepalive_mints.add(pos.token.mint_address)
                    logger.info(f"  KlineBuilder: re-registered {pos.token.symbol} "
                                f"for swap collection (recovered position)")
            self.db.record_event("INFO", "on_start",
                                 f"Recovered {len(recovered)} positions: "
                                 + ", ".join(p.token.symbol for p in recovered))

        # 2026-05-19 v0.7.4 — Orphan scanner: rebuild Positions for tokens that
        # the wallet holds but bot never recorded (Gateway status=0 race orphans).
        # Runs AFTER load_open_positions so existing positions aren't duplicated;
        # falls back to the latency_events buy_submit metadata for sizing/symbol.
        try:
            await self._scan_wallet_for_orphans()
        except Exception as e:
            logger.warning(f"[ORPHAN SCANNER] on_start hook raised: {e}", exc_info=True)

        # Phase B.1: launch cleaning_service bridge diagnostic logger task
        # (sync ingest path was already wired in __init__ via GeyserPumpSwapStream).
        if self._cleaning_service_bridge is not None:
            try:
                await self._cleaning_service_bridge.start()
            except Exception as e:
                logger.warning(
                    "Phase B.1: cleaning_service bridge.start() failed (%s) "
                    "— ingest will continue but no periodic diagnostics",
                    e,
                )

        # Start Yellowstone gRPC stream (if configured)
        if self._grpc_stream:
            self._grpc_stream.start()
            logger.info("M1/M2: Yellowstone gRPC stream started")
            # Phase 16.3 Step 3 audit fix — re-subscribe rug-event watcher
            # for recovered positions. gRPC pool subscriptions live in daemon
            # memory and are LOST on container restart; without this loop,
            # recovered positions silently lose rug_event_exit coverage.
            for _pos in self._positions:
                if _pos.token.pool_address:
                    try:
                        self._grpc_stream.watch_pool(
                            _pos.token.pool_address, _pos.token.mint_address)
                        logger.info(
                            f"  gRPC: re-subscribed {_pos.token.symbol} "
                            f"rug-event watcher (recovered position)")
                    except Exception as e:
                        logger.warning(
                            f"  gRPC: re-subscribe failed for {_pos.token.symbol}: {e}")
            # Rug-event watcher ALWAYS runs — it records every detected rug
            # event to `rug_events` for post-hoc analysis. The gate is on
            # ACTION (panic sell), not on observation.
            self._rug_event_watcher_task = asyncio.ensure_future(
                self._rug_event_watcher())
            mode = "EXIT_ENABLED" if self.config.rug_event_exit_enabled else "OBSERVE_ONLY"
            logger.info(
                f"M4: rug-event watcher started [{mode}] "
                f"(sol_threshold={self.config.rug_event_sol_threshold:.1f} SOL)")

            # Phase 22.S.P1 — price event watcher (gRPC fast SL/EC trigger)
            self._price_event_watcher_task = asyncio.ensure_future(
                self._price_event_watcher())
            p1_mode = ("OFF" if not self.config.grpc_event_exit_enabled
                        else ("SHADOW" if self.config.grpc_event_exit_shadow_only
                               else "LIVE"))
            logger.info(
                f"M4: price-event watcher started [{p1_mode}] "
                f"(SL={'on' if self.config.grpc_event_exit_sl_enabled else 'off'}, "
                f"EC={'on' if self.config.grpc_event_exit_ec_enabled else 'off'}, "
                f"min_delta={self.config.grpc_event_min_price_delta:.2%})")

        # Shadow resolver tasks (exit-warn + sw_v2). Guards verbatim in harness.
        shadow_orchestrator.start_resolvers(self)

        # Creator cap_hit resolver — async worker pool that resolves each new
        # mint's cap_hit flag and caches in DB. Closes the v3b snapshot gap
        # window. Started here so RPC client is initialized.
        # See creator_resolver.py + V3B_AUDIT_FINDINGS.md C-4.
        if self._creator_resolver_pending and self.discovery is not None:
            try:
                rpc = await self.discovery._get_rpc()
                if rpc is not None:
                    # Funder resolver MUST start first so creator_resolver
                    # can chain-enqueue into it.
                    if HAS_FUNDER_RESOLVER:
                        try:
                            self.funder_resolver = FunderResolver(
                                rpc=rpc, db=self.db, n_workers=3)
                            await self.funder_resolver.start()
                            logger.info("FunderResolver started (3 workers)")
                        except Exception as e:
                            logger.error(f"funder_resolver start failed: {e}")
                            self.funder_resolver = None

                    # Creator resolver — pass chain callback that enqueues
                    # to funder_resolver when a non-cap-hit creator is resolved.
                    # Audit fix 2026-05-19: bind funder_resolver as default arg
                    # to avoid dangling reference if self.funder_resolver is
                    # later set to None (on_stop or init failure mid-flight).
                    funder_chain_cb = None
                    if self.funder_resolver is not None:
                        _fr = self.funder_resolver
                        funder_chain_cb = (
                            lambda creator, fr=_fr:
                                fr.enqueue(creator) if fr is not None else None)

                    self.creator_resolver = CreatorResolver(
                        rpc=rpc, db=self.db, n_workers=3,
                        on_creator_resolved=funder_chain_cb)
                    await self.creator_resolver.start()
                else:
                    logger.warning("creator_resolver: discovery._get_rpc() returned None")
            except Exception as e:
                logger.error(f"creator_resolver start failed: {e}")
                self.creator_resolver = None
            logger.info(
                "M4: shadow-exit-warn resolver started "
                f"(interval={self.config.shadow_exit_warn_resolver_interval}s, "
                f"tick_interval={self.config.shadow_exit_warn_tick_interval}s)")

        # Recover pending observations into memory (for intermediate snapshots + flush)
        asyncio.ensure_future(self._recover_observations())

    def _compute_inline_vwmp_for_mint(self, mint: str):
        """Phase B.2 v4 — helper that synthesizes inline VWMP for a mint NOT
        currently held as a position.

        Mirrors the existing position-time VWMP path (meme_sniper.py:8330+):
        pulls from `_grpc_stream.get_recent_swap_records(mint, max_age_sec)`
        then calls compute_vwmp_with_diagnostics with same config params.

        Returns (vwmp_price, vwmp_diag). vwmp_price is None if buffer
        insufficient or stream unavailable. Caller checks for None.
        """
        if not HAS_VWMP:
            return None, None
        stream = getattr(self, "_grpc_stream", None)
        if stream is None or not hasattr(stream, "get_recent_swap_records"):
            return None, None
        try:
            records = stream.get_recent_swap_records(
                mint, max_age_sec=self.config.vwmp_window_s * 2.0)
            if not records:
                return None, None
            swap_tuples = [
                (r.timestamp, r.price_sol, r.volume_sol)
                for r in records
            ]
            return compute_vwmp_with_diagnostics(
                swap_tuples,
                vol_gate=self.config.vwmp_vol_gate_sol,
                window_s=self.config.vwmp_window_s,
                min_swaps=self.config.vwmp_min_swaps,
            )
        except Exception:
            return None, None

    async def _run_b2_shadow_for_observing_mints(self) -> None:
        """Phase B.2 v4 — log shadow rows for observing mints (no position).

        Called once per M4 tick. Iterates `_observation_pool` (capped) +
        any mints with `_swap_collection_until` active. For each:
          1. Synthesize inline VWMP via `_compute_inline_vwmp_for_mint`
          2. Query cs.get_authoritative_price for the same `now`
          3. Compute divergence + 3-tier flags
          4. Persist single-row to cs_price_divergence_log (notes='observing')

        Increments same WINDOW + CUMULATIVE counters as position-side path so
        summary log captures combined view.

        Defensive: try/except around the whole loop body so a single bad
        mint never breaks the tick.
        """
        if not (self.config.cs_divergence_shadow_enabled
                and self.config.cs_divergence_include_observing
                and self._cleaning_service_bridge is not None):
            return

        # Sample rate gate (applies to the whole call, not per-mint, to keep
        # log shape consistent with position path)
        if not (self.config.cs_divergence_log_sample_rate >= 1.0
                or random.random() < self.config.cs_divergence_log_sample_rate):
            return

        # Collect candidate mints — observation pool + swap-collecting set
        cands = set(self._observation_pool.keys()) if self._observation_pool else set()
        cands |= set(self._swap_collection_until.keys()) if self._swap_collection_until else set()
        # Exclude mints with active positions (those are handled by position-side path)
        # Audit fix P0.3: snapshot position mint set at start of tick so the
        # exclusion is atomic vs the iteration body (defends against bot
        # asynchronously opening positions mid-tick).
        position_mints = set()
        if self._positions:
            position_mints = {p.token.mint_address for p in self._positions}
            cands -= position_mints
        if not cands:
            return

        # Audit fix (throttle): per-mint cadence — log a mint at most once per
        # interval_sec. Without this, observing path runs every M4 tick (~1Hz)
        # × ~30 observing mints = 820K rows/day (real-data, 2026-05-19).
        now_mono = time.monotonic()
        interval = float(self.config.cs_divergence_observing_per_mint_interval_sec)
        if interval > 0:
            cands = {
                m for m in cands
                if now_mono - self._cs_observing_last_t.get(m, 0.0) >= interval
            }
            if not cands:
                return

        # Cap to avoid hot-loop blowup (deterministic order via sorted to make
        # mint selection reproducible across ticks — P2.3 fix)
        cap = max(int(self.config.cs_divergence_observing_max_per_tick), 0)
        cands_list = sorted(cands)
        if cap and len(cands_list) > cap:
            cands_list = cands_list[:cap]

        for mint in cands_list:
            # Audit fix P0.3 (defense in depth): re-check position membership
            # right before logging — if a position opened during this iteration,
            # skip the mint so it's not double-counted by both paths.
            if mint in position_mints:
                continue
            try:
                self._cs_divergence_query_count += 1
                self._cs_divergence_query_count_window += 1
                # Mark this mint as "sampled this tick" — throttle gate
                self._cs_observing_last_t[mint] = now_mono

                inline_price = None
                inline_status = None
                inline_had_error = False
                try:
                    v, diag = self._compute_inline_vwmp_for_mint(mint)
                    if v is not None and v > 0:
                        inline_price = float(v)
                    if diag is not None:
                        inline_status = diag.get('vwmp_status')
                except Exception:
                    # Audit fix P1.3: separate inline-error count from null-count.
                    # Silent exception was previously indistinguishable from
                    # legitimate no-data (insufficient swaps). Now tracked.
                    inline_had_error = True
                    self._cs_divergence_inline_error_count += 1

                # Query cs
                cs_authoritative_price = None
                cs_latency_ms = None
                try:
                    t_start = time.perf_counter()
                    cs_authoritative_price = await self._cleaning_service_bridge.service.get_authoritative_price(
                        mint, time.time(),
                    )
                    cs_latency_ms = (time.perf_counter() - t_start) * 1000.0
                    self._cs_latency_ms_samples.append(cs_latency_ms)
                except Exception as _cs_e:
                    if self._cs_divergence_error_count < 10:
                        logger.warning(
                            "Phase B.2 (observing): get_authoritative_price raised for %s: %s",
                            mint[:12], _cs_e,
                        )
                    self._cs_divergence_error_count += 1

                inline_returned_null = 1 if inline_price is None else 0
                cs_returned_null = 1 if cs_authoritative_price is None else 0
                if inline_returned_null:
                    self._cs_divergence_inline_null_count += 1
                    self._cs_divergence_inline_null_count_window += 1
                if cs_returned_null:
                    self._cs_divergence_cs_null_count += 1
                    self._cs_divergence_cs_null_count_window += 1

                divergence_pct = None
                flagged_5pct = flagged_2pct = flagged_1pct = 0
                if (inline_price is not None
                        and cs_authoritative_price is not None
                        and inline_price > 0):
                    divergence_pct = abs(cs_authoritative_price - inline_price) / inline_price
                    flagged_5pct = 1 if divergence_pct > 0.05 else 0
                    flagged_2pct = 1 if divergence_pct > 0.02 else 0
                    flagged_1pct = 1 if divergence_pct > 0.01 else 0
                    if flagged_5pct:
                        self._cs_divergence_flagged_5pct += 1
                        self._cs_divergence_flagged_5pct_window += 1
                    if flagged_2pct:
                        self._cs_divergence_flagged_2pct += 1
                        self._cs_divergence_flagged_2pct_window += 1
                    if flagged_1pct:
                        self._cs_divergence_flagged_1pct += 1
                        self._cs_divergence_flagged_1pct_window += 1
                    self._cs_divergence_total_count += 1
                    self._cs_divergence_total_count_window += 1
                    self._cs_divergence_pct_samples.append(divergence_pct)

                # Persist (defensive)
                try:
                    self.db.log_cs_price_divergence(
                        timestamp=time.time(),
                        mint_address=mint,
                        pool_price_sol=None,    # no pool snapshot in observing path
                        inline_vwmp_price=inline_price,
                        inline_vwmp_status=inline_status,
                        inline_returned_null=inline_returned_null,
                        cs_authoritative_price=cs_authoritative_price,
                        cs_returned_null=cs_returned_null,
                        cs_latency_ms=cs_latency_ms,
                        divergence_pct=divergence_pct,
                        flagged_5pct=flagged_5pct,
                        flagged_2pct=flagged_2pct,
                        flagged_1pct=flagged_1pct,
                        current_price_used=inline_price,  # observing has no position price; inline is the canonical
                        notes="observing",
                    )
                except Exception:
                    pass
            except Exception as _e:
                # Top-level defensive — never let observing shadow break M4 tick
                if self._cs_divergence_error_count < 10:
                    logger.warning(
                        "Phase B.2 (observing): unexpected error for %s: %s",
                        mint[:12], _e,
                    )
                self._cs_divergence_error_count += 1

    def _emit_cs_divergence_summary(self) -> None:
        return cs_divergence_monitor.emit_cs_divergence_summary(self)

    def _emit_b4_inference_summary(self) -> None:
        """Phase B.4 5-min summary (piggy-backed on cs_divergence_summary cadence).

        Emits per-model WINDOW + CUMULATIVE stats. Reset window counters
        each emit (matches Week 4 audit pattern). Mode tag indicates
        whether PROMOTE flag is on for the model — operators can see at
        a glance whether shadow data still being collected vs live drive.
        """
        for tag, attempts_cum, agree_cum, disagree_cum, nnull_cum, \
                attempts_w, agree_w, disagree_w, nnull_w, promoted in [
            ("V5.5.3",
             self._cs_b4_v5_5_3_attempt_count,
             self._cs_b4_v5_5_3_agreement_count,
             self._cs_b4_v5_5_3_disagreement_count,
             self._cs_b4_v5_5_3_new_null_count,
             self._cs_b4_v5_5_3_attempt_count_window,
             self._cs_b4_v5_5_3_agreement_count_window,
             self._cs_b4_v5_5_3_disagreement_count_window,
             self._cs_b4_v5_5_3_new_null_count_window,
             self.config.cs_b4_promoted_v5_5_3),
            ("V5.5.2",
             self._cs_b4_v5_5_2_attempt_count,
             self._cs_b4_v5_5_2_agreement_count,
             self._cs_b4_v5_5_2_disagreement_count,
             self._cs_b4_v5_5_2_new_null_count,
             self._cs_b4_v5_5_2_attempt_count_window,
             self._cs_b4_v5_5_2_agreement_count_window,
             self._cs_b4_v5_5_2_disagreement_count_window,
             self._cs_b4_v5_5_2_new_null_count_window,
             self.config.cs_b4_promoted_v5_5_2),
            ("V5.5.6",
             self._cs_b4_v5_5_6_attempt_count,
             self._cs_b4_v5_5_6_agreement_count,
             self._cs_b4_v5_5_6_disagreement_count,
             self._cs_b4_v5_5_6_new_null_count,
             self._cs_b4_v5_5_6_attempt_count_window,
             self._cs_b4_v5_5_6_agreement_count_window,
             self._cs_b4_v5_5_6_disagreement_count_window,
             self._cs_b4_v5_5_6_new_null_count_window,
             self.config.cs_b4_promoted_v5_5_6),
            ("V5.5.7 v2.1",
             self._cs_b4_v5_5_7_v2_1_attempt_count,
             self._cs_b4_v5_5_7_v2_1_agreement_count,
             self._cs_b4_v5_5_7_v2_1_disagreement_count,
             self._cs_b4_v5_5_7_v2_1_new_null_count,
             self._cs_b4_v5_5_7_v2_1_attempt_count_window,
             self._cs_b4_v5_5_7_v2_1_agreement_count_window,
             self._cs_b4_v5_5_7_v2_1_disagreement_count_window,
             self._cs_b4_v5_5_7_v2_1_new_null_count_window,
             self.config.cs_b4_promoted_v5_5_7_v2_1),
            ("V5.6",
             self._cs_b4_v5_6_attempt_count,
             self._cs_b4_v5_6_agreement_count,
             self._cs_b4_v5_6_disagreement_count,
             self._cs_b4_v5_6_new_null_count,
             self._cs_b4_v5_6_attempt_count_window,
             self._cs_b4_v5_6_agreement_count_window,
             self._cs_b4_v5_6_disagreement_count_window,
             self._cs_b4_v5_6_new_null_count_window,
             self.config.cs_b4_promoted_v5_6),
        ]:
            if attempts_cum == 0:
                continue
            mode = "PROMOTED" if promoted else "SHADOW"
            ar_w = (100.0 * agree_w / attempts_w) if attempts_w > 0 else 0.0
            ar_c = (100.0 * agree_cum / attempts_cum) if attempts_cum > 0 else 0.0
            logger.info(
                f"Phase B.4 {tag} WINDOW [{mode}]: attempts={attempts_w} "
                f"agreement={agree_w} ({ar_w:.1f}%) disagreement={disagree_w} "
                f"new_null={nnull_w} errors_old={self._cs_b4_old_compute_error_count} "
                f"errors_new={self._cs_b4_new_compute_error_count}")
            logger.info(
                f"Phase B.4 {tag} CUMULATIVE [{mode}]: agreement_rate={ar_c:.1f}% "
                f"over {attempts_cum} attempts (disagree={disagree_cum}, "
                f"new_null={nnull_cum})")
        # Reset window counters
        self._cs_b4_v5_5_3_attempt_count_window = 0
        self._cs_b4_v5_5_3_agreement_count_window = 0
        self._cs_b4_v5_5_3_disagreement_count_window = 0
        self._cs_b4_v5_5_3_new_null_count_window = 0
        self._cs_b4_v5_5_2_attempt_count_window = 0
        self._cs_b4_v5_5_2_agreement_count_window = 0
        self._cs_b4_v5_5_2_disagreement_count_window = 0
        self._cs_b4_v5_5_2_new_null_count_window = 0
        self._cs_b4_v5_5_6_attempt_count_window = 0
        self._cs_b4_v5_5_6_agreement_count_window = 0
        self._cs_b4_v5_5_6_disagreement_count_window = 0
        self._cs_b4_v5_5_6_new_null_count_window = 0
        self._cs_b4_v5_5_7_v2_1_attempt_count_window = 0
        self._cs_b4_v5_5_7_v2_1_agreement_count_window = 0
        self._cs_b4_v5_5_7_v2_1_disagreement_count_window = 0
        self._cs_b4_v5_5_7_v2_1_new_null_count_window = 0
        self._cs_b4_v5_6_attempt_count_window = 0
        self._cs_b4_v5_6_agreement_count_window = 0
        self._cs_b4_v5_6_disagreement_count_window = 0
        self._cs_b4_v5_6_new_null_count_window = 0

    def on_stop(self):
        """Cleanup on shutdown."""
        total_pnl = sum(t.pnl_usd for t in self._trade_log)
        logger.info("=" * 60)
        logger.info("MemeSniper controller STOPPING")
        logger.info(f"  total_trades = {self.risk.total_trades}")
        logger.info(f"  total_pnl    = ${total_pnl:.2f}")
        logger.info(f"  positions    = {len(self._positions)}")
        logger.info(f"  swap_keepalive_active = {len(self._swap_keepalive_mints)}")

        # v0.7.16 (Bug 19 root-cause fix) — cancel in-flight reconcile tasks
        # FIRST so they get a CancelledError at their next await, then flush
        # all in-memory Positions to DB so the orphan_scanner can recover any
        # mid-reconcile builds on next boot.
        try:
            buy_inflight = getattr(self, "_reconcile_inflight", None) or {}
            sell_inflight = getattr(self, "_sell_reconcile_inflight", None) or {}
            for mint, task in list(buy_inflight.items()):
                try:
                    task.cancel()
                except Exception:
                    pass
            for mint, task in list(sell_inflight.items()):
                try:
                    task.cancel()
                except Exception:
                    pass
            if buy_inflight or sell_inflight:
                logger.warning(
                    f"on_stop: cancelled {len(buy_inflight)} BUY + "
                    f"{len(sell_inflight)} SELL in-flight reconciles")
        except Exception as e:
            logger.warning(f"on_stop: reconcile-cancel pass raised: {e}")

        # Flush memory-only Positions to DB. save_position is INSERT OR REPLACE
        # so this is idempotent: positions already saved get re-written with
        # same data; positions only in memory (reconcile won the race but DB
        # save was deferred / failed) become recoverable on next boot.
        flushed = 0
        for pos in list(self._positions):
            if getattr(pos, "_removed", False):
                continue
            try:
                self.db.save_position(pos)
                flushed += 1
            except Exception as e:
                logger.warning(
                    f"on_stop: save_position failed for {pos.token.symbol}: {e}")
        if flushed:
            logger.warning(f"on_stop: flushed {flushed} in-memory positions to DB")
        if self.creator_resolver is not None:
            stats = self.creator_resolver.stats()
            logger.info(f"  creator_resolver stats = {stats}")
            # Cancel workers synchronously (they exit on next loop iteration)
            for w in self.creator_resolver._workers:
                w.cancel()
        if self.funder_resolver is not None:
            fstats = self.funder_resolver.stats()
            logger.info(f"  funder_resolver stats = {fstats}")
            for w in self.funder_resolver._workers:
                w.cancel()
        logger.info("=" * 60)

        # Flush any swap collectors that are still in keepalive mode (open
        # positions or queued candidates) so we don't lose hold-period data
        # on graceful shutdown. After this, the local DB has the best-effort
        # snapshot for any in-flight trades.
        for mint in list(self._swap_keepalive_mints):
            try:
                self._cleanup_swap_keepalive(mint, reason="on_stop")
            except Exception as e:
                logger.warning(f"on_stop: keepalive flush failed for {mint[:12]}...: {e}")

        # Also flush any non-keepalive collectors (30min observation set) so
        # we don't lose buffered swaps on graceful shutdown. Restart will pick
        # up new tokens from M1 anyway.
        for mint in list(self._swap_collection_until.keys()):
            try:
                self._flush_swaps_to_db(mint, unregister=True)
            except Exception as e:
                logger.warning(f"on_stop: obs-collector flush failed for {mint[:12]}...: {e}")

        # Stop gRPC stream
        if self._grpc_stream:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._grpc_stream.stop())
            except RuntimeError:
                pass

        # Phase B.1: stop cleaning_service bridge (flushes WAL, cancels logger)
        if self._cleaning_service_bridge is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._cleaning_service_bridge.stop())
            except RuntimeError:
                # No running loop — best-effort sync WAL close
                try:
                    self._cleaning_service_bridge.service.close_wal()
                except Exception:
                    pass

        self.db.record_event("INFO", "on_stop",
                             f"Stopped: trades={self.risk.total_trades}, pnl=${total_pnl:.2f}")
        self.db.close()
        # Close http sessions — schedule in running event loop if available
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        for client in [self.discovery, self.signal, self.trader, self.kline_builder]:
            if client and hasattr(client, "close"):
                if loop and loop.is_running():
                    loop.create_task(client.close())
                else:
                    # No running loop — close synchronously via new loop
                    try:
                        asyncio.get_event_loop().run_until_complete(client.close())
                    except Exception:
                        pass

    def get_custom_info(self) -> dict:
        """Published via MQTT every 1s for hummingbot-api monitoring."""
        return {
            "total_trades": self.risk.total_trades,
            "active_positions": len(self._positions),
            "pending_tokens": len(self._pending),
            "candidate_queue": len(self._candidate_queue),
            "candidates_seen": self._candidates_seen,
            "cumulative_pnl_usd": round(sum(t.pnl_usd for t in self._trade_log), 4),
            "cumulative_pnl_sol": round(sum(t.pnl_sol for t in self._trade_log), 6),
            "sol_price_usd": round(self._sol_price_usd, 2),
            "risk": self.risk.status(),
            "positions": [
                {
                    "symbol": p.token.symbol,
                    "entry_price_sol": p.entry_price_sol,
                    "hold_sec": round(p.hold_seconds()),
                    "sol_invested": p.sol_invested,
                    "peak_pnl_pct": round(p.peak_pnl_pct * 100, 1),
                    "trailing_active": p.trailing_activated,
                }
                for p in self._positions
            ],
            "recent_trades": [
                {
                    "symbol": t.token_symbol,
                    "pnl_usd": round(t.pnl_usd, 4),
                    "exit_reason": t.exit_reason,
                    "hold_sec": round(t.hold_seconds),
                }
                for t in self._trade_log[-5:]
            ],
        }

    # ──────────────────────────────────────────
    # Internal methods
    # ──────────────────────────────────────────

    def _sweep_expired_swap_collectors(self, now: float):
        """Final-flush + unregister swap collectors that have hit their deadline.

        Skips mints in _swap_keepalive_mints (real trade still in flight —
        keepalive-cleanup at trade exit handles those).
        """
        if not self._swap_collection_until:
            return
        expired = [m for m, deadline in self._swap_collection_until.items()
                   if deadline <= now and m not in self._swap_keepalive_mints]
        for mint in expired:
            try:
                self._flush_swaps_to_db(mint, unregister=True)
            except Exception as e:
                logger.warning(f"swap-sweep: flush failed for {mint[:12]}...: {e}")
            # Defense in depth: ensure deadline removed even if flush early-returned
            # (e.g. kline_builder already unregistered via another path).
            self._swap_collection_until.pop(mint, None)
        if expired:
            logger.info(f"swap-sweep: flushed {len(expired)} expired collector(s)")

    def _flush_swaps_to_db(self, mint: str, unregister: bool = True):
        """Save collected swap data to DB for backtesting, optionally unregister from kline builder."""
        if not self.kline_builder or not self.kline_builder.is_registered(mint):
            return
        swaps = self.kline_builder.get_swaps(mint)
        pool = self.kline_builder.get_pool_address(mint)
        if swaps and pool:
            try:
                self.db.save_swaps(mint, pool, swaps)
                logger.info(f"KlineBuilder: saved {len(swaps)} swaps to DB for {mint[:12]}...")
            except Exception as e:
                logger.warning(f"KlineBuilder: failed to save swaps for {mint[:12]}...: {e}")
        if unregister:
            self.kline_builder.unregister(mint)
            self._pool_discovery_attempts.pop(mint, None)
            self._swap_collection_until.pop(mint, None)

    def _cleanup_swap_keepalive(self, mint: str, reason: str = ""):
        """Final flush + unregister for a token that was kept alive past M2.

        Idempotent — safe to call multiple times. Removes the mint from the
        keepalive set, persists any remaining swaps, and unregisters the
        kline builder so memory is freed.

        Call this when a token leaves the trade lifecycle:
          - trade exit (success or failure)
          - buy retry exhausted (will never trade)
          - candidate queue cleared (hard block)
        """
        if mint not in self._swap_keepalive_mints:
            return
        self._swap_keepalive_mints.discard(mint)
        if reason:
            logger.info(f"KlineBuilder: cleanup {mint[:12]}... ({reason})")
        self._flush_swaps_to_db(mint, unregister=True)

    def _add_to_observation_pool(self, token: GraduatedToken,
                                  model_score, model_passed: bool,
                                  reject_reason: str = "",
                                  features=None, kline_6m=None,
                                  keep_swap_collector: bool = False):
        """Record observation to DB and schedule deferred GMGN kline fetch.

        Args:
            keep_swap_collector: if True, do not unregister the swap collector.
                Used when the token is heading toward a real trade — we want to
                keep collecting swap data through the entire hold period so the
                local DB has the full price path for P12-D training. The final
                flush + unregister happens at trade exit (or at preflight/buy
                failure paths).
        """
        if token.mint_address in self._observation_pool:
            return

        # Async enqueue v3b creator_resolver — covers ALL obs-pool entry paths
        # (M1 main loop, M1c hot_rank, M1b trending, simple_t5m_pending etc.)
        # so v3b shadow at +600s can lookup cap_hit from DB. Idempotent.
        if self.creator_resolver is not None:
            try:
                self.creator_resolver.enqueue(token.mint_address)
            except Exception as e:
                logger.debug(f"creator_resolver.enqueue failed: {e}")

        if keep_swap_collector:
            # Save what we have so far, but DO NOT unregister — collector keeps
            # running and will be flushed again at trade exit.
            self._flush_swaps_to_db(token.mint_address, unregister=False)
            self._swap_keepalive_mints.add(token.mint_address)
        else:
            # Save what we have, but keep the collector running until the
            # swap_collection_window_sec deadline (30min default). The
            # heartbeat sweep finalizes flush + unregister at deadline.
            self._flush_swaps_to_db(token.mint_address, unregister=False)
        try:
            gmgn_entry = getattr(token, "_gmgn_info_raw", None)
            # T2.1 observability: record when the entry snapshot was collected.
            # _gmgn_info_raw is populated by M1 at discovery time (no separate fetch
            # here); we record collected_at=now as the write-time proxy and
            # fallback_used=False (M1 always uses GMGN directly — no fallback branch
            # at this call site).
            entry_meta = (
                {"entry": {"collected_at": time.time(), "fallback_used": False, "source": "gmgn"}}
                if gmgn_entry else None
            )
            obs_id = self.db.record_observation(
                token, model_score, model_passed,
                reject_reason=reject_reason,
                features=features, kline_6m=kline_6m,
                gmgn_info_entry=gmgn_entry,
                gmgn_collection_meta=entry_meta)
            # Fetch GMGN kline 62 min after graduation (60 min data + 2 min buffer for GMGN indexing)
            expire = token.graduation_time + 62 * 60
            entry = ObservationEntry(token=token, obs_id=obs_id, expire_time=expire)
            self._observation_pool[token.mint_address] = entry

            # Snapshot timing fix: gmgn_info_t0 historically fired ~T+10min due
            # to observation-pool loop cadence. But _gmgn_info_raw was captured
            # by M1 at ~T+30-60s (when graduation was first detected). Write it
            # to gmgn_info_t0 NOW and mark the snapshot done so the scheduler
            # skips the late re-fetch. This gives true T+30-60s GMGN data in
            # the t0 column for v3.1+ and analytics.
            if gmgn_entry:
                try:
                    t0_meta = {"t0": {"collected_at": time.time(), "fallback_used": False, "source": "gmgn"}}
                    self.db.update_observation_snapshot(
                        obs_id, "gmgn_info_t0", gmgn_entry,
                        gmgn_collection_meta=t0_meta)
                    entry.snapshots_done.add("gmgn_info_t0")
                except Exception as e:
                    logger.debug(f"OBS: {token.symbol} early gmgn_info_t0 write failed: {e}")

            self._register_shadow_pending(token, obs_id)
            logger.info(f"OBS: {token.symbol} added to observation pool "
                        f"(id={obs_id}, GMGN fetch in {expire - time.time():.0f}s)")
        except Exception as e:
            logger.warning(f"OBS: failed to record observation for {token.symbol}: {e}")

    def _record_latency_event(self, token: GraduatedToken, event_name: str, **metadata):
        """Best-effort live latency instrumentation.

        Pass ``error_category=<ErrorCategory.VALUE>`` (or the bare string) to
        populate the T2.3 structured column.  Any other kwargs go into the
        free-text metadata JSON blob as before (behavior-preserving).
        """
        if not self.config.latency_events_enabled:
            return
        try:
            # T2.3: extract error_category from kwargs so it lands in its own
            # column rather than being embedded in the metadata JSON blob.
            error_category = metadata.pop("error_category", None)
            age_sec = time.time() - token.graduation_time if token.graduation_time else None
            self.db.record_latency_event(
                mint_address=token.mint_address,
                symbol=token.symbol,
                graduation_time=token.graduation_time,
                event_name=event_name,
                age_sec=age_sec,
                metadata=metadata or None,
                error_category=error_category,
            )
        except Exception as e:
            logger.debug(f"LATENCY: failed to record {event_name} for {token.symbol}: {e}")

    @staticmethod
    def _classify_preflight_failure(reject_reason: str) -> str:
        """Classify a preflight failure for retry-backoff selection.

        Returns one of:
          "soft"     — AMM/price jitter; retry immediately (next tick).
                       Most often resolves within a single Jupiter recompute.
          "hard"     — Upstream API down or transient network issue;
                       exponential backoff so we don't hammer the dead service.
          "terminal" — Pool/token property won't change; give up entirely
                       (token evicted from queue, no further retry).
        """
        if not reject_reason:
            return "hard"  # unknown → safe default
        # Soft = recoverable on re-quote within 1-2s
        SOFT_PREFIXES = (
            "price_impact=",
            "entry_chase=",
            "price_divergence=",
            "route_missing_biggest_pool",   # Jupiter routing varies tick-to-tick
            "invalid_quote",                # Jupiter may return empty quote transiently
        )
        # Terminal = property of the token/pool, won't change
        TERMINAL_PREFIXES = (
            "live_liquidity=",              # pool too small, structural
            "cum_return=",                  # already crashed
            "cum_drawdown=",                # already crashed
            "lookback_return=",             # last 3 bars confirmed dump
            "rug_gate ",                    # rug filter said no
            "missing_biggest_pool_address",
            # 2026-05-19 v0.7.1 — Gateway timeout with NO signature recoverable:
            # one 30/60s execute call timed out and we couldn't extract a tx
            # signature from the exception → cannot reconcile on-chain. The
            # bottleneck is Solana RPC / Jupiter — retrying in 20s will hit
            # the same wall. Terminal-after-first-attempt = abandon the coin.
            "gateway_timeout",
            # 2026-05-19 v0.7.4 — Gateway exception with no signature found.
            # Distinct from gateway_timeout (latter is network-level; this is
            # missing-data level). Cannot reconcile → abandon.
            "onchain_no_sig",
            # 2026-05-19 v0.7.4 — `_execute_buy` saw status != 1 AND
            # _spawn_buy_reconciliation succeeded (background task in flight).
            # We must NOT retry this candidate while reconcile is in progress
            # — re-calling _execute_buy would send a SECOND on-chain tx that
            # may double-confirm if the first one also confirms. Setting
            # terminal here makes _evaluate_pending evict the candidate on
            # the next tick (buy_fail_count > 0 condition at line 3106).
            # If reconcile times out / reports dropped, M1 graduation path
            # can re-discover the token (no permanent ban).
            "buy_pending_onchain_verify",
            # NOTE 2026-05-19 v0.7.4 — REMOVED `onchain_status_0` and
            # `buy_zero_tokens` from TERMINAL. v0.7.3 treated them as terminal
            # which compounded the BUG: Gateway status=0 is PENDING (not fail),
            # tx may still confirm on-chain. Real data: DAVID 10:11:24 returned
            # status=0; wallet later showed 3051 tokens (~quote.amount_out × 0.988).
            # v0.7.4 routes these to TERMINAL via the explicit
            # `buy_pending_onchain_verify` reason when reconcile spawns, and
            # via `onchain_no_sig` when reconcile is disabled / unavailable.
        )
        for prefix in SOFT_PREFIXES:
            if reject_reason.startswith(prefix):
                return "soft"
        for prefix in TERMINAL_PREFIXES:
            if reject_reason.startswith(prefix):
                return "terminal"
        # Default = hard (snapshot_unavailable, jupiter_quote_error, etc.)
        return "hard"

    @staticmethod
    def _safe_relative_delta(actual: Optional[float], reference: Optional[float]) -> Optional[float]:
        """Return actual/reference - 1 when both values are usable."""
        try:
            actual_f = float(actual) if actual is not None else None
            reference_f = float(reference) if reference is not None else None
        except (TypeError, ValueError):
            return None
        if actual_f is None or reference_f is None or reference_f <= 0:
            return None
        return actual_f / reference_f - 1.0

    def _record_sell_trigger_once(self, pos: Position, reason: str, **metadata):
        mint = pos.token.mint_address
        if self._exit_signal_reason.get(mint) == reason:
            return
        self._exit_signal_reason[mint] = reason
        self._record_latency_event(pos.token, "sell_trigger", reason=reason, **metadata)

    def _sell_retry_wait_sec(self, mint: str) -> float:
        retry_after = self._sell_retry_after.get(mint, 0.0)
        return max(0.0, retry_after - time.time())

    def _defer_exit_for_retry(self, pos: Position, reason: str) -> bool:
        """Suppress repeated exit logs while a failed sell is in backoff."""
        mint = pos.token.mint_address
        wait_sec = self._sell_retry_wait_sec(mint)
        if wait_sec <= 0:
            return False
        now = time.time()
        if now >= self._sell_retry_notice_after.get(mint, 0.0):
            attempt = self._sell_retry_count.get(mint, 0)
            logger.info(
                f"M4: {pos.token.symbol} — exit retry pending "
                f"(reason={reason}, retry_in={wait_sec:.0f}s, attempt #{attempt})"
            )
            self._sell_retry_notice_after[mint] = now + min(60.0, max(wait_sec, 5.0))
        return True

    # P_RETRY A1+A2 helper (2026-05-15) — see
    # research_notebooks/meme_sniper/latency/exit/执行延迟/06_DYING_POOL_RETRY.md
    MAX_SELL_RETRIES = 5
    ABANDON_COOLDOWN_SEC = 3600.0  # 1h pause after MAX retries

    def _schedule_sell_retry(self, pos: Position, reason: str,
                             context: str) -> tuple[bool, int, float]:
        """Centralized sell-retry scheduler with A1+A2 patches (P_RETRY).

        A1 (2026-05-15): time_limit + early_crash promoted to urgent backoff.
            Previously non-urgent (30→60→120→240→300s), causing 75-100s of
            avoidable retry latency on dying-pool exits. Now: 5→10→15→20s
            linear, capped at 20s.
        A2 (2026-05-15): Cap consecutive retries at MAX_SELL_RETRIES (=5).
            Empirical: `fomo` mint 2026-04-29 retried 43 times = ~3.4 hours
            wasted on a permanently-failing sell. After cap, log abandon
            event + 1h cooldown + reset state.

        Returns: (abandoned, fails, backoff_sec)
            abandoned=True → caller must `return` immediately, position
                is paused for 1h.
            fails / backoff_sec → for logging the active attempt.
        """
        mint = pos.token.mint_address
        fails = self._sell_retry_count.get(mint, 0) + 1

        if fails > self.MAX_SELL_RETRIES:
            self._sell_retry_count[mint] = 0
            self._sell_retry_after[mint] = time.time() + self.ABANDON_COOLDOWN_SEC
            logger.error(
                f"M3: [SELL ABANDONED] {pos.token.symbol} reason={reason} "
                f"after {fails - 1} consecutive fails ({context}) — pause "
                f"{self.ABANDON_COOLDOWN_SEC:.0f}s")
            try:
                self._record_latency_event(
                    pos.token, "sell_abandoned",
                    reason=reason, attempts=fails - 1, context=context)
                self.db.record_event(
                    "ERROR", "sell",
                    f"SELL ABANDONED {pos.token.symbol} after {fails - 1} fails")
            except Exception:
                pass
            return (True, fails - 1, self.ABANDON_COOLDOWN_SEC)

        self._sell_retry_count[mint] = fails
        # A1: early_crash + time_limit urgent. P4 (2026-06-03): promote take_profit
        # + ALL soft-exit reasons + rug_event/rug_protect_blind/price_unavailable to
        # urgent so every time-sensitive exit retries on 5s-linear (cap 20s) backoff
        # instead of 30->300s exponential. rug_event is the highest-urgency exit
        # (gRPC-detected active dump, racing to panic-sell) and must NOT sit on the
        # slow path. Full 14-reason emitted universe (verified vs source 2026-06-03).
        urgent = reason in ("stop_loss", "trailing_stop",
                            "early_crash", "time_limit", "take_profit",
                            "v5_softexit", "v5_3_softexit", "v5_5_softexit",
                            "v5_5_3_softexit", "v5_6_softexit", "v4_softexit",
                            "rug_event", "rug_protect_blind", "price_unavailable")
        backoff = (min(5 * fails, 20) if urgent
                   else min(30 * (2 ** (fails - 1)), 300))
        logger.info(f"P4_BACKOFF reason={reason} urgent={urgent} fails={fails} "
                    f"backoff={backoff:.0f}s mint={mint[:8]}")
        self._sell_retry_after[mint] = time.time() + backoff
        return (False, fails, float(backoff))

    def _get_wallet_addr(self) -> str:
        """v0.7.17.6: env-first wallet address resolver. The Field
        wallet_address has `is_updatable=True` → Hummingbot config refresh
        can reset it to the empty yml default at any time. Routes that need
        wallet for RPC calls MUST go through here, else they pass "" to
        RPC and get `WrongSize` (base58 decode of 0 bytes != 32 bytes)."""
        return (os.environ.get("SOLANA_WALLET_ADDRESS", "").strip()
                or self.config.wallet_address)

    async def _get_wallet_token_balance_ui(self, mint_address: str) -> Optional[float]:
        """P5-7 delegate → ms/execution/position_ops.get_wallet_token_balance_ui."""
        return await _get_wallet_token_balance_ui_fn(self, mint_address)

    def _prune_stale_position(self, pos: Position, reason: str, wallet_balance_ui: float):
        """P5-7 delegate → ms/execution/position_ops.prune_stale_position."""
        _prune_stale_position_fn(self, pos, reason, wallet_balance_ui)

    async def _recover_phantom_sell(self, pos: "Position", reason: str,
                                     wallet_balance_ui: float,
                                     trigger_pnl_pct: float = 0.0,
                                     trigger_price_sol: Optional[float] = None,
                                     trigger_mid_price_sol: Optional[float] = None,
                                     trigger_mid_source: Optional[str] = None) -> bool:
        """P5-5 delegate → ms/execution/reconciliation.recover_phantom_sell."""
        return await _recover_phantom_sell_fn(
            self, pos, reason, wallet_balance_ui,
            trigger_pnl_pct, trigger_price_sol,
            trigger_mid_price_sol, trigger_mid_source)

    # ──────────────────────────────────────────────────────────────────
    # v0.7.4 BUY RECONCILIATION (2026-05-19) — P5-5 delegates
    # ──────────────────────────────────────────────────────────────────
    async def _reconcile_pending_buy(
        self, candidate: TradeCandidate, signature: str,
        preflight_meta: dict, sol_amount_intended: float,
    ) -> None:
        """P5-5 delegate → ms/execution/reconciliation.reconcile_pending_buy."""
        await _reconcile_pending_buy_fn(
            self, candidate, signature, preflight_meta, sol_amount_intended)

    def _spawn_buy_reconciliation(
        self, candidate: TradeCandidate, signature: str,
        preflight_meta: dict, sol_amount_intended: float,
    ) -> bool:
        """P5-5 delegate → ms/execution/reconciliation.spawn_buy_reconciliation."""
        return _spawn_buy_reconciliation_fn(
            self, candidate, signature, preflight_meta, sol_amount_intended)

    # ──────────────────────────────────────────────────────────────────
    # v0.7.6 SELL RECONCILIATION (2026-05-19) — P5-5 delegates
    # ──────────────────────────────────────────────────────────────────
    async def _reconcile_pending_sell(
        self, pos: Position, signature: str, reason: str,
        trigger_pnl_pct: float,
        trigger_price_sol: Optional[float],
        trigger_mid_price_sol: Optional[float],
        trigger_mid_source: Optional[str],
    ) -> None:
        """P5-5 delegate → ms/execution/reconciliation.reconcile_pending_sell."""
        await _reconcile_pending_sell_fn(
            self, pos, signature, reason,
            trigger_pnl_pct, trigger_price_sol,
            trigger_mid_price_sol, trigger_mid_source)

    def _spawn_sell_reconciliation(
        self, pos: Position, signature: str, reason: str,
        trigger_pnl_pct: float = 0.0,
        trigger_price_sol: Optional[float] = None,
        trigger_mid_price_sol: Optional[float] = None,
        trigger_mid_source: Optional[str] = None,
    ) -> bool:
        """P5-5 delegate → ms/execution/reconciliation.spawn_sell_reconciliation."""
        return _spawn_sell_reconciliation_fn(
            self, pos, signature, reason,
            trigger_pnl_pct, trigger_price_sol,
            trigger_mid_price_sol, trigger_mid_source)

    async def _scan_wallet_for_orphans(self) -> None:
        """P5-5 delegate → ms/execution/reconciliation.scan_wallet_for_orphans."""
        await _scan_wallet_for_orphans_fn(self)

    def _register_shadow_pending(self, token: GraduatedToken, obs_id: int):
        """Queue a token for post-hoc live shadow evaluation once 3m bars exist."""
        if not self.config.shadow_rule_first_enabled:
            pass
        else:
            self._shadow_pending[token.mint_address] = {
                "token": token,
                "obs_id": obs_id,
                "registered_at": time.time(),
                "attempts": 0,
            }
        if self.config.shadow_ev3m_enabled:
            self._shadow_ev3m_pending[token.mint_address] = {
                "token": token,
                "obs_id": obs_id,
                "registered_at": time.time(),
                "attempts": 0,
            }
        if self.config.shadow_super_winner_event_enabled:
            self._shadow_super_winner_event_pending[token.mint_address] = {
                "token": token,
                "obs_id": obs_id,
                "registered_at": time.time(),
                "attempts": 0,
            }
        if self.config.shadow_vshape_enabled and self.shadow_vshape_model is not None:
            self._shadow_vshape_pending[token.mint_address] = {
                "token": token,
                "obs_id": obs_id,
                "registered_at": time.time(),
                "attempts": 0,
            }
        if (self.config.shadow_event_enabled
                and self.flow_model is not None):
            self._shadow_event_pending[token.mint_address] = {
                "token": token,
                "registered_at": time.time(),
                "last_scan_sec": -1,
                "triggered": False,
            }
        # BigWinner pending registration for scan loop.
        # Defensive: check both canonical + legacy in case alias coalesce
        # didn't persist (Phase 25g 2026-05-03 regression).
        if (self._enabled_any(self.config, "big_winner_enabled", "big_winner_v1_enabled")
                and self.big_winner_loaded):
            self._big_winner_pending[token.mint_address] = {
                "token": token,
                "registered_at": time.time(),
                "last_scan_sec": -1,
                "triggered": False,
            }
        if (self.config.shadow_event_invariant_enabled
                and self.event_invariant_model is not None):
            self._shadow_event_invariant_pending[token.mint_address] = {
                "token": token,
                "registered_at": time.time(),
                "last_scan_sec": -1,
                "triggered": False,
            }

    def _cache_ev3m_live_audit(
        self,
        token: GraduatedToken,
        *,
        status: str,
        feature_source: str,
        age_sec: float,
        score: Optional[float],
        band: str,
        cutoff: Optional[float],
        passed: Optional[bool],
        reason: Optional[str],
    ) -> None:
        if not self.config.shadow_ev3m_enabled:
            return
        self._ev3m_live_audit[token.mint_address] = {
            "status": status,
            "feature_source": feature_source,
            "age_sec": float(age_sec),
            "score": float(score) if score is not None else None,
            "band": band,
            "cutoff": float(cutoff) if cutoff is not None else None,
            "passed": passed,
            "reason": reason,
            "ts": time.time(),
        }

    def _build_ev3m_recon_metadata(
        self,
        mint: str,
        shadow_score: Optional[float],
        shadow_feature_source: str,
        ready_age_sec: float,
    ) -> Dict[str, Optional[float]]:
        return entry_scanner.build_ev3m_recon_metadata(
            self, mint, shadow_score, shadow_feature_source, ready_age_sec)

    def _log_ev3m_reconciliation(
        self,
        token: GraduatedToken,
        *,
        shadow_score: Optional[float],
        shadow_feature_source: str,
        ready_age_sec: float,
    ) -> None:
        return entry_scanner.log_ev3m_reconciliation(
            self, token,
            shadow_score=shadow_score,
            shadow_feature_source=shadow_feature_source,
            ready_age_sec=ready_age_sec,
        )

    async def _collect_shadow_features(self, token: GraduatedToken) -> Tuple[Optional[Dict[str, float]], str]:
        return await entry_scanner.collect_shadow_features(self, token)

    async def _run_shadow_rule_first(self):
        return await entry_scanner.run_shadow_rule_first(self)

    def _shadow_ev3m_band_names(self) -> List[str]:
        return entry_scanner.shadow_ev3m_band_names(self)

    def _ev3m_entry_offset_sec(self, token: GraduatedToken, delay_sec: int) -> float:
        return entry_scanner.ev3m_entry_offset_sec(token, delay_sec)

    def _shadow_ev3m_entry_offset_sec(self, token: GraduatedToken) -> float:
        return entry_scanner.shadow_ev3m_entry_offset_sec(self, token)

    async def _collect_ev3m_features(
        self,
        token: GraduatedToken,
        delay_sec: int,
        model: Optional[ConfirmedContinuationEVShadowModel],
    ) -> Tuple[Optional[Dict[str, float]], str]:
        if model is None:
            return None, "no_ev3m_model"
        if self.kline_builder is None:
            return None, "no_onchain_kline_builder"
        try:
            await self.kline_builder.poll_swaps(token.mint_address)
        except Exception:
            pass

        graduation_time = float(token.graduation_time)
        anchor_ts = int(math.floor(graduation_time / 60.0) * 60)
        anchor_offset = graduation_time - float(anchor_ts)
        entry_rel_minute = int(math.ceil((delay_sec + anchor_offset) / 60.0))
        n_bars_needed = max(3, entry_rel_minute + 1)
        kline_bars, shifted = self.kline_builder.build_kline(
            mint=token.mint_address,
            start_ts=anchor_ts,
            n_bars=n_bars_needed,
            resolution=60,
            sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
        )
        if kline_bars and not shifted:
            features = compute_confirmed_entry_live_features(
                kline_bars=kline_bars,
                graduation_time=graduation_time,
                delay_sec=delay_sec,
            )
            if features is not None:
                return features, "on-chain_confirmed_entry_proxy"

        # Fallback to GMGN 1m kline when the on-chain builder does not yet have a
        # stable aligned window. This is still a live proxy, but it avoids
        # rejecting otherwise scoreable tokens purely due to swap-collection lag.
        gmgn_bars = await self._fetch_gmgn_obs_kline(
            token.mint_address,
            token.symbol,
            graduation_time,
        )
        if gmgn_bars:
            features = compute_confirmed_entry_live_features(
                kline_bars=gmgn_bars,
                graduation_time=graduation_time,
                delay_sec=delay_sec,
            )
            if features is not None:
                return features, "gmgn_confirmed_entry_proxy"

        if not kline_bars:
            return None, "insufficient_onchain_bars"
        if shifted:
            return None, "shifted_onchain_window"
        return None, "confirmed_entry_features_unavailable"

    async def _collect_shadow_ev3m_features(self, token: GraduatedToken) -> Tuple[Optional[Dict[str, float]], str]:
        return await entry_scanner.collect_shadow_ev3m_features(self, token)

    async def _run_shadow_ev3m(self):
        return await entry_scanner.run_shadow_ev3m(self)

    # NOTE: _collect_ev3m_features stays on the controller (shared with update_processed_data:3279)

    def _shadow_super_winner_event_band_specs(self) -> List[Tuple[str, str]]:
        return entry_scanner.shadow_super_winner_event_band_specs(self)

    def _shadow_super_winner_event_n_bars_needed(self, token: GraduatedToken) -> int:
        return entry_scanner.shadow_super_winner_event_n_bars_needed(self, token)

    async def _collect_shadow_super_winner_event_view(
        self,
        token: GraduatedToken,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        return await entry_scanner.collect_shadow_super_winner_event_view(self, token)

    async def _run_shadow_super_winner_event(self):
        return await entry_scanner.run_shadow_super_winner_event(self)

    def _shadow_stage2_rule_name(self) -> str:
        return entry_scanner.shadow_stage2_rule_name(self)

    def _shadow_stage2_rule_params(self) -> dict:
        return entry_scanner.shadow_stage2_rule_params(self)

    def _shadow_stage2_pass(self, stage2_rule_name: str, range_1to3m: float, drawdown_1to3m: float) -> bool:
        return entry_scanner.shadow_stage2_pass(self, stage2_rule_name, range_1to3m, drawdown_1to3m)

    async def _run_shadow_vshape(self):
        return await entry_scanner.run_shadow_vshape(self)

    async def _run_shadow_event(self):
        return await entry_scanner.run_shadow_event(self)

    async def _run_shadow_big_winner(self):
        return await entry_scanner.run_shadow_big_winner(self)

    async def _rug_event_watcher(self):
        """P5-8 delegate → ms/monitor/phantom_guard.run_rug_event_watcher.

        Phase 5-8 (2026-06-01): body extracted verbatim to
        ms/monitor/phantom_guard.py::run_rug_event_watcher.
        self→ctrl substitution only. Launch site in on_start UNCHANGED.
        """
        await _run_rug_event_watcher_fn(self)

    # ──────────────────────────────────────────
    # Phase 22.S.P1 — gRPC price event watcher (fast SL/EC trigger)
    # ──────────────────────────────────────────

    async def _price_event_watcher(self):
        """P5-8 delegate → ms/monitor/grpc_exit_watcher.run_price_event_watcher.

        Phase 5-8 (2026-06-01): body extracted verbatim to
        ms/monitor/grpc_exit_watcher.py::run_price_event_watcher.
        self→ctrl substitution only. Launch site in on_start UNCHANGED.
        """
        await _run_price_event_watcher_fn(self)

    # ──────────────────────────────────────────
    # 14y shadow-exit-warn (dual-model drop + rug)
    # ──────────────────────────────────────────

    def _get_exit_models_mod(self):
        # Phase 4f-1: delegate to exit_model_engine (SYNC)
        return exit_model_engine.get_exit_models_mod(self)

    def _run_shadow_exit_warn_tick(self, pos, now: float) -> None:
        # Phase 4f-1: delegate to exit_model_engine (SYNC — NO await, see R2)
        return exit_model_engine.run_shadow_exit_warn_tick(self, pos, now)

    async def _shadow_exit_warn_resolver_loop(self):
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow
        return await sw_v2_shadow.shadow_exit_warn_resolver_loop(self)

    def _compute_shadow_exit_warn_outcome(self, mint: str, t: float) -> Optional[Dict]:
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow
        return sw_v2_shadow.compute_shadow_exit_warn_outcome(self, mint, t)

    # ──────────────────────────────────────────
    # 19c sw_v2 (F1 entry) shadow scoring
    # ──────────────────────────────────────────

    def _get_sw_v2_mod(self):
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow
        return sw_v2_shadow.get_sw_v2_mod(self)

    def _run_sw_v2_shadow_once(self, token, now: float) -> None:
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow (SYNC — NO await, see R2)
        return sw_v2_shadow.run_sw_v2_shadow_once(self, token, now)

    async def _run_pregrad_factors_shadow_once(self, token, now: float) -> None:
        # 2026-06-28: delegate to pregrad_factors_shadow (ASYNC — RPC backfill).
        # OBSERVE-ONLY; whole body is try/except-guarded inside the runner.
        return await pregrad_factors_shadow.run_pregrad_factors_shadow_once(
            self, token, now)

    async def _shadow_sw_v2_resolver_loop(self):
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow
        return await sw_v2_shadow.shadow_sw_v2_resolver_loop(self)

    def _compute_shadow_sw_v2_outcome(
        self, *, mint: str, entry_time: float,
        entry_price_sol: Optional[float], horizon_sec: int,
    ) -> Optional[Dict]:
        # Phase 4h (2026-06-01): delegate to sw_v2_shadow
        return sw_v2_shadow.compute_shadow_sw_v2_outcome(
            self, mint=mint, entry_time=entry_time,
            entry_price_sol=entry_price_sol, horizon_sec=horizon_sec)

    async def _run_shadow_event_invariant(self):
        return await entry_scanner.run_shadow_event_invariant(self)

    async def _flush_expired_observations(self):
        """Fetch GMGN 1m kline + token_info + security for expired observations."""
        now = time.time()
        expired = [m for m, obs in self._observation_pool.items() if now >= obs.expire_time]
        for mint in expired:
            obs = self._observation_pool.pop(mint)
            token = obs.token
            try:
                kline_60m = await self._fetch_gmgn_obs_kline(
                    token.mint_address, token.symbol, token.graduation_time)
                _info60_t = time.time()
                info_60m = await self._fetch_gmgn_token_info(token.mint_address, token.symbol)
                _sec60_t = time.time()
                security = await self._fetch_gmgn_token_security(token.mint_address, token.symbol)
                # T2.1 observability: build 60m completion meta.
                _complete_meta = {}
                if info_60m is not None:
                    _complete_meta["60m"] = {"collected_at": _info60_t, "fallback_used": False, "source": "gmgn"}
                if security is not None:
                    _complete_meta["security"] = {"collected_at": _sec60_t, "fallback_used": False, "source": "gmgn"}
                trade_pnl = None
                trade_exit = None
                for rec in reversed(self._trade_log):
                    if rec.mint_address == mint:
                        trade_pnl = rec.pnl_usd
                        trade_exit = rec.exit_reason
                        break
                self.db.complete_observation(
                    obs.obs_id, kline_30m=kline_60m,
                    trade_pnl_usd=trade_pnl, trade_exit_reason=trade_exit,
                    gmgn_info_60m=info_60m, gmgn_security=security,
                    gmgn_collection_meta=_complete_meta if _complete_meta else None)
                n_bars = len(kline_60m) if kline_60m else 0

                # Update trader records: determine if token was a "winner"
                # Winner = positive 3m net return from T+90s entry
                is_winner = False
                if kline_60m and len(kline_60m) >= 4:
                    try:
                        entry_bar = next(
                            (b for b in kline_60m
                             if int(b.get("time", 0)) / 1000 - token.graduation_time >= 90),
                            None)
                        exit_idx = min(len(kline_60m) - 1,
                                       (kline_60m.index(entry_bar) + 3) if entry_bar else 3)
                        if entry_bar:
                            ep = float(entry_bar.get("open", 0))
                            xp = float(kline_60m[exit_idx].get("close", 0))
                            if ep > 0 and xp > 0:
                                is_winner = (xp / ep - 1) > 0.0514  # positive after cost
                    except Exception:
                        pass
                n_traders_updated = self.db.update_trader_records(mint, is_winner)

                logger.info(f"OBS: {token.symbol} completed "
                            f"(id={obs.obs_id}, {n_bars} bars, "
                            f"info={'Y' if info_60m else 'N'}, "
                            f"sec={'Y' if security else 'N'}, "
                            f"traded={'yes' if trade_pnl is not None else 'no'}, "
                            f"traders_updated={n_traders_updated}, "
                            f"winner={'Y' if is_winner else 'N'})")
            except Exception as e:
                logger.warning(f"OBS: failed to complete observation for {token.symbol}: {e}")

    async def _collect_observation_snapshots(self):
        """Fetch GMGN snapshots at scheduled offsets from graduation time.

        SNAPSHOT_SCHEDULE contains entries like (30, "gmgn_info_t0") and
        (900, "gmgn_info_15m"). For each scheduled snapshot:
        - Fetch token_info from GMGN and save it to the named column.
        - For t0 snapshots specifically, also fetch security data into
          gmgn_security_t0 (the "birth certificate" — captures dev history,
          honeypot/burn/renounce state, holder structure at the very start).
        """
        now = time.time()
        for mint, obs in list(self._observation_pool.items()):
            grad_t = obs.token.graduation_time
            elapsed = now - grad_t
            for offset_sec, col_name in SNAPSHOT_SCHEDULE:
                if col_name in obs.snapshots_done:
                    # Special case: if gmgn_info_t0 was pre-written at M1 time
                    # (via _add_to_observation_pool early-capture), we still
                    # need to fire the security + dexscr snapshots for this
                    # token. Handle that path without re-fetching info.
                    if (col_name == "gmgn_info_t0"
                            and "gmgn_security_t0_done" not in obs.snapshots_done
                            and elapsed >= offset_sec):
                        try:
                            _sec_early_t = time.time()
                            sec_early = await self._fetch_gmgn_token_security(
                                obs.token.mint_address, obs.token.symbol)
                            if sec_early:
                                _sec_early_meta = {"security": {"collected_at": _sec_early_t, "fallback_used": False, "source": "gmgn"}}
                                self.db.update_observation_snapshot(
                                    obs.obs_id, "gmgn_security_t0", sec_early,
                                    gmgn_collection_meta=_sec_early_meta)
                            if self.config.dexscreener_collect_enabled:
                                try:
                                    dx_early = await self._fetch_dexscreener_token_info(
                                        obs.token.mint_address, obs.token.symbol)
                                    if dx_early:
                                        self.db.update_observation_snapshot(
                                            obs.obs_id, "dexscr_info_t0", dx_early)
                                except Exception as e:
                                    logger.debug(
                                        f"OBS: {obs.token.symbol} early dexscr fetch failed: {e}"
                                    )
                            obs.snapshots_done.add("gmgn_security_t0_done")
                            logger.info(
                                f"OBS: {obs.token.symbol} sec+dexscr snapshot "
                                f"(info_t0 was pre-captured at M1)"
                            )
                        except Exception as e:
                            logger.debug(
                                f"OBS: {obs.token.symbol} early sec/dexscr fetch failed: {e}"
                            )
                            obs.snapshots_done.add("gmgn_security_t0_done")
                    continue
                if elapsed < offset_sec:
                    break
                try:
                    _fetch_t = time.time()
                    info = await self._fetch_gmgn_token_info(
                        obs.token.mint_address, obs.token.symbol)
                    # T2.1 observability: derive checkpoint label from column name.
                    # col_name is like "gmgn_info_15m" → key "15m"; "gmgn_info_t0" → "t0".
                    _ck = col_name.replace("gmgn_info_", "") or col_name
                    _info_meta = {_ck: {"collected_at": _fetch_t, "fallback_used": False, "source": "gmgn"}}
                    if info:
                        self.db.update_observation_snapshot(obs.obs_id, col_name, info,
                                                            gmgn_collection_meta=_info_meta)
                    obs.snapshots_done.add(col_name)

                    # For t0: also fetch security snapshot immediately.
                    # This gives us honeypot/burn/renounce/tax at the moment
                    # the token first appears — before any trading or
                    # manipulation changes these flags.
                    sec = None
                    if col_name == "gmgn_info_t0":
                        obs.snapshots_done.add("gmgn_security_t0_done")
                        try:
                            _sec_t = time.time()
                            sec = await self._fetch_gmgn_token_security(
                                obs.token.mint_address, obs.token.symbol)
                            _sec_meta = {"security": {"collected_at": _sec_t, "fallback_used": False, "source": "gmgn"}}
                            if sec:
                                self.db.update_observation_snapshot(
                                    obs.obs_id, "gmgn_security_t0", sec,
                                    gmgn_collection_meta=_sec_meta)
                            sec_status = "OK" if sec else "empty"
                        except Exception as e:
                            sec_status = f"failed: {e}"
                            logger.debug(f"OBS: {obs.token.symbol} security_t0: {e}")
                        # DexScreener snapshot (orthogonal signal, saved for
                        # future rug_filter_v2 retraining). Non-blocking.
                        dexscr_status = "skipped"
                        if self.config.dexscreener_collect_enabled:
                            try:
                                dx = await self._fetch_dexscreener_token_info(
                                    obs.token.mint_address, obs.token.symbol)
                                if dx:
                                    self.db.update_observation_snapshot(
                                        obs.obs_id, "dexscr_info_t0", dx)
                                    dexscr_status = (
                                        "OK" if dx.get("pair_found") else "no_pair")
                                else:
                                    dexscr_status = "empty"
                            except Exception as e:
                                dexscr_status = f"failed: {e}"
                        label = (f"t0 (info={'OK' if info else 'empty'}, "
                                 f"sec={sec_status}, dexscr={dexscr_status})")
                    else:
                        label = col_name.replace("gmgn_info_", "")
                        label = f"{label} ({'OK' if info else 'empty'})"

                    logger.info(f"OBS: {obs.token.symbol} snapshot @{label}")

                    # Rug filter shadow scoring: runs once per token at t0
                    # snapshot. Logs to shadow_rug_evals. Never affects
                    # trading decisions — this is observational data only.
                    if (col_name == "gmgn_info_t0"
                            and self.rug_filter_model is not None
                            and self.config.shadow_rug_enabled
                            and info is not None):
                        try:
                            feats = flatten_gmgn_features(info, sec)
                            score = self.rug_filter_model.predict_score(feats)
                            band = self.config.shadow_rug_cutoff_band
                            cutoff_val = self.rug_filter_model.cutoffs.get(band, 1.0)
                            would_reject = score >= cutoff_val
                            self.db.record_shadow_rug_eval(
                                mint_address=obs.token.mint_address,
                                symbol=obs.token.symbol,
                                graduation_time=obs.token.graduation_time,
                                snapshot_delay_sec=now - obs.token.graduation_time,
                                score=score, cutoff_band=band,
                                cutoff_value=cutoff_val, would_reject=would_reject,
                                model_version=self.rug_filter_model.version,
                                features=feats)
                            logger.info(
                                f"RUG-SHADOW: {obs.token.symbol} "
                                f"score={score:.3f} cutoff_{band}={cutoff_val:.3f} "
                                f"{'REJECT' if would_reject else 'PASS'}")
                        except Exception as e:
                            logger.debug(
                                f"RUG-SHADOW: {obs.token.symbol} scoring failed: {e}")
                except Exception as e:
                    logger.debug(f"OBS: {obs.token.symbol} snapshot @{col_name} failed: {e}")
                    obs.snapshots_done.add(col_name)

    async def _run_shadow_rug_filter_v4(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_filter_v4(self)

    async def _run_shadow_rug_filter_v4_3(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_filter_v4_3(self)

    async def _run_shadow_rug_filter_v4_4(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_filter_v4_4(self)

    async def _run_shadow_rug_filter_v3b(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_filter_v3b(self)

    async def _run_shadow_rug_funder_v1(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_funder_v1(self)

    async def _run_shadow_rug_filter_v4_2(self):
        # Phase 4e-1: extracted to ms/shadow/rug_filter_runner.py
        return await rug_filter_runner.run_shadow_rug_filter_v4_2(self)

    async def _recover_observations(self):
        """Reload all pending observations from DB into memory, then sweep expired ones."""
        pending = self.db.load_pending_observations()
        if not pending:
            return
        now = time.time()
        recovered = 0
        for p in pending:
            if p["mint"] in self._observation_pool:
                continue
            token = GraduatedToken(
                mint_address=p["mint"], symbol=p["symbol"], name=p["symbol"],
                decimals=6, graduation_time=p["graduation_time"],
                liquidity_usd=p.get("liquidity_usd", 0) or 0,
                price_usd=p.get("price_usd", 0) or 0,
                pool_address=p.get("pool"), source=p.get("source", "chainstack"),
            )
            expire = p["graduation_time"] + 62 * 60
            done = set()
            if p.get("has_15m"):
                done.add("gmgn_info_15m")
            if p.get("has_30m"):
                done.add("gmgn_info_30m")
            if p.get("has_45m"):
                done.add("gmgn_info_45m")
            entry = ObservationEntry(
                token=token, obs_id=p["id"], expire_time=expire,
                snapshots_done=done)
            self._observation_pool[p["mint"]] = entry
            recovered += 1
            # Recovery path bypasses _add_to_observation_pool — enqueue
            # creator_resolver here too so reloaded mints get cap_hit
            # resolved for the v3b shadow eval at +600s. Idempotent.
            if self.creator_resolver is not None:
                try:
                    self.creator_resolver.enqueue(p["mint"])
                except Exception as e:
                    logger.debug(f"creator_resolver.enqueue recovery failed: {e}")
            # If creator already resolved (previous run cached), chain-enqueue
            # to funder_resolver here too (the on_creator_resolved callback
            # only fires on NEW resolves, not on cache hits).
            # Phase 1A audit follow-up 2026-05-19.
            if self.funder_resolver is not None:
                try:
                    rows = self.db._fetchall(
                        "SELECT creator FROM creator_cap_hit_cache "
                        "WHERE mint_address = ? AND cap_hit = 0 "
                        "AND creator IS NOT NULL LIMIT 1", (p["mint"],))
                    if rows and rows[0][0]:
                        self.funder_resolver.enqueue(rows[0][0])
                except Exception as e:
                    logger.debug(f"funder_resolver.enqueue recovery failed: {e}")
        if recovered:
            logger.info(f"OBS: recovered {recovered} pending observations from DB")
        # Now sweep any that are already expired
        await self._sweep_orphan_observations()

    async def _sweep_orphan_observations(self):
        """Flush all DB orphans that were lost from memory (e.g. after bot restart)."""
        orphans = self.db.load_orphan_observations()
        if not orphans:
            return
        logger.info(f"OBS: sweeping {len(orphans)} orphan observations from DB")
        flushed = 0
        for orph in orphans:
            if orph["mint"] in self._observation_pool:
                continue
            try:
                kline = await self._fetch_gmgn_obs_kline(
                    orph["mint"], orph["symbol"], orph["graduation_time"])
                _orph_info_t = time.time()
                info_60m = await self._fetch_gmgn_token_info(orph["mint"], orph["symbol"])
                _orph_sec_t = time.time()
                security = await self._fetch_gmgn_token_security(orph["mint"], orph["symbol"])
                # T2.1 observability: build orphan-sweep 60m completion meta.
                _orph_meta = {}
                if info_60m is not None:
                    _orph_meta["60m"] = {"collected_at": _orph_info_t, "fallback_used": False, "source": "gmgn"}
                if security is not None:
                    _orph_meta["security"] = {"collected_at": _orph_sec_t, "fallback_used": False, "source": "gmgn"}
                trade_pnl = None
                trade_exit = None
                for rec in reversed(self._trade_log):
                    if rec.mint_address == orph["mint"]:
                        trade_pnl = rec.pnl_usd
                        trade_exit = rec.exit_reason
                        break
                self.db.complete_observation(
                    orph["id"], kline_30m=kline,
                    trade_pnl_usd=trade_pnl, trade_exit_reason=trade_exit,
                    gmgn_info_60m=info_60m, gmgn_security=security,
                    gmgn_collection_meta=_orph_meta if _orph_meta else None)
                n_bars = len(kline) if kline else 0
                flushed += 1
                logger.info(f"OBS: orphan {orph['symbol']} completed "
                            f"(id={orph['id']}, {n_bars} bars, "
                            f"info={'Y' if info_60m else 'N'}, sec={'Y' if security else 'N'})")
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.warning(f"OBS: orphan {orph['symbol']} flush failed: {e}")
        logger.info(f"OBS: orphan sweep done — {flushed}/{len(orphans)} flushed")

    async def _fetch_gmgn_obs_kline(self, mint_address: str, symbol: str,
                                     graduation_time: float) -> Optional[list]:
        """Fetch 60-min 1m kline from GMGN for research observation."""
        if not self.signal:
            return None
        try:
            client = await self.signal._get_client()
            grad_ts_ms = int(graduation_time) * 1000
            window_ms = 61 * 60_000
            params = {
                **self.signal._auth_params(),
                "chain": "sol",
                "address": mint_address,
                "resolution": "1m",
                "from": str(grad_ts_ms),
                "to": str(grad_ts_ms + window_ms),
            }
            resp = await client.get(
                f"{self.signal.base_url}{self.signal.KLINE_PATH}",
                params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.warning(f"OBS: GMGN kline error for {symbol}: {data}")
                return None
            bars = data.get("data", {}).get("list", [])
            return bars if bars else None
        except Exception as e:
            logger.warning(f"OBS: GMGN kline fetch failed for {symbol}: {e}")
            return None

    async def _fetch_gmgn_token_info(self, mint_address: str, symbol: str) -> Optional[dict]:
        """Fetch GMGN /v1/token/info for research snapshot."""
        if not self.discovery:
            return None
        try:
            client = await self.discovery._get_gmgn_client()
            params = {**self.discovery._auth_params(), "chain": "sol", "address": mint_address}
            resp = await client.get(
                f"{self.discovery.base_url}{self.discovery.TOKEN_INFO_PATH}",
                params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return None
            return data.get("data")
        except Exception as e:
            logger.debug(f"OBS: token_info fetch failed for {symbol}: {e}")
            return None

    async def _fetch_gmgn_token_security(self, mint_address: str, symbol: str) -> Optional[dict]:
        """Fetch GMGN /v1/token/security for research snapshot."""
        if not self.discovery:
            return None
        try:
            client = await self.discovery._get_gmgn_client()
            params = {**self.discovery._auth_params(), "chain": "sol", "address": mint_address}
            resp = await client.get(
                f"{self.discovery.base_url}{self.discovery.TOKEN_SECURITY_PATH}",
                params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return None
            return data.get("data")
        except Exception as e:
            logger.debug(f"OBS: token_security fetch failed for {symbol}: {e}")
            return None

    async def _fetch_dexscreener_token_info(self, mint_address: str,
                                             symbol: str) -> Optional[dict]:
        """Fetch DexScreener /latest/dex/tokens/{mint} for research snapshot.

        Free public API, 300 req/min limit. At ~100 graduations/day the
        bot generates ~0.07 req/sec — well within limits.

        Returns the first Solana pair (usually PumpSwap for graduated
        tokens) + merged top-level info (boosts, socials, websites).
        Returns None on network/decode errors; never raises.
        """
        try:
            client = await self.trader._get_client()
            url = f"https://api.dexscreener.com/latest/dex/tokens/{mint_address}"
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            pairs = data.get("pairs") or []
            # Prefer Solana pair with highest liquidity
            sol_pairs = [p for p in pairs if p.get("chainId") == "solana"]
            if not sol_pairs:
                return {"pair_found": False}
            best = max(sol_pairs,
                        key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0)
            # Flatten to a snapshot dict; keep only numeric/bool/string scalars
            # and a few nested dicts we know how to consume.
            return {
                "pair_found": True,
                "dex_id": best.get("dexId"),
                "price_usd": best.get("priceUsd"),
                "price_native": best.get("priceNative"),
                "pair_created_at": best.get("pairCreatedAt"),
                "fdv": best.get("fdv"),
                "market_cap": best.get("marketCap"),
                "liquidity": best.get("liquidity") or {},
                "volume": best.get("volume") or {},
                "price_change": best.get("priceChange") or {},
                "txns": best.get("txns") or {},
                "info": best.get("info") or {},
                "boosts": best.get("boosts") or {},
                "url": best.get("url"),
            }
        except Exception as e:
            logger.debug(f"OBS: dexscreener fetch failed for {symbol}: {e}")
            return None

    def _apply_m2_safety_filters(self, token: "GraduatedToken",
                                   features: Optional[Dict[str, float]],
                                   kline_source: str,
                                   *, source: str = "vshape"
                                   ) -> Tuple[bool, Optional[str]]:
        """Phase 16.3 Step 3 — shared M2 hard-reject safety chain.

        Used by `_evaluate_token` (V-shape entry path) AND by
        `_run_shadow_big_winner` LIVE PASS branch. Ensures every entry
        respects basic crash/pump/liquidity guards.

        Filters (same order as `_evaluate_token`):
          1. low_activity (trade_count + stale_pct + volume floor)
          2. max_5min_return (anti-pump)
          3. max_vol_liq_ratio (anti-rug)
          4. max_entry_return_floor (extreme crash)
          5. max_cum_drawdown_at_entry
          6. lookback_return floor (last-3-bar freefall)

        `source` kwarg selects threshold profile:
          - "vshape" (default): V-shape's tight thresholds
          - "big_winner": BigWinner's wider thresholds
            (calibrated 2026-04-28 from 14d alpha-impact analysis;
            recovers ~$155/14d that V-shape thresholds were killing).

        Returns (passes: bool, reject_reason: Optional[str]).
        Side effects (logging + DB) are caller's responsibility.
        """
        if features is None:
            return False, "insufficient_kline"

        # Per-source threshold selection.
        # Phase 25g-fix-2 (2026-05-03): defensive _cfg_value reads —
        # canonical Optional[float] fields can be None at runtime even
        # after _normalize_model_config_aliases.
        if self._is_big_winner_entry_source(source):
            max_5m = float(self._cfg_value(
                self.config, "big_winner_max_5min_return",
                "big_winner_v1_max_5min_return", default=2.00))
            max_vl = float(self._cfg_value(
                self.config, "big_winner_max_vol_liq_ratio",
                "big_winner_v1_max_vol_liq_ratio", default=15.0))
            max_cret_floor = float(self._cfg_value(
                self.config, "big_winner_max_entry_return_floor",
                "big_winner_v1_max_entry_return_floor", default=-0.85))
            max_cdd = float(self._cfg_value(
                self.config, "big_winner_max_cum_drawdown",
                "big_winner_v1_max_cum_drawdown", default=0.95))
            lb_floor = float(self._cfg_value(
                self.config, "big_winner_lookback_return_floor",
                "big_winner_v1_lookback_return_floor", default=-0.85))
        else:  # "vshape" or any other → use V-shape (global) thresholds
            max_5m = float(self.config.max_5min_return)
            max_vl = float(self.config.max_vol_liq_ratio)
            max_cret_floor = float(self.config.max_entry_return_floor)
            max_cdd = float(self.config.max_cum_drawdown_at_entry)
            lb_floor = -0.50  # V-shape hardcoded historical floor

        if "5min_trade_count" in features:
            trade_count = features.get("5min_trade_count", 0)
            stale_pct = features.get("5min_stale_pct", 0)
            vol_usd = features.get("5min_volume_usd", 0)
        else:
            trade_count = features.get("cum_trades_t0_to_entry", 0)
            vol_usd = features.get("cum_volume_t0_to_entry", 0)
            stale_pct = 0
        trade_missing = isinstance(trade_count, float) and math.isnan(trade_count)
        if trade_missing:
            trade_count = 0

        if kline_source == "on-chain":
            trade_count_ok = trade_count >= 30
            min_vol = 1000
        else:
            trade_count_ok = True
            min_vol = 3000

        if not trade_count_ok or stale_pct > 0.6 or vol_usd < min_vol:
            tc_str = "NaN" if trade_missing else f"{trade_count:.0f}"
            return False, (f"low_activity trades={tc_str} stale={stale_pct:.0%} "
                           f"vol=${vol_usd:.0f} src={kline_source}")

        ret_5m = features.get("5min_return", features.get("cum_return_t0_to_entry", 0))
        if ret_5m > max_5m:
            return False, f"5min_return={ret_5m:.3f}>{max_5m}"

        vol_5m = features.get("5min_volume_usd", features.get("cum_volume_t0_to_entry", 0))
        liq = token.liquidity_usd if token.liquidity_usd > 0 else 1
        vol_liq_ratio = vol_5m / liq
        if vol_liq_ratio > max_vl:
            return False, f"vol_liq_ratio={vol_liq_ratio:.2f}>{max_vl}"

        cum_ret = features.get("cum_return_t0_to_entry",
                                features.get("5min_return", 0))
        cum_dd = features.get("cum_drawdown_t0_to_entry", 0)
        lb_ret = features.get("lookback_return", 0)

        if cum_ret < max_cret_floor:
            return False, f"cum_return={cum_ret:.3f}<{max_cret_floor}"
        if abs(cum_dd) > max_cdd:
            return False, f"cum_drawdown={cum_dd:.3f}>{max_cdd}"
        if lb_ret < lb_floor:
            return False, f"lookback_return={lb_ret:.3f}<{lb_floor}"

        return True, None

    async def _build_m2_kline_for_safety(self, token: "GraduatedToken"
                                           ) -> Tuple[Optional[List[Dict]], str,
                                                       Optional[Dict[str, float]]]:
        """Phase 16.3 Step 3 — shared M2 kline + feature build for safety check.

        Returns (kline_bars, kline_source, features). features may be None if
        insufficient data; caller should treat that as "fail safety check".
        """
        kline_bars = None
        kline_shifted = False
        kline_source = "gmgn"
        if self.kline_builder and self.kline_builder.is_registered(token.mint_address):
            # 2026-06-03: skip the redundant RPC poll when gRPC Geyser is feeding the
            # same swaps into the builder (mirrors main-loop guard ~:2716). build_kline
            # reads the gRPC-fed buffer; saves 1-3 getSignaturesForAddress RTs on the M2
            # build/decision path. Falls back to poll when gRPC disconnected.
            if not (self._grpc_stream and self._grpc_stream.connected):
                try:
                    await self.kline_builder.poll_swaps(token.mint_address)
                except Exception:
                    pass
            n_bars_needed = max(3, self.config.feature_delay_sec // 60)
            kline_bars, kline_shifted = self.kline_builder.build_kline(
                mint=token.mint_address,
                start_ts=int(token.graduation_time),
                n_bars=n_bars_needed, resolution=60,
                sol_price_usd=self._sol_price_usd
                              if self._sol_price_usd > 0 else 80.0,
            )
            if kline_bars:
                kline_source = "on-chain"
        features = await self.signal.collect_features(
            token, kline_bars=kline_bars, kline_shifted=kline_shifted)
        return kline_bars, kline_source, features

    async def _evaluate_token(self, token: GraduatedToken) -> Optional[TradeCandidate]:
        """M2: collect features + predict survival."""
        # Check stop-loss blacklist — skip tokens that recently stopped out
        bl_until = self._stoploss_blacklist.get(token.mint_address)
        if bl_until and time.time() < bl_until:
            remaining = bl_until - time.time()
            logger.info(f"M2: {token.symbol} SKIP — stop-loss blacklisted ({remaining:.0f}s remaining)")
            self.db.record_discovery(token, None, False, "stoploss_blacklisted")
            return None

        if not self.signal:
            logger.warning(f"M2: no model loaded, skipping {token.symbol}")
            self.db.record_discovery(token, None, False, "no_model")
            self._flush_swaps_to_db(token.mint_address)
            return None
        try:
            # Try on-chain kline first (built from pool swap data)
            kline_bars = None
            kline_shifted = False
            kline_source = "gmgn"
            if self.kline_builder and self.kline_builder.is_registered(token.mint_address):
                # Final poll to get latest swaps before building kline.
                # 2026-06-03: skip when gRPC is feeding the buffer (mirrors main-loop
                # guard ~:2716); saves a getSignaturesForAddress RT on the M2 build path.
                if not (self._grpc_stream and self._grpc_stream.connected):
                    try:
                        await self.kline_builder.poll_swaps(token.mint_address)
                    except Exception:
                        pass
                # Use graduation_time as kline start (swaps collected since graduation).
                # (Previously trending tokens used most-recent; trending feature removed.)
                n_bars_needed = max(3, self.config.feature_delay_sec // 60)
                kline_start = int(token.graduation_time)
                kline_bars, kline_shifted = self.kline_builder.build_kline(
                    mint=token.mint_address,
                    start_ts=kline_start,
                    n_bars=n_bars_needed, resolution=60,
                    sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
                )
                if kline_bars:
                    total_swaps = self.kline_builder.get_swap_count(token.mint_address)
                    pool_count = self.kline_builder.get_pool_count(token.mint_address)
                    kline_source = "on-chain"
                    logger.info(f"M2: {token.symbol} using on-chain kline "
                                f"({total_swaps} swaps, {len(kline_bars)} bars, {pool_count} pool(s), "
                                f"shifted={kline_shifted})")
                else:
                    pool_count = self.kline_builder.get_pool_count(token.mint_address)
                    n_swaps = self.kline_builder.get_swap_count(token.mint_address)
                    swaps = self.kline_builder.get_swaps(token.mint_address)
                    if swaps:
                        first_ts = swaps[0].timestamp
                        last_ts = swaps[-1].timestamp
                        span = last_ts - first_ts
                        # Coverage relative to required window
                        grad_ts = int(token.graduation_time)
                        gap_at_start = first_ts - grad_ts  # how far first swap is from grad
                        logger.warning(f"M2: {token.symbol} on-chain kline insufficient "
                                       f"({n_swaps} swaps, {pool_count} pool(s), "
                                       f"span={span:.0f}s, gap_at_start={gap_at_start:.0f}s, "
                                       f"need {n_bars_needed} bars from grad_time)")
                    else:
                        logger.warning(f"M2: {token.symbol} on-chain kline insufficient "
                                       f"(0 swaps, {pool_count} pool(s))")

            features = await self.signal.collect_features(
                token, kline_bars=kline_bars, kline_shifted=kline_shifted)
            # Cache last swap price (used by M3 preflight double-check)
            cached_swap_price_sol = 0.0
            if self.kline_builder and self.kline_builder.is_registered(token.mint_address):
                _swaps = self.kline_builder.get_swaps(token.mint_address)
                if _swaps:
                    cached_swap_price_sol = float(_swaps[-1].price_sol)
            if features is None:
                logger.info(f"M2: insufficient kline data for {token.symbol} "
                            f"({token.mint_address[:12]}...) [source={kline_source}]")
                self.db.record_discovery(token, None, False, "insufficient_kline")
                kline_6m_serialized = [b.__dict__ if hasattr(b, '__dict__') else b
                                       for b in kline_bars] if kline_bars else None
                self._add_to_observation_pool(
                    token, None, False, "insufficient_kline",
                    kline_6m=kline_6m_serialized)
                return None

            # Phase 16.3 Step 3 — shared M2 safety chain (DRY refactor).
            # All hard-reject filters now live in `_apply_m2_safety_filters`,
            # called by both V-shape entry (here) and big_winner LIVE entry.
            # Logging + observation_pool side effects remain caller-side.
            _kline_6m_pre = ([b.__dict__ if hasattr(b, '__dict__') else b
                              for b in kline_bars] if kline_bars else None)
            safety_ok, safety_reason = self._apply_m2_safety_filters(
                token, features, kline_source)
            # Special-case: low_activity needs detailed log identical to legacy.
            # The shared helper returns reasons keyed by name; rich logging
            # below preserves the original V-shape diagnostic detail.
            if not safety_ok and safety_reason and safety_reason.startswith("low_activity"):
                logger.info(f"M2: {token.symbol} REJECT — {safety_reason}")
                self.db.record_discovery(token, None, False, safety_reason, features=features)
                self._add_to_observation_pool(
                    token, None, False, safety_reason,
                    features=features, kline_6m=_kline_6m_pre)
                return None

            # P10 outlier guards REMOVED (2026-04-07 — Muchamaru postmortem)
            # Empirical analysis on 9965 training samples shows cum_max > 100x tokens
            # actually have HIGHER label_rate (52-80%) and better mean_final than baseline.
            # Rejecting them was a mistake. Trust the model's clip_bounds preprocessing.
            #
            # Reference distribution from training:
            #   cum_max > 100:    n=175, label=52%, mean_final=-2.6%
            #   cum_max > 500:    n=81,  label=61%, mean_final=+6.3%
            #   cum_max > 1000:   n=68,  label=63%, mean_final=+2.2%
            #   cum_max > 10000:  n=10,  label=80%, mean_final=+19.0%
            # Reject pattern: opposite of expected — extreme spikes are MORE profitable.
            #
            # Muchamaru (cum_max=2046x) was rejected by old guard but actually went +71%.

            # Phase 16.3 Step 3 — remaining hard-reject filters (max_5min_return,
            # vol_liq, cum_return_floor, cum_drawdown, lookback_freefall) all
            # consolidated in `_apply_m2_safety_filters`. The shared helper
            # already ran above; if it failed for low_activity we returned.
            # Re-check for the remaining reasons here so the original log
            # detail (with computed values) is preserved.
            _kline_6m = _kline_6m_pre  # reuse serialized form
            ret_5m = features.get("5min_return", features.get("cum_return_t0_to_entry", 0))
            vol_5m = features.get("5min_volume_usd", features.get("cum_volume_t0_to_entry", 0))
            liq = token.liquidity_usd if token.liquidity_usd > 0 else 1
            vol_liq_ratio = vol_5m / liq
            cum_ret = features.get("cum_return_t0_to_entry", features.get("5min_return", 0))
            cum_dd = features.get("cum_drawdown_t0_to_entry", 0)
            lb_ret = features.get("lookback_return", 0)

            if not safety_ok and safety_reason:
                # Build matching detailed log for V-shape parity (the shared
                # helper returned a reason like "5min_return=0.65>0.50";
                # decode back to the legacy log format).
                if safety_reason.startswith("5min_return="):
                    logger.info(f"M2: {token.symbol} REJECT — 5min_return={ret_5m:.1%} > "
                                f"max {self.config.max_5min_return:.0%} (pump-dump risk)")
                elif safety_reason.startswith("vol_liq_ratio="):
                    logger.info(f"M2: {token.symbol} REJECT — vol/liq={vol_liq_ratio:.2f} "
                                f"(vol=${vol_5m:.0f}/liq=${liq:.0f}) > max "
                                f"{self.config.max_vol_liq_ratio:.1f}")
                elif safety_reason.startswith("cum_return="):
                    logger.info(f"M2: {token.symbol} REJECT — cum_return={cum_ret:+.1%} < "
                                f"{self.config.max_entry_return_floor:+.0%} (extreme crash)")
                elif safety_reason.startswith("cum_drawdown="):
                    logger.info(f"M2: {token.symbol} REJECT — cum_drawdown={cum_dd:+.1%}, "
                                f"exceeds {self.config.max_cum_drawdown_at_entry:.0%} "
                                f"(near-total crash)")
                elif safety_reason.startswith("lookback_return="):
                    logger.info(f"M2: {token.symbol} REJECT — lookback_return={lb_ret:+.1%} "
                                f"(>50% drop in last 3 bars)")
                else:
                    logger.info(f"M2: {token.symbol} REJECT — {safety_reason}")
                self.db.record_discovery(token, None, False, safety_reason, features=features)
                self._add_to_observation_pool(token, None, False, safety_reason,
                                                features=features, kline_6m=_kline_6m)
                return None

            logger.info(f"M2: {token.symbol} [chainstack] 5min_return={ret_5m:.1%}, "
                        f"vol/liq={vol_liq_ratio:.2f}")

            # P11 retired: kept loaded only for SignalPipeline feature
            # collection (the score itself is not a decision input). V-shape
            # live gate is the real entry decision. BK case demonstrated
            # P11=0.372 and V-shape=0.531 disagreed; P11 has no live value.
            model_score = self.signal.predict_survival(features)
            threshold = self.signal.threshold
            logger.debug(f"M2: {token.symbol} p11_score={model_score:.3f} "
                         f"threshold={threshold} (retired, not gating)")
            # P10/P11 retired; feature log always emitted in P12+ era.
            _mv = getattr(self.signal, "model_version", "")
            is_p10 = _mv in ("p10", "p11")
            if is_p10:
                logger.info(
                    f"M2: {token.symbol} [features] "
                    f"cum_ret={features.get('cum_return_t0_to_entry', 0):+.1%} "
                    f"cum_vol=${features.get('cum_volume_t0_to_entry', 0):.0f} "
                    f"cum_trades={features.get('cum_trades_t0_to_entry', 0):.0f} "
                    f"cum_dd={features.get('cum_drawdown_t0_to_entry', 0):.1%} "
                    f"lb_ret={features.get('lookback_return', 0):+.1%} "
                    f"lb_vol_accel={features.get('lookback_volume_acceleration', 0):.2f} "
                    f"lb_volatility={features.get('lookback_return_std', 0):.3f}")
            logger.debug(f"M2: {token.symbol} all_features={features}")
            self.db.record_discovery(token, model_score, True, "", features=features)
            # Pass keep_swap_collector=True so the swap collector stays
            # registered through the trade entry / hold / exit window.
            # The collector will be flushed + unregistered at trade exit
            # (or at preflight/buy failure paths if the trade never happens).
            self._add_to_observation_pool(token, model_score, True, "",
                                          features=features, kline_6m=_kline_6m,
                                          keep_swap_collector=True)
            return TradeCandidate(token=token, model_score=model_score, features=features,
                                  last_swap_price_sol=cached_swap_price_sol)
        except Exception as e:
            logger.error(f"M2: evaluation failed for {token.symbol}: {e}", exc_info=True)
            self.db.record_discovery(token, None, False, f"error: {e}")
            self._add_to_observation_pool(token, None, False, f"error: {e}")
        return None

    async def _preflight_buy(self, candidate: TradeCandidate, sol_amount: float) -> Tuple[bool, Optional[dict], str, Dict]:
        """Execution-time guardrails before sending a buy to Gateway.

        Phase 5-4 (2026-06-01): body extracted verbatim to
        ms/execution/preflight.py::preflight_buy.  self→ctrl substitution
        only.  Call site in _execute_buy is UNCHANGED.
        """
        return await preflight_buy(self, candidate, sol_amount)

    # Valid sizing modes — see _compute_position_size for semantics
    _SIZING_MODES_VALID = {"fixed", "slippage_budget"}

    def _compute_position_size(self, candidate: "TradeCandidate") -> Tuple[float, dict]:
        return sizing.compute_position_size(self, candidate)

    async def _execute_buy(self, candidate: TradeCandidate) -> bool:
        """P5-6 delegate → ms/execution/buy.execute_buy.

        Phase 5-6 (2026-06-01): body extracted verbatim to
        ms/execution/buy.py::execute_buy.  self→ctrl substitution
        only.  Call site in control_task is UNCHANGED.
        """
        return await _execute_buy_fn(self, candidate)

    async def _monitor_positions(self):
        """P5-8 delegate → ms/monitor/position_monitor.run_monitor_positions.

        Phase 5-8 (2026-06-01): body extracted verbatim to
        ms/monitor/position_monitor.py::run_monitor_positions.
        self→ctrl substitution only. Call site in control_task UNCHANGED.
        """
        await _run_monitor_positions_fn(self)

    async def _execute_sell(self, pos: Position, reason: str, trigger_pnl_pct: float = 0.0,
                            trigger_price_sol: Optional[float] = None,
                            trigger_mid_price_sol: Optional[float] = None,
                            trigger_mid_source: Optional[str] = None):
        """P5-7 delegate → ms/execution/sell.execute_sell.

        Phase 5-7 (2026-06-01): body extracted verbatim to
        ms/execution/sell.py::execute_sell.  self→ctrl substitution
        only.  Call sites (_monitor_positions, _price_event_watcher,
        _rug_event_watcher) are UNCHANGED.
        """
        await _execute_sell_fn(
            self, pos, reason, trigger_pnl_pct,
            trigger_price_sol, trigger_mid_price_sol, trigger_mid_source)

    async def _refresh_sol_price(self):
        """Update cached SOL/USD price (at most every 60s).
        Fail-closed: if price is unavailable, leave _sol_price_usd at 0 — _execute_buy
        will check and refuse to open new positions with unknown SOL price."""
        if time.time() - self._last_sol_price_update < 60 and self._sol_price_usd > 0:
            return
        try:
            price = await self.trader.get_sol_price_usd()
            if price > 0:
                self._sol_price_usd = price
                self._last_sol_price_update = time.time()
                self.discovery.sol_price_usd = price
                logger.debug(f"SOL price updated: ${price:.2f}")
        except Exception as e:
            logger.warning(f"SOL price fetch failed: {e}")
            # Fail-closed: do NOT set a hardcoded fallback.
            # If we have a stale price (> 5min old), invalidate it.
            if time.time() - self._last_sol_price_update > 300:
                if self._sol_price_usd > 0:
                    logger.warning(f"SOL price stale (>5min), invalidating — buys blocked until refresh")
                    self._sol_price_usd = 0.0
