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
import logging
import math
import os
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import Field

from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction

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

try:
    from controllers.generic.geyser_stream import GeyserPumpSwapStream
    HAS_GEYSER = True
except ImportError:
    HAS_GEYSER = False

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


class MemeSniperConfig(ControllerConfigBase):
    controller_name: str = "meme_sniper"
    controller_type: str = "generic"
    candles_config: list = []

    # Gateway (direct REST API, not through GatewaySwap connector)
    gateway_url: str = Field(default="http://gateway-meme-sniper:15888",
                             json_schema_extra={"is_updatable": True})
    # Default empty — must be set via conf yml (per-host) or SOLANA_WALLET_ADDRESS env.
    # Never hardcode here: this file ships in the public Docker image.
    wallet_address: str = Field(default="",
                                json_schema_extra={"is_updatable": True})
    connector: str = "jupiter/router"
    chain_network: str = "solana-mainnet-beta"

    # Strategy params (from Gate 1.5/1.6 optimal — keep in sync with conf_meme_sniper.yml)
    position_size_usd: Decimal = Field(default=Decimal("10"),
                                       json_schema_extra={"is_updatable": True})

    # Phase 22.G.2 — Capacity-aware dynamic sizing (slippage-budget formula).
    # When sizing_mode="slippage_budget":
    #   L_sol  = pool_liq_usd / sol_price_usd / 2   (one-side SOL reserve)
    #   S_sol  = sizing_slippage_budget * L_sol      (max SOL such that one-way
    #                                                  slippage ≤ β)
    #   S_usd  = S_sol * sol_price_usd
    #   S_usd  = clamp(S_usd, sizing_min_usd, sizing_max_usd)
    # When sizing_mode="fixed": legacy behaviour (uses position_size_usd or
    # candidate.position_size_usd override). Default = "fixed" to keep
    # backward compatibility — flip to "slippage_budget" in yml when ready.
    # Theory: CPMM x*y=k gives slippage = Δx/x exactly (Angeris et al. 2020).
    # β=0.005 → ~0.5% one-way → ~1% round-trip.  Phase 22.G empirical THIN
    # edge is +4.13%, so 1% friction leaves ample positive expectancy.
    sizing_mode: str = Field(default="fixed",
                              json_schema_extra={"is_updatable": True})
    sizing_slippage_budget: float = Field(default=0.005,
                                           json_schema_extra={"is_updatable": True})
    sizing_min_usd: float = Field(default=5.0,
                                   json_schema_extra={"is_updatable": True})
    sizing_max_usd: float = Field(default=30.0,
                                   json_schema_extra={"is_updatable": True})
    # When True, candidate.position_size_usd (per-source override, e.g.
    # BigWinner's canary size) is honored as a HARD CAP on top of the
    # slippage-budget compute. ⚠️ DEFAULT IS FALSE — when running
    # slippage_budget mode the formula is the single source of truth for
    # capacity-aware sizing. Setting True will silently shrink big_winner
    # trades back to $5 regardless of pool depth, defeating dynamic sizing.
    # Set True only if you specifically want canary caps to remain active.
    sizing_respect_source_override_as_cap: bool = Field(default=False,
                                                          json_schema_extra={"is_updatable": True})
    # Defaults MUST match conf_meme_sniper.yml — yml is the source of truth.
    stop_loss_pct: float = Field(default=0.30,
                                 json_schema_extra={"is_updatable": True})
    time_limit_sec: int = Field(default=1800,
                                json_schema_extra={"is_updatable": True})
    slippage_pct: float = Field(default=2.0,
                                json_schema_extra={"is_updatable": True})
    max_buy_price_impact_pct: float = Field(default=0.10,
                                            json_schema_extra={"is_updatable": True})
    max_buy_vs_mid_premium_pct: float = Field(default=0.15,
                                              json_schema_extra={"is_updatable": True})
    max_entry_chase_pct: float = Field(default=0.30,
                                       json_schema_extra={"is_updatable": True})
    require_live_market_snapshot: bool = Field(default=True,
                                               json_schema_extra={"is_updatable": True})
    require_route_uses_biggest_pool: bool = Field(default=True,
                                                  json_schema_extra={"is_updatable": True})
    # Phase 22.S.1 (2026-04-30) — Jupiter index wait removal.
    # Old: hardcoded 3s sleep after register_token before preflight quote.
    # Live data: avg entry chase = 11% on 45 trades; 3s sleep at 3.1%/sec
    # accounts for ~9% of that chase. Removing it saves ~$0.30-0.50/trade
    # at $5 sizing, ~$2-13/day expected.
    # Token age at M3: 7-15min (v3.3) or 3-10min (big_winner). Jupiter
    # typically indexes within 30-90s, so wait is unnecessary.
    # Set > 0 (e.g. 1.0 or 3.0) for instant rollback if quote-fail rate
    # spikes after deploy. quote failures are safely caught in preflight.
    buy_jupiter_wait_sec: float = Field(default=0.0,
                                          json_schema_extra={"is_updatable": True})

    # V-shape trail_20_10: activate at +20% peak, sell when drops 10% from peak.
    trailing_activation_pct: float = Field(default=0.20,
                                           json_schema_extra={"is_updatable": True})
    # Phase 25n (2026-05-04): tightened 0.10→0.07. Code default must match
    # yml to keep yml-fail fallback safe.
    trailing_drop_pct: float = Field(default=0.07,
                                     json_schema_extra={"is_updatable": True})

    # Risk management
    max_positions: int = Field(default=5, json_schema_extra={"is_updatable": True})
    cooldown_sec: float = Field(default=30.0, json_schema_extra={"is_updatable": True})
    candidate_queue_ttl_sec: float = Field(default=120.0,
                                            json_schema_extra={"is_updatable": True})
    daily_loss_limit_usd: Decimal = Field(default=Decimal("200"),
                                          json_schema_extra={"is_updatable": True})
    max_total_trades: int = Field(default=500, json_schema_extra={"is_updatable": True})
    max_consecutive_losses: int = Field(default=50, json_schema_extra={"is_updatable": True})
    token_stoploss_cooldown_sec: float = Field(default=1800.0,
                                                json_schema_extra={"is_updatable": True})

    # Pre-trade safety filters.
    # Recalibrated 2026-05-07 (Phase M2-FilterRelax): 14d × 1537 forward sim showed
    # only `max_5min_return` is a net-positive filter. Other 4 reject groups have
    # mean PnL +19% to +42% (V-shape rebound population). Relaxed to ≈disabled.
    # Source: research_notebooks/meme_sniper/p13_statistical_research/outputs/
    #         filter_recalib/RECOMMENDATION_2026-05-07.md
    max_5min_return: float = Field(default=0.50,
                                   json_schema_extra={"is_updatable": True})
    max_vol_liq_ratio: float = Field(default=10.0,
                                     json_schema_extra={"is_updatable": True})

    # Disabled-style floors (kept as sanity checks for truly extreme tokens).
    # Old defaults 0.80 / -0.60 / -0.50 blocked V-shape rebounds (e.g. tokens
    # that dumped -90% then 80×'d). New defaults only reject near-dead tokens.
    max_cum_drawdown_at_entry: float = Field(default=0.99,
                                              json_schema_extra={"is_updatable": True})
    max_entry_return_floor: float = Field(default=-0.99,
                                           json_schema_extra={"is_updatable": True})
    lookback_return_floor: float = Field(default=-0.99,
                                          json_schema_extra={"is_updatable": True})

    # Data sources
    gmgn_base_url: str = "https://openapi.gmgn.ai"
    min_liquidity_usd: float = Field(default=5_000.0,
                                     json_schema_extra={"is_updatable": True})
    graduation_poll_interval: int = Field(default=10,
                                          json_schema_extra={"is_updatable": True})
    # V-shape v1.4: T+10m entry window (600s = 10 × 1min bars).
    feature_delay_sec: int = Field(default=600, json_schema_extra={"is_updatable": True})
    price_poll_interval: int = Field(default=5, json_schema_extra={"is_updatable": True})
    price_poll_interval_trailing: int = Field(default=2, json_schema_extra={"is_updatable": True})
    swap_poll_interval: int = Field(default=60, json_schema_extra={"is_updatable": True})

    # 30min raw-swap collection window for v5.2 retraining dataset.
    # All graduated tokens keep their swap collector alive until grad+window,
    # regardless of trade decision. Traded tokens override via _swap_keepalive_mints.
    swap_collection_window_sec: int = Field(default=1800,
                                             json_schema_extra={"is_updatable": True})

    # P13 instant-buy mode: skip feature_delay and M2 evaluation entirely.
    # Buys immediately after M1 safety filters pass. M2 runs post-buy for observation only.
    # Purpose: measure real-world latency for the zero-delay edge discovered in P13 Phase 1.
    instant_buy_mode: bool = Field(default=False, json_schema_extra={"is_updatable": True})

    # Phase 6.1 live shadow / latency instrumentation.
    # These are logging-only skeletons and do not change live trade decisions
    # unless explicitly wired into the strategy later.
    shadow_rule_first_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    latency_events_enabled: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    shadow_rule_stage2_name: str = Field(default="drawdown_1to3m_ge_q60",
                                         json_schema_extra={"is_updatable": True})
    shadow_rule_range_1to3m_max: float = Field(default=0.536496,
                                               json_schema_extra={"is_updatable": True})
    shadow_rule_drawdown_1to3m_min: float = Field(default=0.0,
                                                  json_schema_extra={"is_updatable": True})
    ev3m_live_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    ev3m_live_model_path: str = Field(default="/home/hummingbot/models/confirmed_continuation_ev3m_v2.pkl",
                                      json_schema_extra={"is_updatable": True})
    ev3m_live_entry_delay_sec: int = Field(default=90, json_schema_extra={"is_updatable": True})
    ev3m_live_selection_band: str = Field(default="top10", json_schema_extra={"is_updatable": True})
    ev3m_live_disable_trailing: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    shadow_ev3m_enabled: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    shadow_ev3m_model_path: str = Field(default="/home/hummingbot/models/confirmed_continuation_ev3m_v2.pkl",
                                        json_schema_extra={"is_updatable": True})
    shadow_ev3m_entry_delay_sec: int = Field(default=90, json_schema_extra={"is_updatable": True})
    shadow_ev3m_selection_bands: str = Field(default="top10,top15",
                                             json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_model_path: str = Field(
        default="/home/hummingbot/models/super_winner_event_shadow_v1.pkl",
        json_schema_extra={"is_updatable": True},
    )
    shadow_super_winner_event_scan_start_sec: int = Field(default=180, json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_scan_end_sec: int = Field(default=600, json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_scan_step_sec: int = Field(default=60, json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_default_band: str = Field(default="top3", json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_fallback_band: str = Field(default="top7p5", json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_arbitration_rule: str = Field(
        default="earliest_then_ab_then_score",
        json_schema_extra={"is_updatable": True},
    )
    shadow_super_winner_event_default_policy: str = Field(default="fixed_15m", json_schema_extra={"is_updatable": True})
    shadow_super_winner_event_fallback_policy: str = Field(default="break30_time15", json_schema_extra={"is_updatable": True})

    # V-shape T+10m model (Phase 8 / 14n)
    shadow_vshape_enabled: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    shadow_vshape_model_path: str = Field(
        default="/home/hummingbot/models/vshape_v1_model.pkl",
        json_schema_extra={"is_updatable": True},
    )
    shadow_vshape_selection_bands: str = Field(default="top10,top15,top20",
                                               json_schema_extra={"is_updatable": True})
    # Phase 25g (2026-05-03) fail-closed safety net.
    # If True and no entry-model gate sets entry_source on a candidate,
    # the candidate is REJECTED before reaching M3 buy queue. Prevents
    # silent regression where v3.4 + big_winner gates fail to fire and
    # tokens flow through ungated. Default True for safety.
    require_entry_source: bool = Field(default=True, json_schema_extra={"is_updatable": True})

    # Legacy V-shape v1.6.1 live gate. Keep disabled when the rolling v3/v3.4
    # first-passage gate is enabled; running both creates an unintended
    # double-gate and v1.6.1 has a known last-bar leakage issue.
    vshape_live_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    vshape_live_selection_band: str = Field(default="top10", json_schema_extra={"is_updatable": True})
    # V-shape early crash detector (Phase 1 SL cap)
    # If price drops > threshold within window after entry → exit immediately
    vshape_early_crash_pct: float = Field(default=0.10, json_schema_extra={"is_updatable": True})
    vshape_early_crash_window_sec: int = Field(default=120, json_schema_extra={"is_updatable": True})

    # Phase 24 V-shape v3.4 rolling first-passage scan.
    # Replaces v1.6.1 (which has detect_vshape_live last-bar leakage bug).
    # When enabled, scans every `vshape_entry_scan_step_sec` in
    # [scan_min_sec, scan_max_sec] and fires on first cross above cutoff.
    # Set vshape_live_enabled=False to disable v1.6.1 simultaneously.
    # Model bundle is currently vshape_v3_4_model.pkl. The legacy v3.1/v3.3
    # field names below are compatibility aliases only.
    # Canonical fields use vshape_entry_*; vshape_v3_1_* remains as a
    # boot-time compatibility alias for older yml/deploy bundles.
    vshape_entry_enabled: Optional[bool] = Field(default=None,
                                                  json_schema_extra={"is_updatable": True})
    vshape_entry_model_version: str = Field(default="vshape_v3_4",
                                             json_schema_extra={"is_updatable": True})
    vshape_entry_model_path: Optional[str] = Field(default=None,
                                                    json_schema_extra={"is_updatable": True})
    vshape_entry_selection_band: Optional[str] = Field(default=None,
                                                        json_schema_extra={"is_updatable": True})
    vshape_entry_cutoff_override: Optional[float] = Field(default=None,
                                                           json_schema_extra={"is_updatable": True})
    vshape_entry_ood_min_buyers: Optional[int] = Field(default=None,
                                                        json_schema_extra={"is_updatable": True})
    vshape_entry_ood_min_buy_vol_sol: Optional[float] = Field(default=None,
                                                               json_schema_extra={"is_updatable": True})
    vshape_entry_scan_min_sec: Optional[int] = Field(default=None,
                                                      json_schema_extra={"is_updatable": True})
    vshape_entry_scan_max_sec: Optional[int] = Field(default=None,
                                                      json_schema_extra={"is_updatable": True})
    vshape_entry_scan_step_sec: Optional[int] = Field(default=None,
                                                       json_schema_extra={"is_updatable": True})
    vshape_v3_1_enabled: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    # Phase 22 v3.3 retrain (2026-04-30): direct replacement of v3.1 D.
    # 33 features = 29 v3.1 core + 4 OOD-defense (sf_swap_density,
    # hk_branching_buy, hc_top3_share_t, tb_kyle_lambda_180s).
    vshape_v3_1_model_path: str = Field(
        default="/home/hummingbot/models/vshape_v3_4_model.pkl",
        json_schema_extra={"is_updatable": True},
    )
    vshape_v3_1_selection_band: str = Field(default="top_5pct",
                                             json_schema_extra={"is_updatable": True})
    # cutoff_override pins live cutoff at the v3.3 top_5pct holdout value
    # (0.6746). Setting it explicitly documents the live threshold and lets
    # us tune without touching the bundle. Set to -1.0 to revert to band lookup.
    vshape_v3_1_cutoff_override: float = Field(default=0.7223,
                                                json_schema_extra={"is_updatable": True})
    # OOD pre-filter (2026-04-30 audit fix). v3.1 D was trained on Birdeye Tier 2
    # data which excludes dead/low-volume tokens; live Chainstack data has more
    # of these and the model gives spurious high scores to them (overfitting).
    # Reject entries where m5 buyer/volume features fall below training p10.
    # Empirically humancoin lost $1.29 on m5_unique_buyers=58 (training p10).
    vshape_v3_1_ood_min_buyers: int = Field(default=60,
                                              json_schema_extra={"is_updatable": True})
    vshape_v3_1_ood_min_buy_vol_sol: float = Field(default=30.0,
                                                     json_schema_extra={"is_updatable": True})
    vshape_v3_1_scan_min_sec: int = Field(default=420,
                                           json_schema_extra={"is_updatable": True})  # T+7min
    vshape_v3_1_scan_max_sec: int = Field(default=900,
                                           json_schema_extra={"is_updatable": True})  # T+15min
    vshape_v3_1_scan_step_sec: int = Field(default=30,
                                            json_schema_extra={"is_updatable": True})

    # Phase 15a/15d (2026-04-26): three-model gate deployment
    # rug_v3_gate_enabled REMOVED 2026-05-06 — v3 entry-gate retired in favor
    # of v4 (Hopeless Trajectory T+5m). See model_specs/2026-05-06_rug_filter_v4_*.md
    # v4_exit_enabled: promote v4 from shadow to real soft-exit gate.
    # Score column: p_rug_v4 (ensemble) | p_y_60s | p_y_120s. Cutoff calibrated
    # by backtest (Phase 15d: 0.65 on p_rug_v4 → +$3.59 vs live -$4.46 on 37 trades).
    # Requires shadow_exit_warn_enabled=True. Soft-exit fires only when
    # price_source=="grpc_pool" and hold_sec >= 30 (let EC handle first 30s).
    v4_exit_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    v4_exit_score: str = Field(default="p_rug_v4", json_schema_extra={"is_updatable": True})
    v4_exit_cutoff: float = Field(default=0.40, json_schema_extra={"is_updatable": True})
    v4_exit_score_max_age_sec: float = Field(default=60.0, json_schema_extra={"is_updatable": True})
    v4_exit_grace_sec: int = Field(default=30, json_schema_extra={"is_updatable": True})
    # Phase 15d (2026-04-27) profit-protect mode:
    # v4 was firing at -25% pnl (acting as SL confirmation, hurting PnL by 7pp
    # vs no-v4 baseline). Add two gates so v4 only fires when:
    #   (1) position is in profit (pnl >= v4_exit_min_pnl_pct, default +5%)
    #   (2) price has dropped from peak (drawdown_from_peak >=
    #       v4_exit_min_drawdown_pct, default 3pp)
    # Combined with lowered cutoff (0.65→0.40), v4 becomes a peak-detector
    # that fires when an in-profit position starts reversing.
    v4_exit_min_pnl_pct: float = Field(default=0.05,
                                        json_schema_extra={"is_updatable": True})
    v4_exit_min_drawdown_pct: float = Field(default=0.03,
                                              json_schema_extra={"is_updatable": True})

    # Phase 15d (2026-04-27) v5.2 lean profit-protect gate. Replaces v4_exit.
    # Backtest (37 real trades, $10 sizing, byte-aligned features after
    # 4/27 rebuild): actual $-44.64 → v5.2 $+56.22 (+$101 / 37 trades, win
    # rate 43%→87%). Cutoff 0.30 selected by LOOCV; 80% bootstrap stable.
    # Pre-deploy audit 22/22 PASS.
    # Score column: p_dd_v5 (ensemble) | p_dd_60s | p_dd_120s. The v5.2 model
    # already encodes profit-protect (label only fires when cur_pnl ≥ +5%),
    # so unlike v4_exit there is NO additional PnL/drawdown gate.
    # Requires shadow_exit_warn_enabled=True (score producer).
    v5_exit_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    v5_exit_score: str = Field(default="p_dd_v5", json_schema_extra={"is_updatable": True})
    v5_exit_cutoff: float = Field(default=0.30, json_schema_extra={"is_updatable": True})
    v5_exit_score_max_age_sec: float = Field(default=60.0,
                                              json_schema_extra={"is_updatable": True})
    v5_exit_grace_sec: int = Field(default=30, json_schema_extra={"is_updatable": True})

    # ─── Phase 15e (2026-04-29) v5.3 lean profit-protect ──────────────────────
    # Trained on 1.57M rows from historical BigWinner + V-shape entry distribution.
    # Bot policy in training matches CURRENT live (trail_020/drop_010/sl_030/TL_900).
    # 22/22 pre-deploy audit PASS.
    # When `v5_3_exit_enabled=True`: load model + log shadow evals every tick.
    # When `v5_3_exit_shadow_only=True`: log only, do NOT fire real exits.
    # When `v5_3_exit_shadow_only=False` AND `v5_exit_enabled=False`: v5.3 fires
    #   real soft-exits at cutoff, replacing v5.2.
    v5_3_exit_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    v5_3_exit_shadow_only: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    v5_3_exit_cutoff: float = Field(default=0.50, json_schema_extra={"is_updatable": True})
    # Phase 25n (2026-05-04): peak guard for v5.3 — skip fire when position
    # already up >= threshold. Audit (24h, 37 v5.3 fires) found avg peak 15%
    # but 9 trades had peak ≥20% (avg peak 25%) where v5.3 fired prematurely
    # killing momentum-continuation winners. Letting these run to trail/TP
    # captures more upside. Calibrated 2026-05-06 on 219 trades: 0.20
    # spares 3× more v5.3 fires than 0.30 (+$41 vs +$14 cap).
    v5_3_exit_peak_skip_threshold: float = Field(
        default=0.20, json_schema_extra={"is_updatable": True})
    v5_3_exit_grace_sec: int = Field(default=30, json_schema_extra={"is_updatable": True})
    v5_3_exit_score_max_age_sec: float = Field(default=60.0,
                                                  json_schema_extra={"is_updatable": True})

    # ─── Phase 15g (2026-05-11) v5.5.1 Chainstack-native exit ─────────────────
    # 12-feature Path C (drops ps_current_pnl_pct to avoid v5.3-style single-
    # feature dependency). Bug-fix retrain corrects label/eval-set/PnL bugs.
    # Phase 3 audit 27/28 PASS. Initial deploy: SHADOW only (logs to
    # shadow_v5_5_evals; v5.3 still fires). Day-7 cutoff recalib on live dist.
    # Fire requires BOTH: raw_cur_pnl ≥ profit_gate AND p_dd_v5_5 ≥ cutoff.
    v5_5_exit_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    v5_5_exit_shadow_only: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    v5_5_exit_cutoff: float = Field(default=0.60, json_schema_extra={"is_updatable": True})
    v5_5_exit_profit_gate: float = Field(default=0.05, json_schema_extra={"is_updatable": True})
    v5_5_exit_grace_sec: int = Field(default=30, json_schema_extra={"is_updatable": True})
    v5_5_exit_score_max_age_sec: float = Field(default=60.0,
                                                json_schema_extra={"is_updatable": True})

    # v5.5.3 shadow exit (Phase 15i, 2026-05-11): sandwich filter + fragility clips
    # (Amihud per-swap cap 100, return [-1,5], volatility per-return ±5). Same 12 features
    # as v5.5.2. Shadow-only at deploy; logs to shadow_v5_5_3_evals for A/B vs v5.5.2 LIVE.
    v5_5_3_exit_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    v5_5_3_exit_shadow_only: bool = Field(default=True, json_schema_extra={"is_updatable": True})
    v5_5_3_exit_cutoff: float = Field(default=0.35, json_schema_extra={"is_updatable": True})
    v5_5_3_exit_profit_gate: float = Field(default=0.05, json_schema_extra={"is_updatable": True})
    v5_5_3_exit_grace_sec: int = Field(default=30, json_schema_extra={"is_updatable": True})
    v5_5_3_exit_score_max_age_sec: float = Field(default=60.0,
                                                  json_schema_extra={"is_updatable": True})
    # Optional peak guard (default 1.0 = disabled). v5.5 Path C has
    # `ps_peak_pnl_so_far` as a model feature so peak handling is already
    # baked in — peak_skip is opt-in only if live monitoring shows v5.5
    # firing prematurely on momentum continuation (mirror v5.3 Phase 25n
    # finding). Set < 1.0 in yml to enable.
    v5_5_exit_peak_skip_threshold: float = Field(default=1.0,
                                                  json_schema_extra={"is_updatable": True})

    # Exit-layer hardening (BK postmortem, 2026-04-14)
    # take_profit_pct: hard take-profit. When pool-backed PnL >= this, exit
    # immediately without waiting for trail_drop. Protects against parabolic
    # pumps where a 10% trail retracement still loses 100%+ if liquidity
    # evaporates. 0 disables.
    take_profit_pct: float = Field(default=2.00, json_schema_extra={"is_updatable": True})

    # DexScreener data collection: adds one API call at t0 snapshot time
    # to capture orthogonal signals (paid boosts, social links, dex-side
    # liquidity). Stored to token_observations.dexscr_info_t0. Will feed
    # rug_filter_v2 once 2-4 weeks of data accumulates. Never gates trades.
    dexscreener_collect_enabled: bool = Field(default=False,
                                               json_schema_extra={"is_updatable": True})

    # Rug event exit: subscribe open-position pools to the gRPC stream.
    # When a single sell swap >= rug_event_sol_threshold hits the pool, fire
    # panic sell immediately (bypasses M4 poll loop). Empirically, SL fires
    # at -86% because rugs complete in one block; this path catches the
    # dump transaction itself. Disabled by default for rollout safety.
    rug_event_exit_enabled: bool = Field(default=False,
                                         json_schema_extra={"is_updatable": True})
    rug_event_sol_threshold: float = Field(default=15.0,
                                           json_schema_extra={"is_updatable": True})
    rug_event_sell_slippage_pct: float = Field(default=20.0,
                                               json_schema_extra={"is_updatable": True})
    # Phase 15d (2026-04-27): rug_event_exit pnl-gated to protect winners.
    # Backtest on 52 historical trades showed UNGATED rug_event firing cuts
    # 3 trailing_stop winners early (-$2.50 hurt). Gating by current pnl
    # eliminates winner-cuts: pnl<0 + threshold=7.0 → +$10.23 net (vs +$8.18
    # ungated). Set to 0.0 to fire only when position is in loss; set high
    # (e.g. 1.0) to disable the gate (legacy behavior).
    rug_event_exit_max_pnl_pct: float = Field(default=0.0,
                                               json_schema_extra={"is_updatable": True})

    # Phase 22.S.P1 (2026-05-01) — gRPC event-driven SL/EC fast-trigger.
    # Replaces 5s/2s polling for SL + early_crash with sub-second gRPC events.
    # Polling stays as fallback for trail / v5.3 / time_limit.
    # Source: Phase_22_Latency_Optimization_PLAN_2026-04-30.md §3.1
    # Default OFF for safety. Set shadow_only=true first for 7d audit then
    # flip to false to enable real exits.
    grpc_event_exit_enabled: bool = Field(default=False,
                                            json_schema_extra={"is_updatable": True})
    grpc_event_exit_sl_enabled: bool = Field(default=True,
                                               json_schema_extra={"is_updatable": True})
    grpc_event_exit_ec_enabled: bool = Field(default=True,
                                               json_schema_extra={"is_updatable": True})
    grpc_event_min_price_delta: float = Field(default=0.02,
                                                json_schema_extra={"is_updatable": True})
    grpc_event_exit_shadow_only: bool = Field(default=True,
                                                json_schema_extra={"is_updatable": True})

    # Rug filter shadow: score each token at M2 promotion using
    # rug_filter_v1 (t0 GMGN structural classifier). Logs to
    # `shadow_rug_evals`. Does NOT affect real trades.
    shadow_rug_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    shadow_rug_model_path: str = Field(
        default="/home/hummingbot/models/rug_filter_v1_model.pkl",
        json_schema_extra={"is_updatable": True})
    shadow_rug_cutoff_band: str = Field(default="top10",
                                         json_schema_extra={"is_updatable": True})
    # Rug filter soft-gate: when enabled, M3 preflight blocks buys for
    # tokens whose *t0* rug score exceeds the top-N cutoff. "top5" is the
    # strictest band (highest cutoff, lowest false-positive rate) so we
    # only block clear rugs. Shadow scoring still runs for all tokens.
    rug_filter_gate_enabled: bool = Field(default=False,
                                          json_schema_extra={"is_updatable": True})
    rug_filter_gate_cutoff_band: str = Field(default="top5",
                                             json_schema_extra={"is_updatable": True})

    # Rug filter v3 shadow REMOVED 2026-05-06.
    # See `shadow_rug_v4_*` below for the v4 (Hopeless Trajectory) replacement.

    # Rug filter v4 shadow (2026-05-06): scores each graduation at T+5m using
    # 5 path-geometry + concentration features. Logs to `shadow_rug_filter_v4_evals`.
    # Shadow-only for 7-14 days, then validate precision/recall before gate.
    # Spec: model_specs/2026-05-06_rug_filter_v4_CTA_FROZEN_SPEC.md
    shadow_rug_v4_enabled: bool = Field(default=False,
                                         json_schema_extra={"is_updatable": True})
    shadow_rug_v4_model_path: str = Field(
        default="/home/hummingbot/models/rug_v4_ensemble_balanced.pkl",
        json_schema_extra={"is_updatable": True})
    # 2026-05-07 RECALIBRATED to 0.92 (was 0.65).
    # Spec-frozen 0.65 was calibrated on Birdeye holdout (2.2% rug) → 8.3% flag rate.
    # On bot cohort (43% rug, post M2+v3.1+v1.6.1 pre-filter), 0.65 reject = 41% (over-aggressive).
    # Live recalibration on n=106 resolved evals: 0.92 → 25% reject, +$0.21/trade.
    # R1 retained but is empirically worthless on bot cohort (fires 1.9%, 0/2 hits).
    shadow_rug_v4_cutoff: float = Field(default=0.92,
                                         json_schema_extra={"is_updatable": True})
    # Window for v4 features: T+0 to T+300s post-graduation.
    shadow_rug_v4_window_sec: int = Field(default=300,
                                           json_schema_extra={"is_updatable": True})
    # Minimum swap count to score (training corpus had ≥5 in window; we add buffer)
    shadow_rug_v4_min_swaps: int = Field(default=5,
                                          json_schema_extra={"is_updatable": True})

    # ⭐ Rug filter v4.2 v0.9i hybrid (drawdown_2min < -55%, raw XGB prob, no LR cal)
    # 18 features (microstructure + counts + macro). Birdeye holdout AUC 0.7024,
    # production big_loss AUC 0.5573. Top 20% gate catches ~25% big losses.
    # SHADOW ONLY during validation phase (2026-05-09 onwards).
    # Spec: model_specs/2026-05-08_rug_filter_v4_2_SPEC.md v0.9f section
    shadow_rug_v4_2_enabled: bool = Field(default=True,
                                            json_schema_extra={"is_updatable": True})
    shadow_rug_v4_2_model_path: str = Field(
        default="/home/hummingbot/models/rug_v4_2_v0_9i_hybrid.pkl",
        json_schema_extra={"is_updatable": True})
    # cutoff = top-20% val_raw threshold (best alignment +2.4pp drift).
    # See v4_2_v0_9i_hybrid.pkl["cutoffs_raw"] for {top5..top30} options.
    shadow_rug_v4_2_cutoff: float = Field(default=0.5040,
                                            json_schema_extra={"is_updatable": True})
    shadow_rug_v4_2_window_sec: int = Field(default=300,
                                              json_schema_extra={"is_updatable": True})
    shadow_rug_v4_2_min_swaps: int = Field(default=10,
                                             json_schema_extra={"is_updatable": True})

    # Phase B shadow: event-triggered entry scanner (research artifact).
    # Scans T+3m..T+15m, records hypothetical first-trigger entries to
    # `shadow_event_evals`. Does NOT affect real trades.
    shadow_event_enabled: bool = Field(default=False, json_schema_extra={"is_updatable": True})
    shadow_event_model_path: str = Field(
        default="/home/hummingbot/models/flow_v1_model.pkl",
        json_schema_extra={"is_updatable": True})
    shadow_event_cutoff_band: str = Field(default="top30",
                                           json_schema_extra={"is_updatable": True})
    # Event Invariant v1 Model B (regressor) — Phase B research, the ONLY
    # combo in holdout that produced positive first-trigger EV when
    # filtered to T+3-5m (30 tokens, 73% win, +$12.68). Narrow window,
    # strict cutoff (top5=-0.0095). Shadow-only; zero trading impact.
    # Goal: validate the research finding on live tokens to decide whether
    # to promote to real entry gate.
    shadow_event_invariant_enabled: bool = Field(
        default=False, json_schema_extra={"is_updatable": True})
    shadow_event_invariant_model_path: str = Field(
        default="/home/hummingbot/models/event_invariant_v1_model.pkl",
        json_schema_extra={"is_updatable": True})
    shadow_event_invariant_cutoff_band: str = Field(
        default="top5", json_schema_extra={"is_updatable": True})
    shadow_event_invariant_scan_start_sec: int = Field(
        default=180, json_schema_extra={"is_updatable": True})
    shadow_event_invariant_scan_end_sec: int = Field(
        default=300, json_schema_extra={"is_updatable": True})
    shadow_event_scan_start_sec: int = Field(default=180,
                                              json_schema_extra={"is_updatable": True})
    shadow_event_scan_end_sec: int = Field(default=900,
                                            json_schema_extra={"is_updatable": True})
    shadow_event_scan_step_sec: int = Field(default=30,
                                             json_schema_extra={"is_updatable": True})

    # Phase 25o (2026-05-04) — Simple T+5m entry path.
    # Bypass V-shape and big_winner model gating entirely. At T+offset_sec
    # post-graduation, fire entry IF (1) M2 safety pass.
    # (RugFilter v3 entry-gate removed 2026-05-06; v4 in shadow-only.)
    # Goal: maximize exit-model test volume; entry models reduced to
    # shadow-log status. Sim (8 days, 1087 mints) shows fixed T+5m entry
    # +$8556 vs model-gated +$615.
    # Default True to match yml (sole entry path post 2026-05-04).
    simple_t5m_enabled: bool = Field(default=True,
                                       json_schema_extra={"is_updatable": True})
    simple_t5m_offset_sec: int = Field(default=300,
                                         json_schema_extra={"is_updatable": True})
    simple_t5m_offset_window_sec: int = Field(default=60,
                                                json_schema_extra={"is_updatable": True})
    simple_t5m_position_size_usd: float = Field(default=10.0,
                                                  json_schema_extra={"is_updatable": True})
    simple_t5m_entry_source: str = Field(default="simple_t5m",
                                           json_schema_extra={"is_updatable": True})
    # B7 fix (2026-05-09): non-negative slope pre-filter.
    # Live audit found 53% (202/380) of trades had peak <= 0 — many entered
    # while token was in active dump. Reject if slope in last 60s before
    # T+5m entry < -5%. Default OFF for shadow validation; enable via yml.
    simple_t5m_min_slope_enabled: bool = Field(default=True,
                                                  json_schema_extra={"is_updatable": True})
    simple_t5m_min_slope_pct: float = Field(default=-0.05,
                                              json_schema_extra={"is_updatable": True})
    # B8 fix (2026-05-09): max_slope upper cap — reject if pump too steep.
    # Audit (n=87 24h trades): 27% of big_losers had slope > +20% (pump-dump
    # trap) vs winners' p75 = only +6.6%. +20% cap → catches 27% of big losses
    # (11/41), 18% winner false-positive (2/11), net +$5.70/24h at $1 sizing.
    simple_t5m_max_slope_pct: float = Field(default=0.20,
                                              json_schema_extra={"is_updatable": True})

    # Phase 24 BigWinner v2 entry-side filter.
    # Canonical fields use big_winner_*; big_winner_v1_* remains as a
    # boot-time compatibility alias for older yml/deploy bundles.
    big_winner_enabled: Optional[bool] = Field(default=None,
                                                json_schema_extra={"is_updatable": True})
    big_winner_model_version: str = Field(default="big_winner_v2",
                                           json_schema_extra={"is_updatable": True})
    big_winner_entry_source: str = Field(default="big_winner_v2",
                                          json_schema_extra={"is_updatable": True})
    big_winner_shadow_only: Optional[bool] = Field(default=None,
                                                    json_schema_extra={"is_updatable": True})
    big_winner_cutoff: Optional[float] = Field(default=None,
                                                json_schema_extra={"is_updatable": True})
    big_winner_scan_start_sec: Optional[int] = Field(default=None,
                                                      json_schema_extra={"is_updatable": True})
    big_winner_scan_end_sec: Optional[int] = Field(default=None,
                                                    json_schema_extra={"is_updatable": True})
    big_winner_scan_step_sec: Optional[int] = Field(default=None,
                                                     json_schema_extra={"is_updatable": True})
    big_winner_token_cooldown_sec: Optional[int] = Field(default=None,
                                                          json_schema_extra={"is_updatable": True})
    big_winner_position_size_usd: Optional[float] = Field(default=None,
                                                           json_schema_extra={"is_updatable": True})
    big_winner_canary_trade_limit: Optional[int] = Field(default=None,
                                                          json_schema_extra={"is_updatable": True})
    big_winner_max_5min_return: Optional[float] = Field(default=None,
                                                         json_schema_extra={"is_updatable": True})
    big_winner_max_vol_liq_ratio: Optional[float] = Field(default=None,
                                                           json_schema_extra={"is_updatable": True})
    big_winner_max_entry_return_floor: Optional[float] = Field(default=None,
                                                                json_schema_extra={"is_updatable": True})
    big_winner_max_cum_drawdown: Optional[float] = Field(default=None,
                                                          json_schema_extra={"is_updatable": True})
    big_winner_lookback_return_floor: Optional[float] = Field(default=None,
                                                               json_schema_extra={"is_updatable": True})
    big_winner_v1_enabled: bool = Field(default=False,
                                          json_schema_extra={"is_updatable": True})
    big_winner_v1_shadow_only: bool = Field(default=True,
                                              json_schema_extra={"is_updatable": True})
    big_winner_v1_cutoff: float = Field(default=0.7342,
                                          json_schema_extra={"is_updatable": True})
    big_winner_v1_scan_start_sec: int = Field(default=180,
                                                json_schema_extra={"is_updatable": True})
    big_winner_v1_scan_end_sec: int = Field(default=600,
                                              json_schema_extra={"is_updatable": True})
    big_winner_v1_scan_step_sec: int = Field(default=30,
                                               json_schema_extra={"is_updatable": True})
    big_winner_v1_token_cooldown_sec: int = Field(default=600,
                                                    json_schema_extra={"is_updatable": True})
    # Phase 16.3 Step 3 LIVE — per-source sizing override.
    # Smaller default ($5) than global position_size_usd ($10) for canary
    # phase. Only big_winner-triggered entries use this; V-shape stays at
    # config.position_size_usd.
    big_winner_v1_position_size_usd: float = Field(default=5.0,
                                                     json_schema_extra={"is_updatable": True})
    # Phase 16.3 Step 3 — auto-revert to shadow after N live trades.
    # 0 = no limit (no auto-revert). Counter is read from DB at boot
    # (count of trades.entry_source in BigWinner aliases) + bumped on each
    # successful big_winner buy. Once count >= limit, force shadow_only.
    big_winner_v1_canary_trade_limit: int = Field(default=0,
                                                    json_schema_extra={"is_updatable": True})
    # Phase 16.3 Step 3 — per-source M2 safety thresholds (alpha rescue).
    # 14d analysis (m2_filter_alpha_analysis.py) found that the V-shape M2
    # safety filters reject 75.6% of big_winner PASSes, and the rejected
    # subset has POSITIVE +$2.60/trade EV (M2 was reverse-selecting).
    # These wider thresholds keep only the catastrophic-outlier rejection
    # while letting big_winner enter the high-volatility tokens it was
    # specifically trained to find.
    # Defaults below recover ~$155 / 14d (vs $-13 with V-shape thresholds).
    big_winner_v1_max_5min_return: float = Field(default=2.00,
                                                   json_schema_extra={"is_updatable": True})
    big_winner_v1_max_vol_liq_ratio: float = Field(default=15.00,
                                                     json_schema_extra={"is_updatable": True})
    big_winner_v1_max_entry_return_floor: float = Field(default=-0.85,
                                                          json_schema_extra={"is_updatable": True})
    big_winner_v1_max_cum_drawdown: float = Field(default=0.95,
                                                    json_schema_extra={"is_updatable": True})
    big_winner_v1_lookback_return_floor: float = Field(default=-0.85,
                                                         json_schema_extra={"is_updatable": True})

    # 14y shadow-exit-warn — dual-model (Tier B drop-risk + F2a+HC rug-risk)
    # per-position per-tick OBSERVE_ONLY logging. See
    # reports/Phase8_14y_Shadow_Wiring_Spec.md. Default off.
    shadow_exit_warn_enabled: bool = Field(
        default=False, json_schema_extra={"is_updatable": True})
    shadow_exit_warn_tick_interval: int = Field(
        default=30, json_schema_extra={"is_updatable": True})
    shadow_exit_warn_min_swaps: int = Field(
        default=8, json_schema_extra={"is_updatable": True})
    shadow_exit_warn_resolver_interval: int = Field(
        default=120, json_schema_extra={"is_updatable": True})

    # 19c sw_v2 shadow — per-graduation F1 (sw_v2_f1_v2) entry score at T+180s.
    # OBSERVE_ONLY; score is logged to shadow_sw_v2_evals and resolved at
    # T+18m. See reports/Phase8_19c_SW_v2_Shadow_Wiring_Spec.md. Default off.
    shadow_sw_v2_enabled: bool = Field(
        default=False, json_schema_extra={"is_updatable": True})
    shadow_sw_v2_min_swaps: int = Field(
        default=5, json_schema_extra={"is_updatable": True})
    shadow_sw_v2_entry_offset_sec: int = Field(
        default=180, json_schema_extra={"is_updatable": True})
    shadow_sw_v2_resolver_interval: int = Field(
        default=900, json_schema_extra={"is_updatable": True})
    shadow_sw_v2_resolver_horizon_sec: int = Field(
        default=900, json_schema_extra={"is_updatable": True})
    # Model paths are resolved automatically by meme_sniper_exit_models.py
    # relative to its own location (controllers/generic/models/*.pkl),
    # so there is no path config here. To swap model files, replace the pkl
    # on disk at the fixed path.

    # GMGN Trending discovery REMOVED 2026-04-20. Chainstack graduation
    # detection has ~100% coverage and 6/6 historical trending picks were rugs.

    # Chainstack on-chain discovery (Tier 1) — leave empty to use GMGN Trenches fallback
    chainstack_rpc_url: str = Field(default="",
                                     json_schema_extra={"is_updatable": True})
    chainstack_batch_size: int = 1000  # Solana RPC max; paginate if still behind
    chainstack_tx_concurrency: int = 3   # parallel tx fetches (Chainstack ~25 RPS limit)

    # Yellowstone gRPC (Geyser) — replaces WS + HTTP polling when configured.
    # Single stream for M1 graduation detection and M2 swap collection.
    grpc_url: str = Field(default="", json_schema_extra={"is_updatable": True})
    grpc_token: str = Field(default="", json_schema_extra={"is_updatable": True})

    # Model
    model_path: str = Field(
        default="",
        json_schema_extra={
            "prompt": "Path to survival_model.pkl: ",
            "prompt_on_new": True,
        })


class MemeSniper(ControllerBase):
    """Hybrid architecture controller: ControllerBase lifecycle + direct Gateway REST API."""

    @staticmethod
    def _set_config_value(config: MemeSniperConfig, name: str, value: Any) -> None:
        """Persist canonical/legacy config field across pydantic v2.

        Phase 25g (2026-05-03) regression fix: setattr() under
        Hummingbot's BaseClientModel (validate_assignment=True +
        extra=forbid) silently fails to persist Optional[T]=None
        defaults — boot-time alias coalesce LOOKS like it worked
        (returns True) but heartbeat reads back None. Force-persist
        via object.__setattr__ which bypasses pydantic validators
        AND `__pydantic_fields_set__` accounting; field is then
        readable through normal getattr.
        """
        object.__setattr__(config, name, value)

    @classmethod
    def _coalesce_config_alias(cls, config: MemeSniperConfig,
                               canonical: str, legacy: str) -> Any:
        canonical_value = getattr(config, canonical, None)
        legacy_value = getattr(config, legacy)
        if canonical_value is None or (
            isinstance(canonical_value, str) and canonical_value.strip() == ""
        ):
            value = legacy_value
        else:
            value = canonical_value
        cls._set_config_value(config, canonical, value)
        cls._set_config_value(config, legacy, value)
        return value

    @classmethod
    def _normalize_model_config_aliases(cls, config: MemeSniperConfig) -> None:
        """Normalize current model config keys and legacy aliases at boot."""
        vshape_aliases = (
            ("vshape_entry_enabled", "vshape_v3_1_enabled"),
            ("vshape_entry_model_path", "vshape_v3_1_model_path"),
            ("vshape_entry_selection_band", "vshape_v3_1_selection_band"),
            ("vshape_entry_cutoff_override", "vshape_v3_1_cutoff_override"),
            ("vshape_entry_ood_min_buyers", "vshape_v3_1_ood_min_buyers"),
            ("vshape_entry_ood_min_buy_vol_sol", "vshape_v3_1_ood_min_buy_vol_sol"),
            ("vshape_entry_scan_min_sec", "vshape_v3_1_scan_min_sec"),
            ("vshape_entry_scan_max_sec", "vshape_v3_1_scan_max_sec"),
            ("vshape_entry_scan_step_sec", "vshape_v3_1_scan_step_sec"),
        )
        for canonical, legacy in vshape_aliases:
            cls._coalesce_config_alias(config, canonical, legacy)

        if not str(getattr(config, "vshape_entry_model_version", "") or "").strip():
            cls._set_config_value(config, "vshape_entry_model_version", "vshape_v3_4")

        big_winner_aliases = (
            ("big_winner_enabled", "big_winner_v1_enabled"),
            ("big_winner_shadow_only", "big_winner_v1_shadow_only"),
            ("big_winner_cutoff", "big_winner_v1_cutoff"),
            ("big_winner_scan_start_sec", "big_winner_v1_scan_start_sec"),
            ("big_winner_scan_end_sec", "big_winner_v1_scan_end_sec"),
            ("big_winner_scan_step_sec", "big_winner_v1_scan_step_sec"),
            ("big_winner_token_cooldown_sec", "big_winner_v1_token_cooldown_sec"),
            ("big_winner_position_size_usd", "big_winner_v1_position_size_usd"),
            ("big_winner_canary_trade_limit", "big_winner_v1_canary_trade_limit"),
            ("big_winner_max_5min_return", "big_winner_v1_max_5min_return"),
            ("big_winner_max_vol_liq_ratio", "big_winner_v1_max_vol_liq_ratio"),
            ("big_winner_max_entry_return_floor", "big_winner_v1_max_entry_return_floor"),
            ("big_winner_max_cum_drawdown", "big_winner_v1_max_cum_drawdown"),
            ("big_winner_lookback_return_floor", "big_winner_v1_lookback_return_floor"),
        )
        for canonical, legacy in big_winner_aliases:
            cls._coalesce_config_alias(config, canonical, legacy)

        if not str(getattr(config, "big_winner_model_version", "") or "").strip():
            cls._set_config_value(config, "big_winner_model_version", "big_winner_v2")
        if not str(getattr(config, "big_winner_entry_source", "") or "").strip():
            cls._set_config_value(config, "big_winner_entry_source", "big_winner_v2")

    def _is_big_winner_entry_source(self, source: str) -> bool:
        return str(source or "").strip() in self._big_winner_entry_source_aliases

    @staticmethod
    def _enabled_any(config, *names) -> bool:
        """True if ANY of the given config attrs is True. Defensive against
        Phase 24 alias coalesce regression (2026-05-03): if canonical fields
        like `big_winner_enabled`, `vshape_entry_enabled` somehow remain
        Optional[bool]=None at runtime (e.g., pydantic v2 validate_assignment
        edge case), fall back to legacy `*_v1_enabled` / `*_v3_1_enabled`."""
        for n in names:
            v = getattr(config, n, None)
            if v is True:
                return True
        return False

    @staticmethod
    def _cfg_value(config, *names, default=None):
        """First non-None value among the given attrs (canonical first,
        legacy fallback). Same defensive purpose as _enabled_any but for
        Optional[int]/Optional[str]/Optional[float] fields like
        `vshape_entry_scan_min_sec` (None) → `vshape_v3_1_scan_min_sec`
        (yml-set). Returns `default` if all None."""
        for n in names:
            v = getattr(config, n, None)
            if v is not None:
                return v
        return default

    def __init__(self, config: MemeSniperConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config: MemeSniperConfig = config
        self._normalize_model_config_aliases(config)
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
        self.discovery = TokenDiscovery(
            api_key=gmgn_key,
            base_url=config.gmgn_base_url,
            min_liquidity_usd=config.min_liquidity_usd,
            chainstack_rpc_url=chainstack_url,
            chainstack_batch_size=config.chainstack_batch_size,
            chainstack_tx_concurrency=config.chainstack_tx_concurrency,
            instant_buy_mode=config.instant_buy_mode,
        )
        if chainstack_url:
            logger.info(f"M1: Chainstack on-chain discovery enabled (batch={config.chainstack_batch_size})")
        else:
            logger.warning("M1: No Chainstack RPC configured — using GMGN Trenches fallback only")

        # M2: Signal Pipeline
        self.signal: Optional[SignalPipeline] = None
        if config.model_path and os.path.exists(config.model_path):
            self.signal = SignalPipeline(
                model_path=config.model_path,
                api_key=gmgn_key,
                base_url=config.gmgn_base_url,
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
                        config.shadow_ev3m_model_path
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
                        config.ev3m_live_model_path
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
                        config.shadow_super_winner_event_model_path
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
                    self.rug_filter_model = RugFilterModel(config.shadow_rug_model_path)
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

        # Phase B event-triggered scanner (shadow only)
        self.flow_model = None
        if config.shadow_event_enabled and HAS_EVENT_SCANNER:
            if config.shadow_event_model_path and os.path.exists(config.shadow_event_model_path):
                try:
                    self.flow_model = FlowModel(config.shadow_event_model_path)
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
                bundle = load_big_winner_v2()
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
                        config.shadow_event_invariant_model_path)
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
                    self.shadow_vshape_model = VShapeModel(config.shadow_vshape_model_path)
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
                    self.vshape_entry_model = VShapeModel(config.vshape_entry_model_path)
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
        # Concurrency guard: prevent double-fire from gRPC + poll on same mint
        self._exit_in_progress: Set[str] = set()

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
        self._sell_in_progress: set = set()
        self._sell_retry_after: Dict[str, float] = {}  # mint -> earliest retry timestamp
        self._sell_retry_count: Dict[str, int] = {}  # mint -> consecutive fail count
        self._sell_retry_notice_after: Dict[str, float] = {}  # mint -> next retry-wait log ts
        self._exit_signal_reason: Dict[str, str] = {}  # mint -> last emitted sell trigger reason
        self._price_miss_count: Dict[str, int] = {}  # mint -> consecutive price fetch failures
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
                logger.info(
                    f"SIMPLE-T5M: {token.symbol} PASS — entered at T+"
                    f"{int(age)}s (rug_v3 PASS, M2 safety ✓)")
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
                                            v3_passed = (v3_score >= cutoff_v3) and (not ood_block)
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

        # M1 + M2: discover and evaluate
        await self.update_processed_data()

        # Flush expired observations (GMGN kline fetch) + intermediate snapshots
        await self._flush_expired_observations()
        await self._collect_observation_snapshots()
        await self._run_shadow_rule_first()
        await self._run_shadow_ev3m()
        await self._run_shadow_super_winner_event()
        await self._run_shadow_vshape()
        await self._run_shadow_event()
        await self._run_shadow_event_invariant()
        await self._run_shadow_rug_filter_v4()
        await self._run_shadow_rug_filter_v4_2()
        await self._run_shadow_big_winner()

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
        ttl = self.config.candidate_queue_ttl_sec
        before = len(self._candidate_queue)
        self._candidate_queue = [c for c in self._candidate_queue
                                 if now - c.queued_at < ttl]
        expired = before - len(self._candidate_queue)
        if expired:
            logger.info(f"M3: expired {expired} stale candidates from queue (ttl={ttl:.0f}s)")

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
                continue
            if candidate.buy_fail_count >= max_buy_retries:
                bought.append(candidate)
                logger.info(f"M3: {candidate.token.symbol} evicted — {candidate.buy_fail_count} "
                            f"consecutive buy failures")
                # Token will never enter trade, free the swap collector
                self._cleanup_swap_keepalive(
                    candidate.token.mint_address, reason="buy retries exhausted")
                continue
            backoff = min(10 * (2 ** candidate.buy_fail_count), 120) if candidate.buy_fail_count > 0 else 0
            if candidate.buy_last_fail_time > 0 and now - candidate.buy_last_fail_time < backoff:
                continue
            if self.risk.can_trade(len(self._positions)):
                success = await self._execute_buy(candidate)
                if success:
                    bought.append(candidate)
                    break
                else:
                    candidate.buy_fail_count += 1
                    candidate.buy_last_fail_time = now
                    logger.info(f"M3: {candidate.token.symbol} buy failed (attempt "
                                f"#{candidate.buy_fail_count}), retry in {backoff:.0f}s")
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
                self._candidate_queue.clear()
                break

        # Remove bought candidates from queue
        for c in bought:
            self._candidate_queue.remove(c)

        # M4 + M5: monitor positions and check exits
        await self._monitor_positions()

    async def on_start(self):
        """Called once when controller starts."""
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

        # 14y shadow-exit-warn resolver (runs regardless of main flag; no-op
        # if table stays empty). Cheap.
        if self.config.shadow_exit_warn_enabled:
            self._shadow_exit_warn_resolver_task = asyncio.ensure_future(
                self._shadow_exit_warn_resolver_loop())
            logger.info(
                "M4: shadow-exit-warn resolver started "
                f"(interval={self.config.shadow_exit_warn_resolver_interval}s, "
                f"tick_interval={self.config.shadow_exit_warn_tick_interval}s)")

        # 19c sw_v2 shadow resolver (T+18m outcome backfill).
        if self.config.shadow_sw_v2_enabled:
            self._shadow_sw_v2_resolver_task = asyncio.ensure_future(
                self._shadow_sw_v2_resolver_loop())
            logger.info(
                "M4: sw_v2 shadow resolver started "
                f"(interval={self.config.shadow_sw_v2_resolver_interval}s, "
                f"horizon={self.config.shadow_sw_v2_resolver_horizon_sec}s)")

        # Recover pending observations into memory (for intermediate snapshots + flush)
        asyncio.ensure_future(self._recover_observations())

    def on_stop(self):
        """Cleanup on shutdown."""
        total_pnl = sum(t.pnl_usd for t in self._trade_log)
        logger.info("=" * 60)
        logger.info("MemeSniper controller STOPPING")
        logger.info(f"  total_trades = {self.risk.total_trades}")
        logger.info(f"  total_pnl    = ${total_pnl:.2f}")
        logger.info(f"  positions    = {len(self._positions)}")
        logger.info(f"  swap_keepalive_active = {len(self._swap_keepalive_mints)}")
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
            obs_id = self.db.record_observation(
                token, model_score, model_passed,
                reject_reason=reject_reason,
                features=features, kline_6m=kline_6m,
                gmgn_info_entry=gmgn_entry)
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
                    self.db.update_observation_snapshot(
                        obs_id, "gmgn_info_t0", gmgn_entry)
                    entry.snapshots_done.add("gmgn_info_t0")
                except Exception as e:
                    logger.debug(f"OBS: {token.symbol} early gmgn_info_t0 write failed: {e}")

            self._register_shadow_pending(token, obs_id)
            logger.info(f"OBS: {token.symbol} added to observation pool "
                        f"(id={obs_id}, GMGN fetch in {expire - time.time():.0f}s)")
        except Exception as e:
            logger.warning(f"OBS: failed to record observation for {token.symbol}: {e}")

    def _record_latency_event(self, token: GraduatedToken, event_name: str, **metadata):
        """Best-effort live latency instrumentation."""
        if not self.config.latency_events_enabled:
            return
        try:
            age_sec = time.time() - token.graduation_time if token.graduation_time else None
            self.db.record_latency_event(
                mint_address=token.mint_address,
                symbol=token.symbol,
                graduation_time=token.graduation_time,
                event_name=event_name,
                age_sec=age_sec,
                metadata=metadata or None,
            )
        except Exception as e:
            logger.debug(f"LATENCY: failed to record {event_name} for {token.symbol}: {e}")

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

    async def _get_wallet_token_balance_ui(self, mint_address: str) -> Optional[float]:
        """Best-effort on-chain token balance check for stale-position cleanup."""
        if not self.discovery:
            return None
        try:
            rpc = await self.discovery._get_rpc()
            if rpc is None:
                return None
            result = await rpc.get_token_accounts_by_owner(
                self.config.wallet_address,
                mint_address,
            )
            total_ui = 0.0
            for acct in (result or {}).get("value", []):
                info = (
                    acct.get("account", {})
                    .get("data", {})
                    .get("parsed", {})
                    .get("info", {})
                )
                total_ui += float(info.get("tokenAmount", {}).get("uiAmount") or 0.0)
            return total_ui
        except Exception as e:
            logger.debug(f"M3: wallet balance check failed for {mint_address[:12]}...: {e}")
            return None

    def _prune_stale_position(self, pos: Position, reason: str, wallet_balance_ui: float):
        """Drop a locally stale position that no longer exists on-chain."""
        mint = pos.token.mint_address
        if pos in self._positions:
            self._positions.remove(pos)
        self.db.remove_position(mint)
        if self._grpc_stream is not None and pos.token.pool_address:
            self._grpc_stream.unwatch_pool(pos.token.pool_address)
        self._sell_retry_after.pop(mint, None)
        self._sell_retry_count.pop(mint, None)
        self._sell_retry_notice_after.pop(mint, None)
        self._exit_signal_reason.pop(mint, None)
        self._price_miss_count.pop(mint, None)
        self._first_price_seen.discard(mint)
        self._cleanup_swap_keepalive(mint, reason=f"stale position pruned ({reason})")
        self._record_latency_event(
            pos.token,
            "position_pruned",
            reason=reason,
            wallet_balance_ui=wallet_balance_ui,
        )
        logger.error(   # L3 escalation (postmortem 2026-04-24): prune = no
                        # SELL = no trade record = direct financial loss.
            f"M3: {pos.token.symbol} — pruned stale local position "
            f"(reason={reason}, wallet_balance={wallet_balance_ui:.8f}) — "
            f"NO SELL EXECUTED, NO trade recorded. If this happens on a "
            f"position < 30 min old, check wallet_address config vs "
            f"Gateway-loaded wallet (see POSTMORTEM 2026-04-24)."
        )
        self.db.record_event(
            "WARNING",
            "position",
            f"PRUNE STALE {pos.token.symbol} reason={reason} wallet_balance={wallet_balance_ui:.8f}",
        )

    async def _recover_phantom_sell(self, pos: "Position", reason: str,
                                     wallet_balance_ui: float,
                                     trigger_pnl_pct: float = 0.0,
                                     trigger_price_sol: Optional[float] = None,
                                     trigger_mid_price_sol: Optional[float] = None,
                                     trigger_mid_source: Optional[str] = None) -> bool:
        """BUG 2 FIX (2026-04-30): Recover from Gateway false-error 400.

        When Gateway returns 4xx but wallet is actually drained, the swap
        succeeded on-chain but Gateway response was misleading. Find the
        recent draining tx via RPC + record trade as success.

        Returns True if recovery succeeded, False to fall back to retry path.
        """
        mint = pos.token.mint_address
        try:
            # Find most recent signature for the wallet's token account
            if self.discovery is None:
                return False
            rpc = await self.discovery._get_rpc()
            if rpc is None:
                return False

            # Get token account address (we know it exists since wallet_balance check ran)
            ta_result = await rpc.get_token_accounts_by_owner(
                self.config.wallet_address, mint)
            ta_accs = (ta_result or {}).get("value", [])
            if not ta_accs:
                logger.error(f"M3: [PHANTOM RECOVERY FAIL] {pos.token.symbol} — "
                             f"no token account found for {mint[:12]}...")
                return False
            token_account = ta_accs[0].get("pubkey")
            if not token_account:
                return False

            # Get last 2 signatures — most recent should be the sell
            sigs = await rpc.get_signatures_for_address(token_account, limit=2)
            sigs = sigs or []
            if not sigs:
                return False

            # The first signature is the most recent. Verify it's after our entry
            most_recent = sigs[0]
            sig = most_recent.get("signature")
            block_time = most_recent.get("blockTime") or 0
            if not sig or block_time < pos.entry_time:
                return False  # Not a post-entry tx

            # Get tx details to compute SOL received
            tx = await rpc.get_transaction(sig)
            if not tx or tx.get("meta", {}).get("err") is not None:
                return False  # Tx failed on-chain

            # Confirm tx ACTUALLY drained our token account (not unrelated tx)
            posts = tx.get("meta", {}).get("postTokenBalances", [])
            pre = tx.get("meta", {}).get("preTokenBalances", [])
            wallet = self.config.wallet_address
            our_post_amt = None
            our_pre_amt = None
            for p in posts:
                if p.get("mint") == mint and p.get("owner") == wallet:
                    ui = p.get("uiTokenAmount", {}).get("uiAmount")
                    our_post_amt = float(ui) if ui is not None else 0.0
            for p in pre:
                if p.get("mint") == mint and p.get("owner") == wallet:
                    ui = p.get("uiTokenAmount", {}).get("uiAmount")
                    our_pre_amt = float(ui) if ui is not None else 0.0
            if our_pre_amt is None or our_post_amt is None:
                return False
            if our_pre_amt - (our_post_amt or 0) < pos.token_amount * 0.5:
                # tx didn't drain our position substantially — skip
                return False

            # Compute SOL received from wallet's pre/post lamport balance
            accts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
            wallet_idx = None
            for i, a in enumerate(accts):
                pk = a.get("pubkey") if isinstance(a, dict) else a
                if pk == wallet:
                    wallet_idx = i
                    break
            if wallet_idx is None:
                return False
            pre_lamports = (tx.get("meta", {}).get("preBalances") or [])[wallet_idx]
            post_lamports = (tx.get("meta", {}).get("postBalances") or [])[wallet_idx]
            fee_lamports = tx.get("meta", {}).get("fee", 0)
            sol_received = (post_lamports - pre_lamports + fee_lamports) / 1e9
            if sol_received <= 0:
                return False
            exit_price = sol_received / pos.token_amount if pos.token_amount > 0 else 0
            pnl_sol = sol_received - pos.sol_invested
            pnl_usd = pnl_sol * (self._sol_price_usd or 0)

            # Record trade
            record = TradeRecord(
                token_symbol=pos.token.symbol,
                mint_address=mint,
                entry_price_sol=pos.entry_price_sol,
                exit_price_sol=exit_price,
                token_amount=pos.token_amount,
                sol_invested=pos.sol_invested,
                sol_received=sol_received,
                pnl_sol=pnl_sol,
                pnl_usd=pnl_usd,
                hold_seconds=pos.hold_seconds(),
                exit_reason=f"{reason}_recovered",
                entry_tx=pos.entry_tx,
                exit_tx=sig,
                model_score=pos.model_score,
                entry_time=pos.entry_time,
                peak_pnl_pct=pos.peak_pnl_pct,
                sol_price_usd=self._sol_price_usd,
                trigger_pnl_pct=trigger_pnl_pct,
                m2_ref_price_sol=pos.m2_ref_price_sol,
                preflight_latency_ms=pos.preflight_latency_ms,
                pool_liq_at_entry_usd=pos.pool_liq_at_entry_usd,
                pool_sol_reserve_at_entry=getattr(pos, "pool_sol_reserve_at_entry", None),
                slippage_per_pool_ratio=getattr(pos, "slippage_per_pool_ratio", None),
            )
            self._trade_log.append(record)
            if len(self._trade_log) > self._trade_log_max:
                self._trade_log = self._trade_log[-self._trade_log_max:]
            self.risk.record_trade(pnl_usd)
            self.db.record_trade(record, features=pos.features,
                                 entry_source=getattr(pos, "entry_source", ""))

            # BUG 3 FIX: mark position as removed BEFORE removing to prevent re-save races
            pos._removed = True
            if pos in self._positions:
                self._positions.remove(pos)
            self.db.remove_position(mint)
            if self._grpc_stream is not None and pos.token.pool_address:
                self._grpc_stream.unwatch_pool(pos.token.pool_address)
            self._sell_retry_after.pop(mint, None)
            self._sell_retry_count.pop(mint, None)
            self._sell_retry_notice_after.pop(mint, None)
            self._exit_signal_reason.pop(mint, None)
            self._cleanup_swap_keepalive(mint, reason=f"phantom_sell_recovered ({reason})")

            pnl_pct_disp = (pnl_sol / pos.sol_invested) if pos.sol_invested > 0 else 0.0
            logger.info(
                f"M3: [PHANTOM RECOVERED] {pos.token.symbol} — sol_received={sol_received:.6f}, "
                f"pnl={pnl_pct_disp:+.2%} (${pnl_usd:+.2f}), exit_tx={sig[:32]}...")
            return True
        except Exception as e:
            logger.error(f"M3: [PHANTOM RECOVERY ERROR] {pos.token.symbol}: "
                         f"{type(e).__name__}: {e}")
            return False

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
        live = self._ev3m_live_audit.get(mint)
        if not live:
            return {
                "live_eval_present": False,
                "shadow_feature_source": shadow_feature_source,
                "shadow_ready_age_sec": ready_age_sec,
            }
        live_score = live.get("score")
        score_delta = None
        if live_score is not None and shadow_score is not None:
            score_delta = float(shadow_score) - float(live_score)
        return {
            "live_eval_present": True,
            "live_eval_status": live.get("status"),
            "live_score": live_score,
            "live_band": live.get("band"),
            "live_cutoff": live.get("cutoff"),
            "live_passed": live.get("passed"),
            "live_feature_source": live.get("feature_source"),
            "live_age_sec": live.get("age_sec"),
            "live_reason": live.get("reason"),
            "shadow_feature_source": shadow_feature_source,
            "shadow_ready_age_sec": ready_age_sec,
            "shadow_minus_live_score": score_delta,
        }

    def _log_ev3m_reconciliation(
        self,
        token: GraduatedToken,
        *,
        shadow_score: Optional[float],
        shadow_feature_source: str,
        ready_age_sec: float,
    ) -> None:
        live = self._ev3m_live_audit.get(token.mint_address)
        if not live:
            logger.info(
                f"EV3M-RECON: {token.symbol} live=missing "
                f"shadow={'n/a' if shadow_score is None else f'{shadow_score:.4f}'} "
                f"shadow_src={shadow_feature_source} ready_age={ready_age_sec:.1f}s"
            )
            return
        live_score = live.get("score")
        if live_score is None or shadow_score is None:
            logger.info(
                f"EV3M-RECON: {token.symbol} "
                f"live_status={live.get('status')} live_score={'n/a' if live_score is None else f'{live_score:.4f}'} "
                f"live_src={live.get('feature_source')} live_reason={live.get('reason')} "
                f"shadow={'n/a' if shadow_score is None else f'{shadow_score:.4f}'} "
                f"shadow_src={shadow_feature_source} "
                f"live_age={float(live.get('age_sec') or 0.0):.1f}s ready_age={ready_age_sec:.1f}s"
            )
            return
        delta = float(shadow_score) - float(live_score)
        logger.info(
            f"EV3M-RECON: {token.symbol} "
            f"live={live_score:.4f} src={live.get('feature_source')} status={live.get('status')} "
            f"shadow={shadow_score:.4f} src={shadow_feature_source} "
            f"delta={delta:+.4f} live_age={float(live.get('age_sec') or 0.0):.1f}s "
            f"ready_age={ready_age_sec:.1f}s"
        )

    async def _collect_shadow_features(self, token: GraduatedToken) -> Tuple[Optional[Dict[str, float]], str]:
        """Collect first-3m feature view for live shadow evaluation only."""
        if not self.signal:
            return None, "no_model"

        kline_bars = None
        kline_shifted = False
        feature_source = "gmgn"
        if self.kline_builder and self.kline_builder.is_registered(token.mint_address):
            try:
                await self.kline_builder.poll_swaps(token.mint_address)
            except Exception:
                pass
            n_bars_needed = max(3, self.config.feature_delay_sec // 60)
            kline_bars, kline_shifted = self.kline_builder.build_kline(
                mint=token.mint_address,
                start_ts=int(token.graduation_time),
                n_bars=n_bars_needed,
                resolution=60,
                sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
            )
            if kline_bars:
                feature_source = "on-chain"

        features = await self.signal.collect_features(
            token, kline_bars=kline_bars, kline_shifted=kline_shifted)
        return features, feature_source

    async def _run_shadow_rule_first(self):
        """Run Phase 6.1 shadow rule-first evaluation without affecting trading."""
        if not self.config.shadow_rule_first_enabled or not self.signal or not self._shadow_pending:
            return

        now = time.time()
        completed = []
        timeout_sec = self.config.feature_delay_sec + 300
        for mint, payload in list(self._shadow_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time
            if age_sec < self.config.feature_delay_sec:
                continue

            payload["attempts"] += 1
            try:
                stage2_rule_name = self._shadow_stage2_rule_name()
                features, feature_source = await self._collect_shadow_features(token)
                if features is None:
                    if age_sec >= timeout_sec:
                        self.db.record_shadow_policy_eval(
                            obs_id=payload["obs_id"],
                            mint_address=token.mint_address,
                            symbol=token.symbol,
                            graduation_time=token.graduation_time,
                            feature_delay_sec=self.config.feature_delay_sec,
                            stage1_name="current_signal_proxy",
                            stage1_score=None,
                            stage1_threshold=None,
                            stage1_passed=None,
                            stage2_rule_name=stage2_rule_name,
                            stage2_passed=None,
                            feature_source=feature_source,
                            feature_view="canonical_3m_live_shadow",
                            range_1to3m=None,
                            drawdown_1to3m=None,
                            total_volume_3m=None,
                            total_trades_3m=None,
                            metadata={
                                "status": "insufficient_features",
                                "age_sec": age_sec,
                                "attempts": payload["attempts"],
                                "instant_buy_mode": self.config.instant_buy_mode,
                                "stage2_rule_params": self._shadow_stage2_rule_params(),
                            },
                        )
                        logger.info(f"SHADOW: {token.symbol} timed out waiting for 3m features")
                        completed.append(mint)
                    continue

                stage1_score = float(self.signal.predict_survival(features))
                stage1_threshold = float(getattr(self.signal, "threshold", 0.0))
                stage1_passed = stage1_score >= stage1_threshold
                range_1to3m = float(features.get("lookback_high_low", 0.0))
                drawdown_1to3m = float(features.get("cum_drawdown_t0_to_entry", 0.0))
                total_volume_3m = float(features.get("cum_volume_t0_to_entry", 0.0))
                total_trades_3m = float(features.get("cum_trades_t0_to_entry", 0.0))
                stage2_passed = self._shadow_stage2_pass(
                    stage2_rule_name=stage2_rule_name,
                    range_1to3m=range_1to3m,
                    drawdown_1to3m=drawdown_1to3m,
                )

                self.db.record_shadow_policy_eval(
                    obs_id=payload["obs_id"],
                    mint_address=token.mint_address,
                    symbol=token.symbol,
                    graduation_time=token.graduation_time,
                    feature_delay_sec=self.config.feature_delay_sec,
                    stage1_name="current_signal_proxy",
                    stage1_score=stage1_score,
                    stage1_threshold=stage1_threshold,
                    stage1_passed=stage1_passed,
                    stage2_rule_name=stage2_rule_name,
                    stage2_passed=stage2_passed,
                    feature_source=feature_source,
                    feature_view="canonical_3m_live_shadow",
                    range_1to3m=range_1to3m,
                    drawdown_1to3m=drawdown_1to3m,
                    total_volume_3m=total_volume_3m,
                    total_trades_3m=total_trades_3m,
                    metadata={
                        "age_sec": age_sec,
                        "attempts": payload["attempts"],
                        "instant_buy_mode": self.config.instant_buy_mode,
                        "stage2_rule_params": self._shadow_stage2_rule_params(),
                    },
                )
                logger.info(
                    f"SHADOW: {token.symbol} stage1={stage1_score:.3f}/{stage1_threshold:.3f} "
                    f"stage2={stage2_rule_name}:{'PASS' if stage2_passed else 'FAIL'} "
                    f"range={range_1to3m:.3f} dd={drawdown_1to3m:.3f}"
                )
                completed.append(mint)
            except Exception as e:
                logger.warning(f"SHADOW: failed for {token.symbol}: {e}")
                if age_sec >= timeout_sec:
                    completed.append(mint)

        for mint in completed:
            self._shadow_pending.pop(mint, None)

    def _shadow_ev3m_band_names(self) -> List[str]:
        raw = str(getattr(self.config, "shadow_ev3m_selection_bands", "") or "").strip()
        if not raw:
            return ["top10", "top15"]
        bands = [b.strip() for b in raw.split(",") if b.strip()]
        return bands or ["top10", "top15"]

    def _ev3m_entry_offset_sec(self, token: GraduatedToken, delay_sec: int) -> float:
        graduation_time = float(token.graduation_time)
        anchor_ts = math.floor(graduation_time / 60.0) * 60.0
        anchor_offset = graduation_time - anchor_ts
        entry_rel_minute = int(math.ceil((float(delay_sec) + anchor_offset) / 60.0))
        return float((-anchor_offset) + 60.0 * entry_rel_minute)

    def _shadow_ev3m_entry_offset_sec(self, token: GraduatedToken) -> float:
        return self._ev3m_entry_offset_sec(token, int(self.config.shadow_ev3m_entry_delay_sec))

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
        return await self._collect_ev3m_features(
            token=token,
            delay_sec=int(self.config.shadow_ev3m_entry_delay_sec),
            model=self.shadow_ev3m_model,
        )

    async def _run_shadow_ev3m(self):
        if not self.config.shadow_ev3m_enabled or self.shadow_ev3m_model is None or not self._shadow_ev3m_pending:
            return

        now = time.time()
        completed = []
        bands = self._shadow_ev3m_band_names()
        for mint, payload in list(self._shadow_ev3m_pending.items()):
            token = payload["token"]
            ready_age_sec = self._shadow_ev3m_entry_offset_sec(token)
            age_sec = now - token.graduation_time
            if age_sec < ready_age_sec:
                continue

            payload["attempts"] += 1
            timeout_sec = ready_age_sec + 300.0
            try:
                features, feature_source = await self._collect_shadow_ev3m_features(token)
                if features is None:
                    if age_sec >= timeout_sec:
                        recon_meta = self._build_ev3m_recon_metadata(
                            mint=mint,
                            shadow_score=None,
                            shadow_feature_source=feature_source,
                            ready_age_sec=ready_age_sec,
                        )
                        for band in bands:
                            cutoff = self.shadow_ev3m_model.selection_cutoffs.get(band)
                            self.db.record_shadow_policy_eval(
                                obs_id=payload["obs_id"],
                                mint_address=token.mint_address,
                                symbol=token.symbol,
                                graduation_time=token.graduation_time,
                                feature_delay_sec=int(round(ready_age_sec)),
                                stage1_name=self.shadow_ev3m_model.primary_score_name,
                                stage1_score=None,
                                stage1_threshold=cutoff,
                                stage1_passed=None,
                                stage2_rule_name=f"{self.shadow_ev3m_model.policy_name}:{self.shadow_ev3m_model.primary_score_name}_{band}",
                                stage2_passed=None,
                                feature_source=feature_source,
                                feature_view=f"{self.shadow_ev3m_model.feature_view}_live_shadow",
                                range_1to3m=None,
                                drawdown_1to3m=None,
                                total_volume_3m=None,
                                total_trades_3m=None,
                                metadata={
                                    "status": "insufficient_features",
                                    "policy_name": self.shadow_ev3m_model.policy_name,
                                    "selection_band": band,
                                    "selection_cutoff": cutoff,
                                    "primary_score_name": self.shadow_ev3m_model.primary_score_name,
                                    "micro_aug_enabled": False,
                                    "age_sec": age_sec,
                                    "attempts": payload["attempts"],
                                    "entry_delay_sec": int(self.config.shadow_ev3m_entry_delay_sec),
                                    "ready_age_sec": ready_age_sec,
                                    **recon_meta,
                                },
                            )
                        self._log_ev3m_reconciliation(
                            token,
                            shadow_score=None,
                            shadow_feature_source=feature_source,
                            ready_age_sec=ready_age_sec,
                        )
                        logger.info(f"SHADOW-EV3M: {token.symbol} timed out waiting for confirmed-entry features")
                        completed.append(mint)
                    continue

                score = float(self.shadow_ev3m_model.predict_score(features))
                recon_meta = self._build_ev3m_recon_metadata(
                    mint=mint,
                    shadow_score=score,
                    shadow_feature_source=feature_source,
                    ready_age_sec=ready_age_sec,
                )
                for band in bands:
                    cutoff = self.shadow_ev3m_model.selection_cutoffs.get(band)
                    if cutoff is None:
                        continue
                    passed = score >= cutoff
                    self.db.record_shadow_policy_eval(
                        obs_id=payload["obs_id"],
                        mint_address=token.mint_address,
                        symbol=token.symbol,
                        graduation_time=token.graduation_time,
                        feature_delay_sec=int(round(ready_age_sec)),
                        stage1_name=self.shadow_ev3m_model.primary_score_name,
                        stage1_score=score,
                        stage1_threshold=cutoff,
                        stage1_passed=passed,
                        stage2_rule_name=f"{self.shadow_ev3m_model.policy_name}:{self.shadow_ev3m_model.primary_score_name}_{band}",
                        stage2_passed=passed,
                        feature_source=feature_source,
                        feature_view=f"{self.shadow_ev3m_model.feature_view}_live_shadow",
                        range_1to3m=features.get("close_range_to_entry"),
                        drawdown_1to3m=features.get("close_drawdown_to_entry"),
                        total_volume_3m=features.get("total_volume_to_entry"),
                        total_trades_3m=features.get("total_trades_to_entry"),
                        metadata={
                            "status": "ok",
                            "policy_name": self.shadow_ev3m_model.policy_name,
                            "selection_band": band,
                            "selection_cutoff": cutoff,
                            "primary_score_name": self.shadow_ev3m_model.primary_score_name,
                            "predicted_net_pos_3m": score,
                            "predicted_policy_pass": passed,
                            "micro_aug_enabled": False,
                            "micro_candidate_pool_available": False,
                            "age_sec": age_sec,
                            "attempts": payload["attempts"],
                            "entry_delay_sec": int(self.config.shadow_ev3m_entry_delay_sec),
                            "entry_offset_sec": features.get("entry_offset_sec"),
                            "entry_relative_minute": features.get("entry_relative_minute"),
                            "hist_completed_postgrad_bars": features.get("hist_completed_postgrad_bars"),
                            **recon_meta,
                        },
                    )
                self._log_ev3m_reconciliation(
                    token,
                    shadow_score=score,
                    shadow_feature_source=feature_source,
                    ready_age_sec=ready_age_sec,
                )
                logger.info(
                    f"SHADOW-EV3M: {token.symbol} score={score:.4f} "
                    f"bands={','.join(bands)} ready_age={ready_age_sec:.1f}s"
                )
                completed.append(mint)
            except Exception as e:
                logger.warning(f"SHADOW-EV3M: failed for {token.symbol}: {e}")
                if age_sec >= timeout_sec:
                    completed.append(mint)

        for mint in completed:
            self._shadow_ev3m_pending.pop(mint, None)
            self._ev3m_live_audit.pop(mint, None)

    def _shadow_super_winner_event_band_specs(self) -> List[Tuple[str, str]]:
        specs = [
            (
                str(self.config.shadow_super_winner_event_default_band or "").strip() or "top3",
                str(self.config.shadow_super_winner_event_default_policy or "").strip() or "fixed_15m",
            ),
            (
                str(self.config.shadow_super_winner_event_fallback_band or "").strip() or "top7p5",
                str(self.config.shadow_super_winner_event_fallback_policy or "").strip() or "break30_time15",
            ),
        ]
        deduped: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()
        for spec in specs:
            if spec in seen:
                continue
            seen.add(spec)
            deduped.append(spec)
        return deduped

    def _shadow_super_winner_event_n_bars_needed(self, token: GraduatedToken) -> int:
        graduation_time = float(token.graduation_time)
        anchor_ts = math.floor(graduation_time / 60.0) * 60.0
        anchor_offset = graduation_time - anchor_ts
        max_delay = int(self.config.shadow_super_winner_event_scan_end_sec)
        max_rel_minute = int(math.ceil((float(max_delay) + anchor_offset) / 60.0))
        return max(3, max_rel_minute + 1)

    async def _collect_shadow_super_winner_event_view(
        self,
        token: GraduatedToken,
    ) -> Tuple[Optional[Dict[str, float]], str]:
        if self.shadow_super_winner_event_model is None:
            return None, "no_shadow_model"
        metadata = getattr(self.shadow_super_winner_event_model, "metadata", {}) or {}
        threshold_map = dict(metadata.get("threshold_map", {}))
        family_feature_map = dict(metadata.get("family_feature_map", {}))
        event_defs = dict(metadata.get("event_defs", {}))
        if not threshold_map or not family_feature_map or not event_defs:
            return None, "missing_shadow_metadata"
        if self.kline_builder is None:
            return None, "no_onchain_kline_builder"

        try:
            await self.kline_builder.poll_swaps(token.mint_address)
        except Exception:
            pass

        graduation_time = float(token.graduation_time)
        anchor_ts = int(math.floor(graduation_time / 60.0) * 60)
        n_bars_needed = self._shadow_super_winner_event_n_bars_needed(token)
        sol_price = self._sol_price_usd if self._sol_price_usd > 0 else 80.0
        feature_source = "onchain_1m_plus_onchain_swaps"
        kline_bars, shifted = self.kline_builder.build_kline(
            mint=token.mint_address,
            start_ts=anchor_ts,
            n_bars=n_bars_needed,
            resolution=60,
            sol_price_usd=sol_price,
        )
        if not kline_bars or shifted:
            gmgn_bars = await self._fetch_gmgn_obs_kline(
                token.mint_address,
                token.symbol,
                graduation_time,
            )
            if not gmgn_bars:
                return None, "insufficient_event_bars"
            kline_bars = gmgn_bars
            feature_source = "gmgn_1m_plus_onchain_swaps"

        view = evaluate_super_winner_event_shadow_view(
            kline_bars=kline_bars,
            swaps=self.kline_builder.get_swaps(token.mint_address),
            graduation_time=graduation_time,
            sol_price_usd=sol_price,
            scan_start_sec=int(self.config.shadow_super_winner_event_scan_start_sec),
            scan_end_sec=int(self.config.shadow_super_winner_event_scan_end_sec),
            scan_step_sec=int(self.config.shadow_super_winner_event_scan_step_sec),
            threshold_map=threshold_map,
            family_feature_map=family_feature_map,
            event_defs=event_defs,
            arbitration_rule=str(self.config.shadow_super_winner_event_arbitration_rule or "earliest_then_ab_then_score").strip()
            or "earliest_then_ab_then_score",
        )
        if view is None:
            return None, "no_valid_event"
        return view, feature_source

    async def _run_shadow_super_winner_event(self):
        if (
            not self.config.shadow_super_winner_event_enabled
            or self.shadow_super_winner_event_model is None
            or not self._shadow_super_winner_event_pending
        ):
            return

        now = time.time()
        completed: List[str] = []
        band_specs = self._shadow_super_winner_event_band_specs()
        scan_start = int(self.config.shadow_super_winner_event_scan_start_sec)
        scan_end = int(self.config.shadow_super_winner_event_scan_end_sec)
        timeout_sec = float(scan_end + 300)
        arbitration_rule = (
            str(self.config.shadow_super_winner_event_arbitration_rule or "earliest_then_ab_then_score").strip()
            or "earliest_then_ab_then_score"
        )
        for mint, payload in list(self._shadow_super_winner_event_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time
            if age_sec < float(scan_start):
                continue

            payload["attempts"] += 1
            try:
                view, feature_source = await self._collect_shadow_super_winner_event_view(token)
                if view is None:
                    if age_sec >= timeout_sec:
                        for band, policy_name in band_specs:
                            cutoff = self.shadow_super_winner_event_model.selection_cutoffs.get(band)
                            self.db.record_shadow_policy_eval(
                                obs_id=payload["obs_id"],
                                mint_address=token.mint_address,
                                symbol=token.symbol,
                                graduation_time=token.graduation_time,
                                feature_delay_sec=scan_start,
                                stage1_name=self.shadow_super_winner_event_model.primary_score_name,
                                stage1_score=None,
                                stage1_threshold=cutoff,
                                stage1_passed=None,
                                stage2_rule_name=f"super_winner_event:none:{arbitration_rule}:{band}:{policy_name}",
                                stage2_passed=None,
                                feature_source=feature_source,
                                feature_view=self.shadow_super_winner_event_model.feature_view,
                                range_1to3m=None,
                                drawdown_1to3m=None,
                                total_volume_3m=None,
                                total_trades_3m=None,
                                metadata={
                                    "status": feature_source,
                                    "selection_band": band,
                                    "policy_name": policy_name,
                                    "arbitration_rule": arbitration_rule,
                                    "age_sec": age_sec,
                                    "attempts": payload["attempts"],
                                },
                            )
                        logger.info(f"SHADOW-SW-EVENT: {token.symbol} timed out ({feature_source})")
                        completed.append(mint)
                    continue

                if int(float(view.get("micro_overlay_available_flag", 0.0))) != 1:
                    if age_sec >= timeout_sec:
                        for band, policy_name in band_specs:
                            cutoff = self.shadow_super_winner_event_model.selection_cutoffs.get(band)
                            self.db.record_shadow_policy_eval(
                                obs_id=payload["obs_id"],
                                mint_address=token.mint_address,
                                symbol=token.symbol,
                                graduation_time=token.graduation_time,
                                feature_delay_sec=int(round(float(view.get("event_delay_sec", scan_start)))),
                                stage1_name=self.shadow_super_winner_event_model.primary_score_name,
                                stage1_score=None,
                                stage1_threshold=cutoff,
                                stage1_passed=None,
                                stage2_rule_name=(
                                    f"super_winner_event:{view.get('event_name', 'none')}:"
                                    f"{arbitration_rule}:{band}:{policy_name}"
                                ),
                                stage2_passed=None,
                                feature_source=feature_source,
                                feature_view=self.shadow_super_winner_event_model.feature_view,
                                range_1to3m=None,
                                drawdown_1to3m=None,
                                total_volume_3m=None,
                                total_trades_3m=None,
                                metadata={
                                    "status": "micro_overlay_unavailable",
                                    "event_name": view.get("event_name"),
                                    "event_delay_sec": view.get("event_delay_sec"),
                                    "event_entry_relative_minute": view.get("event_entry_relative_minute"),
                                    "selection_band": band,
                                    "policy_name": policy_name,
                                    "arbitration_rule": arbitration_rule,
                                    "overlay_covered": False,
                                    "micro_overlay_available_flag": int(float(view.get("micro_overlay_available_flag", 0.0))),
                                    "raw_swap_row_covered": int(float(view.get("raw_swap_row_covered", 0.0))),
                                    "micro_overlay_non_null_count": view.get("micro_overlay_non_null_count"),
                                    "age_sec": age_sec,
                                    "attempts": payload["attempts"],
                                },
                            )
                        logger.info(f"SHADOW-SW-EVENT: {token.symbol} timed out waiting for micro overlay")
                        completed.append(mint)
                    continue

                features = {
                    feature_name: view.get(feature_name, np.nan)
                    for feature_name in self.shadow_super_winner_event_model.feature_names
                }
                score = float(self.shadow_super_winner_event_model.predict_score(features))
                event_name = str(view.get("event_name", "event_ab"))
                event_delay_sec = int(round(float(view.get("event_delay_sec", scan_start))))
                for band, policy_name in band_specs:
                    cutoff = self.shadow_super_winner_event_model.selection_cutoffs.get(band)
                    if cutoff is None:
                        continue
                    passed = score >= cutoff
                    self.db.record_shadow_policy_eval(
                        obs_id=payload["obs_id"],
                        mint_address=token.mint_address,
                        symbol=token.symbol,
                        graduation_time=token.graduation_time,
                        feature_delay_sec=event_delay_sec,
                        stage1_name=self.shadow_super_winner_event_model.primary_score_name,
                        stage1_score=score,
                        stage1_threshold=cutoff,
                        stage1_passed=passed,
                        stage2_rule_name=f"super_winner_event:{event_name}:{arbitration_rule}:{band}:{policy_name}",
                        stage2_passed=passed,
                        feature_source=feature_source,
                        feature_view=self.shadow_super_winner_event_model.feature_view,
                        range_1to3m=None,
                        drawdown_1to3m=None,
                        total_volume_3m=None,
                        total_trades_3m=None,
                        metadata={
                            "status": "ok",
                            "event_name": event_name,
                            "event_delay_sec": event_delay_sec,
                            "event_entry_relative_minute": view.get("event_entry_relative_minute"),
                            "event_entry_ret_from_grad": view.get("event_entry_ret_from_grad"),
                            "arbitration_rule": arbitration_rule,
                            "selection_band": band,
                            "policy_name": policy_name,
                            "overlay_covered": True,
                            "micro_overlay_available_flag": 1,
                            "raw_swap_row_covered": int(float(view.get("raw_swap_row_covered", 0.0))),
                            "feature_source": feature_source,
                            "predicted_strong_winner_event": score,
                            "absorption__pass_frac_for__super_winner": view.get("absorption__pass_frac_for__super_winner"),
                            "breadth__pass_frac_for__super_winner": view.get("breadth__pass_frac_for__super_winner"),
                            "false_leader_rejection__pass_frac_for__super_winner": view.get(
                                "false_leader_rejection__pass_frac_for__super_winner"
                            ),
                            "persistence__pass_frac_for__super_winner": view.get("persistence__pass_frac_for__super_winner"),
                            "event_ab__flag": int(float(view.get("event_ab__flag", 0.0))),
                            "event_abp__flag": int(float(view.get("event_abp__flag", 0.0))),
                            "micro_overlay_non_null_count": view.get("micro_overlay_non_null_count"),
                            "age_sec": age_sec,
                            "attempts": payload["attempts"],
                        },
                    )
                logger.info(
                    f"SHADOW-SW-EVENT: {token.symbol} event={event_name} delay={event_delay_sec}s "
                    f"score={score:.4f} bands={','.join(b for b, _ in band_specs)}"
                )
                completed.append(mint)
            except Exception as e:
                logger.warning(f"SHADOW-SW-EVENT: failed for {token.symbol}: {e}")
                if age_sec >= timeout_sec:
                    completed.append(mint)

        for mint in completed:
            self._shadow_super_winner_event_pending.pop(mint, None)

    def _shadow_stage2_rule_name(self) -> str:
        rule_name = str(getattr(self.config, "shadow_rule_stage2_name", "") or "").strip()
        return rule_name or "drawdown_1to3m_ge_q60"

    def _shadow_stage2_rule_params(self) -> dict:
        return {
            "rule_name": self._shadow_stage2_rule_name(),
            "drawdown_1to3m_min": float(self.config.shadow_rule_drawdown_1to3m_min),
            "range_1to3m_max": float(self.config.shadow_rule_range_1to3m_max),
        }

    def _shadow_stage2_pass(self, stage2_rule_name: str, range_1to3m: float, drawdown_1to3m: float) -> bool:
        if stage2_rule_name == "drawdown_1to3m_ge_q60":
            return drawdown_1to3m >= self.config.shadow_rule_drawdown_1to3m_min
        if stage2_rule_name == "stable_shape_q60_q40":
            return (
                range_1to3m <= self.config.shadow_rule_range_1to3m_max and
                drawdown_1to3m >= self.config.shadow_rule_drawdown_1to3m_min
            )
        logger.warning(
            f"Unknown shadow_stage2_rule_name={stage2_rule_name}; "
            "falling back to drawdown_1to3m_ge_q60 semantics"
        )
        return drawdown_1to3m >= self.config.shadow_rule_drawdown_1to3m_min

    async def _run_shadow_vshape(self):
        """Run V-shape T+10m shadow evaluation."""
        if not self.config.shadow_vshape_enabled or self.shadow_vshape_model is None or not self._shadow_vshape_pending:
            return

        now = time.time()
        completed: List[str] = []
        # Phase 15a (2026-04-26): entry delay sourced from model so v1.4r (600)
        # and v1.6 (300) both work. n_bars adjusted accordingly.
        entry_delay = int(self.shadow_vshape_model.entry_delay_sec) if self.shadow_vshape_model else 600
        window_min = max(1, entry_delay // 60)
        timeout_sec = entry_delay + 300

        for mint, payload in list(self._shadow_vshape_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time
            if age_sec < entry_delay:
                continue

            payload["attempts"] += 1
            try:
                # Build 1m bars: window_min bars + 2-bar buffer
                n_bars = max(window_min + 2, 7)
                kline_bars, kline_shifted = self.kline_builder.build_kline(
                    mint=token.mint_address,
                    start_ts=int(token.graduation_time),
                    n_bars=n_bars,
                    resolution=60,
                    sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
                )

                if not kline_bars or len(kline_bars) < 3:
                    if age_sec >= timeout_sec:
                        logger.info(f"VSHAPE: {token.symbol} timed out (no bars)")
                        completed.append(mint)
                    continue

                # Detect V-shape pattern
                vf = detect_vshape_live(kline_bars, token.graduation_time,
                                        entry_offset_sec=entry_delay)
                if vf is None:
                    if age_sec >= timeout_sec:
                        logger.info(f"VSHAPE: {token.symbol} timed out (detect failed)")
                        completed.append(mint)
                    continue

                # Compute micro features from swaps
                swaps_raw = self.db.get_swaps_for_token(token.mint_address)
                micro = compute_micro_live(swaps_raw, token.graduation_time,
                                            window_sec=entry_delay,
                                            feat_prefix=f"m{window_min}")

                # Compute smart money features
                sm = self.db.compute_smart_money_features(token.mint_address)

                # Compute time_span (creation → graduation) from GMGN info
                # MemeTrans paper: strongest pre-migration predictor (AUC 0.658)
                time_span_sec = 0.0
                gmgn_info = getattr(token, "_gmgn_info_raw", None) or {}
                creation_ts = float(gmgn_info.get("creation_timestamp") or
                                    gmgn_info.get("open_timestamp") or 0)
                # Fallback: try gmgn_info_t0 from observation DB
                if creation_ts <= 0:
                    obs_info = self.db.get_observation_gmgn_t0(token.mint_address)
                    if obs_info:
                        creation_ts = float(obs_info.get("creation_timestamp") or
                                            obs_info.get("open_timestamp") or 0)
                if creation_ts > 0 and token.graduation_time > creation_ts:
                    time_span_sec = token.graduation_time - creation_ts

                # Merge all features
                features = {}
                features.update(vf)
                features.update(micro)
                features.update(sm)

                # Score
                score = self.shadow_vshape_model.predict_score(features)
                pattern_name = (
                    "vshape" if vf.get("is_vshape") else
                    "steady_up" if vf.get("is_steady_up") else
                    "near_high" if vf.get("is_near_high") else
                    "reversal" if vf.get("is_reversal") else
                    "none")

                # Log for each band
                bands_raw = str(self.config.shadow_vshape_selection_bands or "").strip()
                bands = [b.strip() for b in bands_raw.split(",") if b.strip()]
                for band in bands:
                    cutoff = self.shadow_vshape_model.selection_cutoffs.get(band)
                    if cutoff is None:
                        continue
                    passed = score >= cutoff and vf.get("any_pattern", 0) == 1

                    self.db.record_shadow_policy_eval(
                        obs_id=payload["obs_id"],
                        mint_address=token.mint_address,
                        symbol=token.symbol,
                        graduation_time=token.graduation_time,
                        feature_delay_sec=entry_delay,
                        stage1_name="vshape_v1_score",
                        stage1_score=score,
                        stage1_threshold=cutoff,
                        stage1_passed=passed,
                        stage2_rule_name=f"vshape_pattern:{pattern_name}",
                        stage2_passed=vf.get("any_pattern", 0) == 1,
                        feature_source="on-chain+swaps",
                        feature_view="vshape_t10m_shadow",
                        range_1to3m=vf.get("vf_range_total", 0),
                        drawdown_1to3m=vf.get("vf_drawdown_from_peak", 0),
                        total_volume_3m=micro.get("m10_buy_vol_total", 0),
                        total_trades_3m=micro.get("m10_unique_buyers", 0),
                        metadata={
                            "pattern": pattern_name,
                            "vf_entry_ret": vf.get("vf_entry_ret"),
                            "vf_peak_ret": vf.get("vf_peak_ret"),
                            "vf_recovery_pct": vf.get("vf_recovery_pct"),
                            "selection_band": band,
                            "sm_smart_vs_dumb_vol": sm.get("sm_smart_vs_dumb_vol"),
                            "m10_buyer_hhi": micro.get("m10_buyer_hhi"),
                            "time_span_sec": time_span_sec,
                            "time_span_min": round(time_span_sec / 60, 1),
                            "early_crash_pct": self.config.vshape_early_crash_pct,
                            "early_crash_window_sec": self.config.vshape_early_crash_window_sec,
                            "model_version": self.shadow_vshape_model.version,
                        },
                    )

                logger.info(
                    f"VSHAPE: {token.symbol} pattern={pattern_name} "
                    f"score={score:.3f} entry_ret={vf.get('vf_entry_ret', 0):.3f} "
                    f"peak={vf.get('vf_peak_ret', 0):.3f} "
                    f"recovery={vf.get('vf_recovery_pct', 0):.3f} "
                    f"time_span={time_span_sec:.0f}s"
                )
                completed.append(mint)

            except Exception as e:
                logger.warning(f"VSHAPE: failed for {token.symbol}: {e}")
                if age_sec >= timeout_sec:
                    completed.append(mint)

        for mint in completed:
            self._shadow_vshape_pending.pop(mint, None)

    async def _run_shadow_event(self):
        """Phase B event-triggered scanner (shadow only).

        Scans each pending token every SCAN_STEP_SEC seconds between
        SCAN_START_SEC and SCAN_END_SEC post-graduation. On first trigger
        (pattern AND score >= cutoff), records a row to `shadow_event_evals`
        and stops scanning that token. Does NOT gate real trades.
        """
        if (not self.config.shadow_event_enabled
                or self.flow_model is None
                or not self._shadow_event_pending):
            return

        cfg = self.config
        cutoff_band = cfg.shadow_event_cutoff_band
        cutoff_value = self.flow_model.selection_cutoffs.get(cutoff_band)
        if cutoff_value is None:
            logger.warning(
                f"EVENT-SHADOW: cutoff band {cutoff_band!r} not in model; "
                f"available: {sorted(self.flow_model.selection_cutoffs.keys())}")
            return

        now = time.time()
        completed: List[str] = []
        for mint, payload in list(self._shadow_event_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time

            # Past scan window or already triggered — schedule cleanup and skip.
            if payload["triggered"] or age_sec > cfg.shadow_event_scan_end_sec + 60:
                completed.append(mint)
                continue
            if age_sec < cfg.shadow_event_scan_start_sec:
                continue

            # Rate-limit scanning: only fire once per SCAN_STEP_SEC window.
            target_scan = int(age_sec) - (int(age_sec) % cfg.shadow_event_scan_step_sec)
            if target_scan <= payload["last_scan_sec"]:
                continue
            payload["last_scan_sec"] = target_scan

            try:
                # Build on-chain 1m kline up to target_scan
                n_bars = max(3, target_scan // 60 + 2)
                kline_bars, _ = self.kline_builder.build_kline(
                    mint=mint,
                    start_ts=int(token.graduation_time),
                    n_bars=n_bars, resolution=60,
                    sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
                )
                if not kline_bars or len(kline_bars) < 3:
                    continue
                vf = detect_pattern_at_t(kline_bars, token.graduation_time, target_scan)
                if vf is None or vf.get("any_pattern", 0) != 1:
                    continue
                swaps_raw = self.db.get_swaps_for_token(mint)
                micro = compute_micro_at_t(
                    swaps_raw, token.graduation_time, target_scan,
                    swap_data_max_sec=self.flow_model.swap_data_max_sec)
                if not micro:
                    continue
                features = {**vf, **micro}
                score = self.flow_model.predict_score(features)

                if score >= cutoff_value:
                    self.db.record_shadow_event_eval(
                        mint_address=mint, symbol=token.symbol,
                        graduation_time=token.graduation_time,
                        scan_t_sec=target_scan,
                        cutoff_band=cutoff_band, cutoff_value=cutoff_value,
                        score=score, pattern=vf["pattern"],
                        entry_price_sol=vf.get("entry_price"),
                        sol_price_usd=self._sol_price_usd,
                        model_version=self.flow_model.version,
                        features=features)
                    logger.info(
                        f"EVENT-SHADOW: {token.symbol} trigger at T+{target_scan//60}m "
                        f"score={score:.4f} cutoff={cutoff_band}={cutoff_value:.4f} "
                        f"pattern={vf['pattern']}")
                    payload["triggered"] = True
            except Exception as e:
                logger.debug(f"EVENT-SHADOW: {token.symbol} scan failed at t={target_scan}s: {e}")

        for mint in completed:
            self._shadow_event_pending.pop(mint, None)

    async def _run_shadow_big_winner(self):
        """Phase 24 BigWinner v2 entry-side filter.

        Scans each pending token between SCAN_START_SEC and SCAN_END_SEC
        post-graduation, every SCAN_STEP_SEC seconds. On first PASS
        (p_big >= cutoff), records to `shadow_big_winner_evals`.

        - SHADOW mode (`big_winner_shadow_only=true`): log only.
        - LIVE mode: create TradeCandidate(entry_source=cfg.big_winner_entry_source,
          position_size_usd=cfg.big_winner_position_size_usd) and append to
          `_candidate_queue`; M3 buy loop picks it up. Token-level cooldown
          (`_big_winner_cooldown[mint]=now+600s`) blocks duplicate entries.
          After `big_winner_canary_trade_limit` realized trades, auto-revert
          to shadow regardless of yml flag.

        See the Phase 24 deploy notes for current cutoff/model selection.
        """
        if (not self._enabled_any(self.config, "big_winner_enabled", "big_winner_v1_enabled")
                or not self.big_winner_loaded
                or not self._big_winner_pending):
            return

        cfg = self.config
        # Phase 25g-fix-2 (2026-05-03): Pydantic v2 validate_assignment can
        # reset Optional[*] canonical fields back to None at runtime even
        # after _normalize_model_config_aliases set them in __init__.
        # Read every canonical big_winner_* field through _cfg_value with
        # legacy big_winner_v1_* fallback. Without this, `float(None)` on
        # line 3985 fired every heartbeat, silently killing the scan loop
        # and leaving zero entries for hours.
        cutoff = float(self._cfg_value(
            cfg, "big_winner_cutoff", "big_winner_v1_cutoff", default=0.50))
        canary_limit = int(self._cfg_value(
            cfg, "big_winner_canary_trade_limit",
            "big_winner_v1_canary_trade_limit", default=0))
        scan_start = int(self._cfg_value(
            cfg, "big_winner_scan_start_sec",
            "big_winner_v1_scan_start_sec", default=180))
        scan_end = int(self._cfg_value(
            cfg, "big_winner_scan_end_sec",
            "big_winner_v1_scan_end_sec", default=600))
        scan_step = int(self._cfg_value(
            cfg, "big_winner_scan_step_sec",
            "big_winner_v1_scan_step_sec", default=30))
        shadow_only_flag = bool(self._cfg_value(
            cfg, "big_winner_shadow_only",
            "big_winner_v1_shadow_only", default=True))
        now = time.time()
        completed: List[str] = []

        # Effective shadow gate — yml flag OR canary cap reached.
        canary_cap_reached = (
            canary_limit > 0
            and self._big_winner_trade_count >= canary_limit
        )
        effective_shadow_only = shadow_only_flag or canary_cap_reached

        for mint, payload in list(self._big_winner_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time

            # Defensive: drop tokens with future graduation_time (clock skew /
            # corrupt data) — feature compute on negative scan_t would silently
            # produce garbage scores.
            if age_sec < 0:
                logger.warning(
                    f"BIG-WINNER: {token.symbol} graduation_time in the FUTURE "
                    f"(age={age_sec:.0f}s) — dropping from pending")
                completed.append(mint)
                continue

            # Past scan window or already triggered
            if payload["triggered"] or age_sec > scan_end + 60:
                completed.append(mint)
                continue
            if age_sec < scan_start:
                continue

            # Rate-limit: one scan per SCAN_STEP_SEC window
            target_scan = int(age_sec) - (int(age_sec) % scan_step)
            if target_scan <= payload["last_scan_sec"]:
                continue
            payload["last_scan_sec"] = target_scan

            try:
                # Load swap data as DataFrame (forward-only enforced inside predict)
                swaps_df = self.db.get_swaps_df_for_token(mint)
                if swaps_df is None or len(swaps_df) < 8:
                    continue

                from controllers.generic.big_winner_inference import (
                    predict_big_winner_with_features)
                scan_t_abs = int(token.graduation_time + target_scan)
                # Phase 25h (2026-05-03): persist features alongside score so
                # sim/live alignment audits can reproduce decisions exactly,
                # without re-deriving features from drifted swap state.
                p_big, feats = predict_big_winner_with_features(
                    swaps_df, scan_t_abs, int(token.graduation_time))
                if p_big is None:
                    continue

                decision_pass = (p_big >= cutoff)

                features_json = None
                if feats is not None:
                    try:
                        import json as _json
                        features_json = _json.dumps(feats, separators=(",", ":"))
                    except Exception:
                        features_json = None

                self.db.record_shadow_big_winner_eval(
                    mint_address=mint, symbol=token.symbol,
                    graduation_time=token.graduation_time,
                    scan_t_sec=target_scan,
                    cutoff_value=cutoff, score=p_big,
                    decision_pass=decision_pass,
                    n_swaps=len(swaps_df),
                    entry_price_sol=None,  # populated at trade time if LIVE
                    sol_price_usd=self._sol_price_usd,
                    model_version=self._big_winner_model_version,
                    features_json=features_json,
                )

                if not decision_pass:
                    logger.debug(
                        f"BIG-WINNER-SHADOW: {token.symbol} score={p_big:.4f} "
                        f"< cutoff {cutoff:.4f} (T+{target_scan//60}m)")
                    continue

                # PASS branch — log + (optionally) wire to M3 entry queue
                if effective_shadow_only:
                    reason = "shadow_only=true" if shadow_only_flag \
                        else f"canary_cap_reached ({self._big_winner_trade_count}/{canary_limit})"
                    logger.info(
                        f"BIG-WINNER-SHADOW: {token.symbol} PASS at "
                        f"T+{target_scan//60}m score={p_big:.4f} >= {cutoff:.4f} "
                        f"[{reason} — no trade]")
                    payload["triggered"] = True
                    continue

                # LIVE entry trigger (Phase 16.3 Step 3) — guard against
                # duplicates and respect token-level cooldown.
                if any(p.token.mint_address == mint for p in self._positions):
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} PASS at T+{target_scan//60}m "
                        f"score={p_big:.4f} [skipped — already has open position]")
                    payload["triggered"] = True
                    continue

                cooldown_until = self._big_winner_cooldown.get(mint, 0.0)
                if cooldown_until > now:
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} PASS but skipped "
                        f"(token cooldown {cooldown_until - now:.0f}s remaining)")
                    payload["triggered"] = True
                    continue

                if any(c.token.mint_address == mint for c in self._candidate_queue):
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} PASS but skipped "
                        f"(already in candidate queue from another source)")
                    payload["triggered"] = True
                    continue

                # Phase 16.3 Step 3 (post-canary-2 fix) — stop-loss blacklist.
                # V-shape's `_evaluate_token` checks `_stoploss_blacklist`
                # before evaluation; big_winner must enforce the same guard
                # to prevent re-entering a token that just stopped out.
                bl_until = self._stoploss_blacklist.get(mint)
                if bl_until and now < bl_until:
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} PASS but skipped "
                        f"— stop-loss blacklisted ({bl_until - now:.0f}s remaining)")
                    payload["triggered"] = True
                    continue

                # RugFilter v3 entry-gate REMOVED 2026-05-06.
                # No rug-side gate active for big_winner during v4 shadow phase.

                # Phase 16.3 Step 3 (post-canary-1 fix) — apply M2 hard-reject
                # filters BEFORE enqueueing. Without this, big_winner buys
                # crashed/rugged tokens that V-shape's `_evaluate_token` would
                # have rejected (BTC -83% / 1000x cum_dd -92% / ScamGPT
                # examples from 2026-04-27 canary).
                try:
                    kline_bars, kline_source, m2_features = (
                        await self._build_m2_kline_for_safety(token))
                except Exception as e:
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} M2 kline build failed: {e} — skip")
                    payload["triggered"] = True
                    continue
                safety_ok, reject_reason = self._apply_m2_safety_filters(
                    token, m2_features, kline_source, source=self._big_winner_entry_source)
                if not safety_ok:
                    logger.info(
                        f"BIG-WINNER-LIVE: {token.symbol} PASS p_big={p_big:.4f} "
                        f"but REJECTED by M2 safety: {reject_reason} [no trade]")
                    self.db.record_event(
                        "INFO", "big_winner",
                        f"REJECT {token.symbol} score={p_big:.4f} "
                        f"src={kline_source} reason={reject_reason}")
                    payload["triggered"] = True
                    continue
                logger.info(
                    f"BIG-WINNER-LIVE: {token.symbol} M2 safety ✓ "
                    f"({kline_source}, ret_5m={m2_features.get('5min_return', m2_features.get('cum_return_t0_to_entry', 0)):+.1%}, "
                    f"vol/liq={(m2_features.get('5min_volume_usd', m2_features.get('cum_volume_t0_to_entry', 0)) / max(token.liquidity_usd, 1)):.2f})")

                # Build candidate and enqueue. M3 loop will execute via
                # the shared `_execute_buy` path; per-source sizing handled
                # inside `_execute_buy` via candidate.position_size_usd.
                from controllers.generic.meme_sniper_utils import TradeCandidate
                bw_size = float(self._cfg_value(
                    cfg, "big_winner_position_size_usd",
                    "big_winner_v1_position_size_usd", default=3.0))
                candidate = TradeCandidate(
                    token=token,
                    model_score=float(p_big),
                    features={
                        "big_winner_score": float(p_big),
                        "big_winner_scan_t_sec": int(target_scan),
                        "big_winner_cutoff": float(cutoff),
                        "entry_source": self._big_winner_entry_source,
                        # Carry M2 safety features forward (used by exit-side
                        # analysis to compare BW-selected vs VS-selected
                        # token quality)
                        "m2_kline_source": kline_source,
                    },
                    queued_at=now,
                    last_swap_price_sol=0.0,
                    entry_source=self._big_winner_entry_source,
                    position_size_usd=bw_size,
                )
                self._candidate_queue.append(candidate)
                self._big_winner_cooldown[mint] = now + float(self._cfg_value(
                    cfg, "big_winner_token_cooldown_sec",
                    "big_winner_v1_token_cooldown_sec", default=600))
                payload["triggered"] = True

                logger.info(
                    f"BIG-WINNER-LIVE: {token.symbol} PASS at T+{target_scan//60}m "
                    f"score={p_big:.4f} >= {cutoff:.4f}, sizing=${bw_size:.2f} "
                    f"[queued for M3; canary {self._big_winner_trade_count}/"
                    f"{canary_limit if canary_limit > 0 else '∞'}]")
            except Exception as e:
                # Phase 25g-fix-2 (2026-05-03): bumped DEBUG → WARNING. The
                # silent DEBUG was hiding the `float(None)` regression for
                # hours (0 evals between 10:19 and 14:48 UTC). A WARNING
                # is loud enough to surface in normal log scans.
                logger.warning(
                    f"BIG-WINNER: {token.symbol} scan failed at t={target_scan}s: {e}")

        for mint in completed:
            self._big_winner_pending.pop(mint, None)

    async def _rug_event_watcher(self):
        """Consume rug events from gRPC stream and fire immediate panic sell.

        Runs as a dedicated task (not polled from heartbeat) so the latency
        from on-chain dump → panic sell is minimized — the goal is to slip
        our sell tx into the same block (or the very next one) as the rug
        dump itself. Each event is logged to `rug_events` for post-hoc
        comparison with M4's poll-based detection.
        """
        while True:
            try:
                event = await self._rug_event_queue.get()
            except asyncio.CancelledError:
                return
            try:
                mint = event["mint_address"]
                pool = event["pool_address"]
                sol_amount = float(event["sol_amount"])
                price_sol = float(event["price_sol"])
                detected_at = float(event["detected_at"])

                # Find the position
                pos = next((p for p in self._positions
                            if p.token.mint_address == mint), None)
                if pos is None:
                    # Position already closed (race with normal exit) — log only
                    self.db.record_rug_event(
                        mint_address=mint, pool_address=pool,
                        sol_amount=sol_amount, price_sol=price_sol,
                        detected_at=detected_at, action_taken="no_position",
                        signature=event.get("signature"))
                    if self._grpc_stream:
                        self._grpc_stream.unwatch_pool(pool)
                    continue

                # Compute PnL at the rug price for logging
                entry = pos.entry_price_sol
                pnl_pct = (price_sol - entry) / entry if entry > 0 else 0.0

                # Phase 15d 2026-04-27 pnl gate: only fire panic sell when
                # position is at or below `rug_event_exit_max_pnl_pct` (default
                # 0.0 = "only when losing"). Backtest on 52 historical trades
                # showed ungated rug_event_exit cuts 3 trailing_stop winners
                # early (-$2.50 hurt). Gating to pnl < 0 + threshold 7.0 SOL
                # nets +$10.23 vs ungated +$8.18.
                pnl_gate_pass = pnl_pct <= float(self.config.rug_event_exit_max_pnl_pct)
                fire_exit = self.config.rug_event_exit_enabled and pnl_gate_pass

                if fire_exit:
                    action = "panic_sell"
                    logger.warning(
                        f"M4: RUG EVENT on {pos.token.symbol} — "
                        f"sol_out={sol_amount:.2f} price={price_sol:.2e} "
                        f"pnl={pnl_pct*100:+.1f}% — PANIC SELL")
                elif self.config.rug_event_exit_enabled and not pnl_gate_pass:
                    # Enabled but pnl gate blocked — token is profitable, skip
                    action = "skipped_pnl_gate"
                    logger.warning(
                        f"M4: RUG EVENT on {pos.token.symbol} [SKIP pnl gate] — "
                        f"sol_out={sol_amount:.2f} pnl={pnl_pct*100:+.1f}% > "
                        f"max={self.config.rug_event_exit_max_pnl_pct*100:.0f}%, "
                        f"letting trailing/v5 handle")
                else:
                    # Observation mode: record but do not execute
                    action = "would_sell_observe"
                    logger.warning(
                        f"M4: RUG EVENT on {pos.token.symbol} [OBSERVE] — "
                        f"sol_out={sol_amount:.2f} pnl_would_be={pnl_pct*100:+.1f}% "
                        f"(exit disabled, letting M4 handle)")

                self.db.record_rug_event(
                    mint_address=mint, pool_address=pool,
                    sol_amount=sol_amount, price_sol=price_sol,
                    detected_at=detected_at, action_taken=action,
                    signature=event.get("signature"),
                    triggered_pnl_pct=pnl_pct)

                if fire_exit:
                    # Fire panic sell with aggressive slippage. _execute_sell
                    # will idempotently no-op if already sold.
                    await self._execute_sell(
                        pos, reason="rug_event",
                        trigger_pnl_pct=pnl_pct,
                        trigger_price_sol=price_sol)
                    # Unwatch — sell path also unwatches; safe if sell fails.
                    if self._grpc_stream:
                        self._grpc_stream.unwatch_pool(pool)
                # In observe mode: keep watching (more rug dumps may follow,
                # and we want to record multiple events per position to see
                # the full dump trajectory). Position continues in M4 normally.
            except Exception as e:
                logger.error(f"M4: rug_event_watcher error: {e}", exc_info=True)

    # ──────────────────────────────────────────
    # Phase 22.S.P1 — gRPC price event watcher (fast SL/EC trigger)
    # ──────────────────────────────────────────

    async def _price_event_watcher(self):
        """Consume gRPC swap events and fire SL/EC immediately when crossed.

        Replaces 5s/2s polling for SL + early_crash exits. Polling stays
        as fallback for trail / v5.3 / time_limit / peak tracking.

        Logic mirrors `_monitor_positions` SL/EC checks byte-for-byte to
        avoid behavior drift. Difference is detection latency: <100ms
        instead of 0-5000ms.

        Concurrency safety:
          - `_exit_in_progress` set prevents double-fire (gRPC + poll race)
          - `_record_sell_trigger_once` is idempotent by sell_id
          - sells are awaited serially per mint
        """
        while True:
            try:
                ev = await self._price_event_queue.get()
            except asyncio.CancelledError:
                return
            try:
                if not self.config.grpc_event_exit_enabled:
                    # Plumbed but disabled — drain queue silently to avoid backlog
                    continue

                mint = ev["mint_address"]
                price_sol = float(ev["price_sol"])
                if price_sol <= 0:
                    continue

                # Concurrency guard
                if mint in self._exit_in_progress:
                    continue

                pos = next((p for p in self._positions
                              if p.token.mint_address == mint), None)
                if pos is None:
                    continue

                # Halt check (defensive — _monitor_positions has none, but
                # exits should respect halt state too).
                if self.risk and self.risk.halted:
                    continue
                # No grace gate: _monitor_positions fires SL/EC from hold=0,
                # so gRPC must too or polling beats us in the first 15s.
                # (Phase 22.S.P1's whole point is sub-100ms SL/EC latency.)
                hold_sec = pos.hold_seconds()

                if pos.entry_price_sol <= 0:
                    continue
                pnl_pct = (price_sol - pos.entry_price_sol) / pos.entry_price_sol

                fire_reason = None
                # SL check — fires anytime after grace
                if (self.config.grpc_event_exit_sl_enabled
                        and pnl_pct <= -self.config.stop_loss_pct):
                    fire_reason = "stop_loss"
                # EC check — fires only within EC window
                # Phase 25j (2026-05-04): peak_pnl_pct > 0 guard. Loss
                # audit found 9 of 30 ECs (4/20-5/4) fired AFTER position
                # had already peaked positive (avg peak +18%, total loss
                # $72) — recoverable bounces misclassified as crashes.
                # Once a trade has been in profit, let trailing/SL handle
                # the deeper drawdown rather than the tight EC threshold.
                elif (self.config.grpc_event_exit_ec_enabled
                        and hold_sec < self.config.vshape_early_crash_window_sec
                        and pnl_pct <= -self.config.vshape_early_crash_pct
                        and pos.peak_pnl_pct <= 0):
                    fire_reason = "early_crash"

                if not fire_reason:
                    continue

                # B6 fix (2026-05-09): require dip confirmation. The single
                # event triggering this path could be a transient sandwich
                # tick (audit: 87% of SL exits recovered >30% within 10min).
                # For SL/EC, require ≥3 swaps in last 5s confirming the dip.
                # Note: this gate runs in BOTH shadow and live modes so the
                # shadow log accurately reflects what live would do.
                stream = getattr(self, "_grpc_stream", None)
                if stream is not None and hasattr(stream, "is_dip_confirmed"):
                    threshold_pct = (-self.config.stop_loss_pct
                                      if fire_reason == "stop_loss"
                                      else -self.config.vshape_early_crash_pct)
                    try:
                        if not stream.is_dip_confirmed(
                                mint, float(pos.entry_price_sol),
                                threshold_pct,
                                window_sec=5.0, min_confirmations=3):
                            logger.debug(
                                f"M4 [GRPC-{fire_reason.upper()}]: {pos.token.symbol} "
                                f"DIP_NOT_CONFIRMED pnl={pnl_pct*100:+.1f}% — defer")
                            continue  # transient — wait for confirmation
                    except Exception:
                        pass  # safety: any error → preserve old behavior

                # Shadow mode: log only
                if self.config.grpc_event_exit_shadow_only:
                    logger.info(
                        f"M4 [GRPC-{fire_reason.upper()}-SHADOW]: {pos.token.symbol} "
                        f"would_fire pnl={pnl_pct*100:+.1f}% hold={hold_sec:.0f}s "
                        f"price={price_sol:.2e}")
                    continue

                # LIVE: dedupe + fire
                self._exit_in_progress.add(mint)
                try:
                    if self._defer_exit_for_retry(pos, fire_reason):
                        continue
                    self._record_sell_trigger_once(
                        pos, fire_reason,
                        pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                        trigger_price_sol=price_sol,
                        trigger_mid_price_sol=price_sol,
                        trigger_mid_source="grpc_event")
                    logger.warning(
                        f"M4 [GRPC-{fire_reason.upper()}]: {pos.token.symbol} "
                        f"FIRE pnl={pnl_pct*100:+.1f}% hold={hold_sec:.0f}s "
                        f"(saved ~{max(0, 5 - hold_sec % 5):.1f}s vs poll)")
                    await self._execute_sell(pos, fire_reason)
                finally:
                    self._exit_in_progress.discard(mint)
            except Exception as e:
                logger.error(f"M4: price_event_watcher error: {e}", exc_info=True)

    # ──────────────────────────────────────────
    # 14y shadow-exit-warn (dual-model drop + rug)
    # ──────────────────────────────────────────

    def _get_exit_models_mod(self):
        """Lazy-import the inference module. Safe to call when disabled."""
        if self._exit_models_mod is not None:
            return self._exit_models_mod
        try:
            from controllers.generic import meme_sniper_exit_models as mdl
            self._exit_models_mod = mdl
            # Warm up: force model load so first inference doesn't stall.
            mdl.load_tier_b()
            mdl.load_f2a_hc()
            try:
                mdl.load_f2a_hc_live_v1()
                live_v1_ok = True
            except Exception as e_v1:
                logger.warning(f"14y: live_v1 model not loaded: {e_v1}")
                live_v1_ok = False
            try:
                mdl.load_14y_v3_enhanced()
                v3_ok = True
            except Exception as e_v3:
                logger.warning(f"14y: v3_enhanced model not loaded: {e_v3}")
                v3_ok = False
            # v5.3 lean profit-protect (Phase 15e, 2026-04-29)
            v5_3_ok = False
            if self.config.v5_3_exit_enabled:
                try:
                    mdl.load_14y_v5_3()
                    v5_3_ok = True
                    logger.info(
                        f"v5.3 exit model loaded: "
                        f"cutoff={self.config.v5_3_exit_cutoff}, "
                        f"shadow_only={self.config.v5_3_exit_shadow_only}, "
                        f"grace={self.config.v5_3_exit_grace_sec}s")
                except Exception as e_v5_3:
                    logger.warning(f"v5.3 model not loaded: {e_v5_3}")
            # v5.5.2 sandwich-filtered Chainstack-native 12-feat (Phase 15h, 2026-05-11)
            v5_5_ok = False
            if self.config.v5_5_exit_enabled:
                try:
                    mdl.load_14y_v5_5_2()
                    v5_5_ok = True
                    logger.info(
                        f"v5.5.2 exit model loaded (sandwich-filtered): "
                        f"cutoff={self.config.v5_5_exit_cutoff}, "
                        f"profit_gate={self.config.v5_5_exit_profit_gate}, "
                        f"shadow_only={self.config.v5_5_exit_shadow_only}, "
                        f"grace={self.config.v5_5_exit_grace_sec}s")
                except Exception as e_v5_5:
                    logger.warning(f"v5.5.2 model not loaded: {e_v5_5}")
            # v5.5.3 sandwich-filtered + fragility-clipped 12-feat (Phase 15i, 2026-05-11)
            v5_5_3_ok = False
            if self.config.v5_5_3_exit_enabled:
                try:
                    mdl.load_14y_v5_5_3()
                    v5_5_3_ok = True
                    logger.info(
                        f"v5.5.3 exit model loaded (sandwich + fragility clips): "
                        f"cutoff={self.config.v5_5_3_exit_cutoff}, "
                        f"profit_gate={self.config.v5_5_3_exit_profit_gate}, "
                        f"shadow_only={self.config.v5_5_3_exit_shadow_only}, "
                        f"grace={self.config.v5_5_3_exit_grace_sec}s")
                except Exception as e_v5_5_3:
                    logger.warning(f"v5.5.3 model not loaded: {e_v5_5_3}")
            logger.info("14y: exit-warn models loaded "
                        f"(Tier B + F2a+HC + live_v1={'yes' if live_v1_ok else 'no'}"
                        f" + v3={'yes' if v3_ok else 'no'}"
                        f" + v5_3={'yes' if v5_3_ok else 'no'}"
                        f" + v5_5={'yes' if v5_5_ok else 'no'})")
        except Exception as e:
            logger.error(f"14y: failed to load exit-warn models: {e}",
                         exc_info=True)
            self._exit_models_mod = None
        return self._exit_models_mod

    def _run_shadow_exit_warn_tick(self, pos, now: float) -> None:
        """Called from _monitor_positions for each active position.

        Rate-limited to `shadow_exit_warn_tick_interval` seconds per mint.
        Fetches recent SwapRecords from gRPC buffer, computes 25 features,
        scores both models, writes one row to shadow_exit_warn_evals.
        All failures are logged and suppressed — never blocks the main
        monitor loop.
        """
        if not self.config.shadow_exit_warn_enabled:
            return
        mint = pos.token.mint_address
        last = self._last_shadow_exit_warn_ts.get(mint, 0.0)
        if now - last < self.config.shadow_exit_warn_tick_interval:
            return
        self._last_shadow_exit_warn_ts[mint] = now

        if self._grpc_stream is None or not self._grpc_stream.connected:
            return
        mdl = self._get_exit_models_mod()
        if mdl is None:
            return

        try:
            t = int(now)
            # Gate: only score when dt_from_grad is inside the training
            # distribution. Outside this window, XGBoost extrapolation is
            # unreliable.
            #
            # 2026-04-26 widened from (600, 1170) → (315, 2400) for v4.1
            # dual-anchor support (see Phase_15a_DeepAudit_2026-04-26.md §F1):
            #   v4.1 sim_300 anchor (v1.6 entry @ grad+300) training range
            #     dt_from_grad ∈ [315, 2085]. Old gate cut 49% of this.
            #   v4.1 sim_600 anchor (v1.4r entry @ grad+600) training range
            #     dt_from_grad ∈ [615, 2385]. Old gate cut 12% of this.
            #   Real-trade anchor: [623, 1483]. Already inside both gates.
            # New (315, 2400) covers ≥99% of all three anchors.
            grad_time_raw = getattr(pos.token, "graduation_time", None)
            grad_time = int(grad_time_raw) if grad_time_raw else None
            if grad_time is None:
                return
            dt_from_grad = t - grad_time
            if not (315 <= dt_from_grad <= 2400):
                return

            records = self._grpc_stream.get_recent_swap_records(
                mint, max_age_sec=600.0)
            if len(records) < self.config.shadow_exit_warn_min_swaps:
                return

            import pandas as pd  # local import; cheap
            df = pd.DataFrame([{
                "token_address": mint,
                "block_time": r.timestamp,
                "trader_address": r.trader_address,
                "is_buy": 1 if r.is_buy else 0,
                "sol_amount": r.volume_sol,
                "token_amount": r.base_amount,
                "effective_price_sol": r.price_sol,
            } for r in records])
            df = df.sort_values("block_time").reset_index(drop=True)

            feats = mdl.compute_all_features(df, t, grad_time)
            if feats is None:
                return
            x_fh = mdl.to_feature_array(
                feats, mdl.F2A_HC_FEATURE_ORDER).reshape(1, -1)

            # ─── DEPRECATED 2026-04-25: p_drop_raw (Tier B 19-feature) ───
            # Real-trade backtest (29 trades): AUC 0.43 (INVERTED), recall 12%
            # @ FPR 43%. Score is anti-correlated with bot losses. See
            # Phase_14y_v2_TargetAligned §1.1 + Phase_14y_v4 §C audit.
            # Kept as sentinel 0.0 because schema column is NOT NULL.
            p_drop = 0.0  # sentinel — deprecated, do not use

            p_rug = float(mdl.load_f2a_hc()["model"].predict_proba(x_fh)[0, 1])

            # ─── DEPRECATED 2026-04-25: p_rug_live_v1 (bot-schema retrain) ───
            # Score distribution collapsed to [0.03, 0.255] — never crosses any
            # actionable cutoff. Real-trade recall = 0/10. Spearman ρ=0.73 with
            # p_rug_raw → fully redundant. See Phase_14y_v4 §C audit.
            p_rug_live_v1: Optional[float] = None  # column nullable
            # v3 enhanced (event-anchored LogReg) — 2026-04-25
            # Multi-window inference: tries 180s primary, falls back to 300s
            # when sparse. Window used is recorded for offline calibration.
            p_rug_v3_window: Optional[int] = None
            try:
                p_rug_v3 = mdl.predict_14y_v3(df, t, int(grad_time))
                if p_rug_v3 is not None:
                    p_rug_v3 = float(p_rug_v3)
                    p_rug_v3_window = mdl.V3_LAST_WINDOW_USED
            except Exception as e_v3:
                logger.debug(f"14y v3 score failed for {mint[:10]}: {e_v3}")
                p_rug_v3 = None

            # v4 position-conditional ensemble (2026-04-25, OOS-clean opt A).
            # Returns dict with p_y_60s, p_y_120s, p_rug_v4 (deploy ensemble).
            # SHADOW ONLY — no production gate.
            p_rug_v4: Optional[float] = None
            p_rug_v4_y60s: Optional[float] = None
            p_rug_v4_y120s: Optional[float] = None
            v4_fail_reason: Optional[str] = None
            try:
                v4_out = mdl.predict_14y_v4(
                    df, t, int(grad_time),
                    entry_time=int(pos.entry_time),
                    entry_price=float(pos.entry_price_sol),
                )
                if v4_out is not None:
                    p_rug_v4 = float(v4_out["p_rug_v4"])
                    p_rug_v4_y60s = float(v4_out["p_y_60s"])
                    p_rug_v4_y120s = float(v4_out["p_y_120s"])
                    # Phase 15d: cache scores on position for M4 soft-exit gate.
                    # Stored as a dict so monitor loop can pick whichever horizon
                    # the config selects (p_rug_v4 / p_y_60s / p_y_120s).
                    pos.last_v4_scores = {
                        "p_rug_v4": p_rug_v4,
                        "p_y_60s": p_rug_v4_y60s,
                        "p_y_120s": p_rug_v4_y120s,
                    }
                    pos.last_v4_score_t = float(now)
                else:
                    # P0 diagnostic — root-cause why predict_14y_v4 returned None
                    if float(pos.entry_price_sol) <= 0:
                        v4_fail_reason = "entry_price_le_zero"
                    elif t < int(pos.entry_time):
                        v4_fail_reason = "t_before_entry"
                    else:
                        v4_fail_reason = "feature_compute_none"
            except Exception as e_v4:
                v4_fail_reason = f"exc:{type(e_v4).__name__}:{str(e_v4)[:80]}"
                logger.debug(f"14y v4 score failed for {mint[:10]}: {e_v4}")

            # v5.2 lean profit-protect ensemble (Phase 15d, 22/22 pre-deploy
            # PASS). 15 features = 11 sf/tb + 2 ps + 2 hc-removed. Live deploy
            # gate at p_dd_v5 >= V5_2_DEPLOY_CUTOFF (0.30). Backtest on 37 real
            # trades: $-44 → $+56 (+$101) at $10 sizing. Replaces v4_exit gate.
            p_dd_v5: Optional[float] = None
            p_dd_v5_60s: Optional[float] = None
            p_dd_v5_120s: Optional[float] = None
            v5_fail_reason: Optional[str] = None
            try:
                v5_out = mdl.predict_14y_v5_2(
                    df, t, int(grad_time),
                    entry_time=int(pos.entry_time),
                    entry_price=float(pos.entry_price_sol),
                )
                if v5_out is not None:
                    p_dd_v5 = float(v5_out["p_dd_v5"])
                    p_dd_v5_60s = float(v5_out["p_dd_60s"])
                    p_dd_v5_120s = float(v5_out["p_dd_120s"])
                    pos.last_v5_scores = {
                        "p_dd_v5": p_dd_v5,
                        "p_dd_60s": p_dd_v5_60s,
                        "p_dd_120s": p_dd_v5_120s,
                    }
                    pos.last_v5_score_t = float(now)
                else:
                    if float(pos.entry_price_sol) <= 0:
                        v5_fail_reason = "entry_price_le_zero"
                    elif t < int(pos.entry_time):
                        v5_fail_reason = "t_before_entry"
                    else:
                        v5_fail_reason = "feature_compute_none"
            except Exception as e_v5:
                v5_fail_reason = f"exc:{type(e_v5).__name__}:{str(e_v5)[:80]}"
                logger.debug(f"14y v5.2 score failed for {mint[:10]}: {e_v5}")

            # ─── v5.3 lean profit-protect (Phase 15e, 2026-04-29) ───
            # Trained on the historical BigWinner + V-shape entry distribution.
            # Bot policy in training matches CURRENT live (trail_020/drop_010/sl_030/TL_900).
            # Cutoff 0.50 (winner of 49-cell joint sweep).
            # Default deploy: SHADOW only (logs to shadow_v5_3_evals; v5.2 still fires).
            p_dd_v5_3: Optional[float] = None
            p_dd_v5_3_60s: Optional[float] = None
            p_dd_v5_3_120s: Optional[float] = None
            v5_3_fail_reason: Optional[str] = None
            if self.config.v5_3_exit_enabled:
                try:
                    v5_3_out = mdl.predict_14y_v5_3(
                        df, t, int(grad_time),
                        entry_time=int(pos.entry_time),
                        entry_price=float(pos.entry_price_sol),
                    )
                    if v5_3_out is not None:
                        p_dd_v5_3 = float(v5_3_out["p_dd_v5_3"])
                        p_dd_v5_3_60s = float(v5_3_out["p_dd_60s"])
                        p_dd_v5_3_120s = float(v5_3_out["p_dd_120s"])
                        pos.last_v5_3_scores = {
                            "p_dd_v5_3": p_dd_v5_3,
                            "p_dd_60s": p_dd_v5_3_60s,
                            "p_dd_120s": p_dd_v5_3_120s,
                        }
                        pos.last_v5_3_score_t = float(now)
                    else:
                        if float(pos.entry_price_sol) <= 0:
                            v5_3_fail_reason = "entry_price_le_zero"
                        elif t < int(pos.entry_time):
                            v5_3_fail_reason = "t_before_entry"
                        else:
                            v5_3_fail_reason = "feature_compute_none"
                except Exception as e_v53:
                    v5_3_fail_reason = f"exc:{type(e_v53).__name__}:{str(e_v53)[:80]}"
                    logger.debug(f"v5.3 score failed for {mint[:10]}: {e_v53}")

                # Always log shadow eval (every tick; 0 cost when v5.3 disabled)
                try:
                    cutoff = float(self.config.v5_3_exit_cutoff)
                    hold_s = int(now - pos.entry_time)
                    would_fire_v53 = int(
                        p_dd_v5_3 is not None
                        and p_dd_v5_3 >= cutoff
                        and hold_s >= int(self.config.v5_3_exit_grace_sec)
                    )
                    # Compute current pnl from last available swap
                    cur_pnl: Optional[float] = None
                    try:
                        last_px = float(df["effective_price_sol"].iloc[-1])
                        ep = float(pos.entry_price_sol)
                        if last_px > 0 and ep > 0:
                            cur_pnl = (last_px - ep) / ep
                    except Exception:
                        cur_pnl = None
                    self.db.record_shadow_v5_3_eval(
                        mint_address=mint,
                        position_id=None,   # positions don't carry trade row id
                        hold_sec=hold_s,
                        p_dd_60s=p_dd_v5_3_60s,
                        p_dd_120s=p_dd_v5_3_120s,
                        p_dd_v5_3=p_dd_v5_3,
                        cutoff_value=cutoff,
                        would_fire=would_fire_v53,
                        pos_pnl_pct=cur_pnl,
                        peak_pnl_pct=float(pos.peak_pnl_pct),
                        sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else None,
                        fail_reason=v5_3_fail_reason,
                    )
                except Exception as e_log:
                    logger.debug(f"v5.3 shadow log failed for {mint[:10]}: {e_log}")

            # ─── v5.5.1 Chainstack-native 12-feat (Phase 15g, 2026-05-11) ───
            # Path C 12-feat model. Bug-fix retrain (BUG-1/2/3 corrected).
            # Phase 3 audit 27/28 PASS. Shadow-only by default — logs every
            # tick to shadow_v5_5_evals for Day-7 cutoff recalib + counterfactual
            # PnL comparison vs live v5.3 fires.
            p_dd_v5_5: Optional[float] = None
            raw_cur_pnl_v5_5: Optional[float] = None
            v5_5_fail_reason: Optional[str] = None
            if self.config.v5_5_exit_enabled:
                try:
                    # v5.5.2 (Phase 15h, 2026-05-11): sandwich-filtered features.
                    # Replaces v5.5.1's vulnerable last-raw-swap reads. Same 12
                    # features, same API, but swap stream is cleaned (window=30,
                    # max_ratio=3.0) before feature compute.
                    v5_5_out = mdl.predict_14y_v5_5_2(
                        df, float(t), float(grad_time),
                        entry_time=float(pos.entry_time),
                        entry_price=float(pos.entry_price_sol),
                    )
                    if v5_5_out is not None:
                        p_dd_v5_5 = float(v5_5_out["p_dd_v5_5"])
                        raw_cur_pnl_v5_5 = float(v5_5_out["raw_cur_pnl"])
                        pos.last_v5_5_score = {
                            "p_dd_v5_5": p_dd_v5_5,
                            "raw_cur_pnl": raw_cur_pnl_v5_5,
                        }
                        pos.last_v5_5_score_t = float(now)
                    else:
                        if float(pos.entry_price_sol) <= 0:
                            v5_5_fail_reason = "entry_price_le_zero"
                        elif t < int(pos.entry_time):
                            v5_5_fail_reason = "t_before_entry"
                        else:
                            v5_5_fail_reason = "feature_compute_none"
                except Exception as e_v55:
                    v5_5_fail_reason = f"exc:{type(e_v55).__name__}:{str(e_v55)[:80]}"
                    logger.debug(f"v5.5 score failed for {mint[:10]}: {e_v55}")

                try:
                    cutoff_v55 = float(self.config.v5_5_exit_cutoff)
                    profit_gate_v55 = float(self.config.v5_5_exit_profit_gate)
                    hold_s_v55 = int(now - pos.entry_time)
                    would_fire_v55 = int(
                        p_dd_v5_5 is not None
                        and raw_cur_pnl_v5_5 is not None
                        and raw_cur_pnl_v5_5 >= profit_gate_v55
                        and p_dd_v5_5 >= cutoff_v55
                        and hold_s_v55 >= int(self.config.v5_5_exit_grace_sec)
                    )
                    cur_pnl_v55: Optional[float] = None
                    try:
                        last_px_v55 = float(df["effective_price_sol"].iloc[-1])
                        ep_v55 = float(pos.entry_price_sol)
                        if last_px_v55 > 0 and ep_v55 > 0:
                            cur_pnl_v55 = (last_px_v55 - ep_v55) / ep_v55
                    except Exception:
                        cur_pnl_v55 = None
                    self.db.record_shadow_v5_5_eval(
                        mint_address=mint,
                        position_id=None,
                        hold_sec=hold_s_v55,
                        p_dd_v5_5=p_dd_v5_5,
                        raw_cur_pnl=raw_cur_pnl_v5_5,
                        cutoff_value=cutoff_v55,
                        profit_gate=profit_gate_v55,
                        would_fire=would_fire_v55,
                        pos_pnl_pct=cur_pnl_v55,
                        peak_pnl_pct=float(pos.peak_pnl_pct),
                        sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else None,
                        fail_reason=v5_5_fail_reason,
                    )
                except Exception as e_log_v55:
                    logger.debug(f"v5.5 shadow log failed for {mint[:10]}: {e_log_v55}")

            # ─── v5.5.3 fragility-clipped 12-feat (Phase 15i, 2026-05-11) ───
            # Sandwich filter + per-swap clip on Amihud / return / volatility.
            # Same 12 features as v5.5.2 but trained on clipped panel + applies
            # training-time p99 outlier clips at inference (helper in exit_models).
            # Shadow-only at deploy → logs to shadow_v5_5_3_evals for 7-day A/B
            # vs v5.5.2 LIVE fires. Active fire path not wired (cutoff to add
            # when promoting; shadow-only=false will be gated then).
            p_dd_v5_5_3: Optional[float] = None
            raw_cur_pnl_v5_5_3: Optional[float] = None
            v5_5_3_fail_reason: Optional[str] = None
            if self.config.v5_5_3_exit_enabled:
                try:
                    v5_5_3_out = mdl.predict_14y_v5_5_3(
                        df, float(t), float(grad_time),
                        entry_time=float(pos.entry_time),
                        entry_price=float(pos.entry_price_sol),
                    )
                    if v5_5_3_out is not None:
                        p_dd_v5_5_3 = float(v5_5_3_out["p_dd_v5_5"])
                        raw_cur_pnl_v5_5_3 = float(v5_5_3_out["raw_cur_pnl"])
                    else:
                        if float(pos.entry_price_sol) <= 0:
                            v5_5_3_fail_reason = "entry_price_le_zero"
                        elif t < int(pos.entry_time):
                            v5_5_3_fail_reason = "t_before_entry"
                        else:
                            v5_5_3_fail_reason = "feature_compute_none"
                except Exception as e_v553:
                    v5_5_3_fail_reason = f"exc:{type(e_v553).__name__}:{str(e_v553)[:80]}"
                    logger.debug(f"v5.5.3 score failed for {mint[:10]}: {e_v553}")

                try:
                    cutoff_v553 = float(self.config.v5_5_3_exit_cutoff)
                    profit_gate_v553 = float(self.config.v5_5_3_exit_profit_gate)
                    hold_s_v553 = int(now - pos.entry_time)
                    would_fire_v553 = int(
                        p_dd_v5_5_3 is not None
                        and raw_cur_pnl_v5_5_3 is not None
                        and raw_cur_pnl_v5_5_3 >= profit_gate_v553
                        and p_dd_v5_5_3 >= cutoff_v553
                        and hold_s_v553 >= int(self.config.v5_5_3_exit_grace_sec)
                    )
                    cur_pnl_v553: Optional[float] = None
                    try:
                        last_px_v553 = float(df["effective_price_sol"].iloc[-1])
                        ep_v553 = float(pos.entry_price_sol)
                        if last_px_v553 > 0 and ep_v553 > 0:
                            cur_pnl_v553 = (last_px_v553 - ep_v553) / ep_v553
                    except Exception:
                        cur_pnl_v553 = None
                    self.db.record_shadow_v5_5_3_eval(
                        mint_address=mint,
                        position_id=None,
                        hold_sec=hold_s_v553,
                        p_dd_v5_5=p_dd_v5_5_3,
                        raw_cur_pnl=raw_cur_pnl_v5_5_3,
                        cutoff_value=cutoff_v553,
                        profit_gate=profit_gate_v553,
                        would_fire=would_fire_v553,
                        pos_pnl_pct=cur_pnl_v553,
                        peak_pnl_pct=float(pos.peak_pnl_pct),
                        sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else None,
                        fail_reason=v5_5_3_fail_reason,
                    )
                except Exception as e_log_v553:
                    logger.debug(f"v5.5.3 shadow log failed for {mint[:10]}: {e_log_v553}")

            # Smoothed mid = last swap's effective_price_sol (5-swap rolling
            # median already applied internally to feature compute; approximate
            # here with the last swap's price for audit / reconstruction).
            last_swap_price = float(df["effective_price_sol"].iloc[-1])

            self.db.record_shadow_exit_warn_eval(
                position_id=None,  # positions don't currently carry a trade row id
                mint_address=mint,
                symbol=pos.token.symbol,
                dt_from_entry=int(now - pos.entry_time),
                dt_from_grad=dt_from_grad,
                p_drop_raw=p_drop,
                p_rug_raw=p_rug,
                mid_price_sol=last_swap_price if last_swap_price > 0 else None,
                n_cache_swaps=int(len(df)),
                feature_window_count=int(feats.get("sf_swap_density", 0)),
                features_json=None,
                p_rug_live_v1=p_rug_live_v1,
                p_rug_v3=p_rug_v3,
                p_rug_v3_window=p_rug_v3_window,
                p_rug_v4=p_rug_v4,
                p_rug_v4_y60s=p_rug_v4_y60s,
                p_rug_v4_y120s=p_rug_v4_y120s,
                p_dd_v5=p_dd_v5,
                p_dd_v5_60s=p_dd_v5_60s,
                p_dd_v5_120s=p_dd_v5_120s,
                v4_fail_reason=v4_fail_reason,
                v5_fail_reason=v5_fail_reason,
            )
        except Exception as e:
            logger.debug(f"14y shadow-exit-warn tick failed for {mint[:10]}: {e}")

    async def _shadow_exit_warn_resolver_loop(self):
        """Background task. Every `shadow_exit_warn_resolver_interval` s:
        1. Fetch unresolved rows older than 60s.
        2. For each, look at swap history from t .. t+60s to compute
           realized_drop_60 and realized_max_sell_60.
        3. Resolve the row.
        """
        import pandas as pd  # noqa — kept local for lazy import
        interval = max(10, int(self.config.shadow_exit_warn_resolver_interval))
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.config.shadow_exit_warn_enabled:
                    continue
                rows = self.db.fetch_unresolved_shadow_exit_warn_evals(
                    older_than_sec=60.0, limit=1000)
                if not rows:
                    continue
                resolved_count = 0
                for row_id, mint, ts in rows:
                    outcome = self._compute_shadow_exit_warn_outcome(mint, ts)
                    if outcome is None:
                        # Skip rows with no swap data (will retry next cycle);
                        # after 15 min, force-resolve as NULL to keep the queue
                        # from growing unbounded.
                        if time.time() - ts > 900:
                            self.db.resolve_shadow_exit_warn_eval(
                                row_id=row_id,
                                realized_drop_60=None,
                                realized_max_sell_60=None,
                                y_drop_actual=None, y_rug_actual=None)
                            resolved_count += 1
                        continue
                    self.db.resolve_shadow_exit_warn_eval(
                        row_id=row_id, **outcome)
                    resolved_count += 1
                if resolved_count:
                    logger.debug(
                        f"14y: shadow-exit-warn resolver processed {resolved_count} rows")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"14y: resolver loop error: {e}", exc_info=True)

    def _compute_shadow_exit_warn_outcome(self, mint: str, t: float) -> Optional[Dict]:
        """Return realized outcomes at (t, t+60s] from the gRPC swap buffer.
        None if the buffer doesn't cover that window.
        """
        if self._grpc_stream is None:
            return None
        records = self._grpc_stream.get_recent_swap_records(
            mint, max_age_sec=1800.0)
        if not records:
            return None
        # We need: the last swap with block_time <= t (for entry mid) AND
        # all swaps with block_time in (t, t+60].
        recs_sorted = sorted(records, key=lambda r: r.timestamp)
        pre_t = [r for r in recs_sorted if r.timestamp <= t]
        future = [r for r in recs_sorted if t < r.timestamp <= t + 60]
        if not pre_t or not future:
            return None
        entry_price = float(pre_t[-1].price_sol)
        if entry_price <= 0:
            return None
        fut_prices = [r.price_sol for r in future if r.price_sol > 0]
        if not fut_prices:
            return None
        min_future_price = float(min(fut_prices))
        realized_drop_60 = (min_future_price / entry_price) - 1.0
        fut_sells = [r.volume_sol for r in future if not r.is_buy]
        realized_max_sell_60 = float(max(fut_sells)) if fut_sells else 0.0
        y_drop_actual = 1 if realized_drop_60 < -0.15 else 0
        y_rug_actual = 1 if realized_max_sell_60 >= 10.0 else 0
        return {
            "realized_drop_60": realized_drop_60,
            "realized_max_sell_60": realized_max_sell_60,
            "y_drop_actual": y_drop_actual,
            "y_rug_actual": y_rug_actual,
        }

    # ──────────────────────────────────────────
    # 19c sw_v2 (F1 entry) shadow scoring
    # ──────────────────────────────────────────

    def _get_sw_v2_mod(self):
        """Lazy-load the sw_v2 inference module. Returns None on failure
        so the hook can silently skip rather than crash the controller.
        """
        if self._sw_v2_mod is not None:
            return self._sw_v2_mod
        try:
            from controllers.generic import meme_sniper_sw_v2_model as mdl  # type: ignore
            self._sw_v2_mod = mdl
            logger.info("sw_v2: inference module loaded")
        except Exception as e:
            logger.warning(f"sw_v2: failed to import module: {e}")
            self._sw_v2_mod = None
        return self._sw_v2_mod

    def _run_sw_v2_shadow_once(self, token, now: float) -> None:
        """OBSERVE_ONLY F1 score at T+180s for a single graduation.

        Called once per token from the main pending-processing loop when
        `age >= shadow_sw_v2_entry_offset_sec`. All exceptions are caught
        and logged at DEBUG; this function must never block or break the
        main loop.
        """
        if not self.config.shadow_sw_v2_enabled:
            return
        mint = token.mint_address
        if self._grpc_stream is None or not self._grpc_stream.connected:
            return
        mdl = self._get_sw_v2_mod()
        if mdl is None:
            return

        try:
            grad_time_raw = getattr(token, "graduation_time", None)
            if grad_time_raw is None:
                return
            grad_time = int(grad_time_raw)
            dt_from_grad = int(now - grad_time)

            # Pull all recent swaps — we'll window them by block_time inside
            # build_aligned_bars. Max-age covers pre-grad + 180 s + margin.
            records = self._grpc_stream.get_recent_swap_records(
                mint, max_age_sec=600.0)
            if len(records) < int(self.config.shadow_sw_v2_min_swaps):
                return

            import pandas as pd  # local import; cheap
            df = pd.DataFrame([{
                "block_time": r.timestamp,
                "sol_amount": r.volume_sol,
                "token_amount": r.base_amount,
                "effective_price_sol": r.price_sol,
                "is_buy": 1 if r.is_buy else 0,
            } for r in records])

            sol_price_usd = float(getattr(self, "_sol_price_usd", 0.0) or 150.0)
            if sol_price_usd <= 0:
                sol_price_usd = 150.0

            out = mdl.predict(df, grad_time, sol_price_usd)
            if out is None:
                self.db.record_shadow_sw_v2_eval(
                    mint_address=mint,
                    symbol=token.symbol,
                    graduation_time=float(grad_time),
                    dt_from_grad=dt_from_grad,
                    entry_offset_sec=None,
                    entry_price_sol=None,
                    grad_price_sol=None,
                    n_bars_pre_entry=None,
                    n_swaps_used=int(len(df)),
                    feature_ready=False,
                    f1_score=None,
                    features_json=None,
                )
                return

            # Persist. Keep features_json compact (floats only).
            import json as _json
            import math as _math
            def _clean(v):
                if v is None:
                    return None
                if isinstance(v, float) and not _math.isfinite(v):
                    return None
                return v
            feats_json = _json.dumps(
                {k: _clean(v) for k, v in out["features"].items()},
                default=str)

            self.db.record_shadow_sw_v2_eval(
                mint_address=mint,
                symbol=token.symbol,
                graduation_time=float(grad_time),
                dt_from_grad=dt_from_grad,
                entry_offset_sec=out["meta"]["entry_offset_sec"],
                entry_price_sol=out["meta"]["entry_price_sol"],
                grad_price_sol=out["meta"]["grad_price_sol"],
                n_bars_pre_entry=out["meta"]["n_bars_pre_entry"],
                n_swaps_used=int(len(df)),
                feature_ready=True,
                f1_score=out["f1_score"],
                features_json=feats_json,
            )
        except Exception as e:
            logger.debug(f"sw_v2 shadow tick failed for {mint[:10]}: {e}")

    async def _shadow_sw_v2_resolver_loop(self):
        """Every `shadow_sw_v2_resolver_interval` seconds, resolve unresolved
        rows older than `shadow_sw_v2_resolver_horizon_sec` by computing
        realized_raw_15m from the swap buffer. Force-resolve rows older than
        2× horizon to keep the queue bounded.
        """
        interval = max(60, int(self.config.shadow_sw_v2_resolver_interval))
        horizon = int(self.config.shadow_sw_v2_resolver_horizon_sec)
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.config.shadow_sw_v2_enabled:
                    continue
                rows = self.db.fetch_unresolved_shadow_sw_v2_evals(
                    older_than_sec=horizon, limit=500)
                if not rows:
                    continue
                resolved = 0
                for row_id, mint, grad_time, entry_offset_sec, entry_price_sol, ts in rows:
                    try:
                        outcome = self._compute_shadow_sw_v2_outcome(
                            mint=mint,
                            entry_time=ts,      # wall-clock when we scored
                            entry_price_sol=entry_price_sol,
                            horizon_sec=horizon,
                        )
                        if outcome is None:
                            # Force-resolve rows older than 2× horizon to bound queue.
                            if time.time() - ts > 2 * horizon:
                                self.db.resolve_shadow_sw_v2_eval(
                                    row_id=row_id,
                                    realized_raw_15m=None,
                                    y_net15_emp_trail_actual=None,
                                    peak_ret_15m=None,
                                    trough_ret_15m=None,
                                    eventual_trade_pnl_usd=None)
                                resolved += 1
                            continue
                        self.db.resolve_shadow_sw_v2_eval(row_id=row_id, **outcome)
                        resolved += 1
                    except Exception as e:
                        logger.debug(f"sw_v2 resolver row {row_id} failed: {e}")
                if resolved:
                    logger.debug(f"sw_v2: resolver processed {resolved} rows")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"sw_v2: resolver loop error: {e}", exc_info=True)

    def _compute_shadow_sw_v2_outcome(
        self, *, mint: str, entry_time: float,
        entry_price_sol: Optional[float], horizon_sec: int,
    ) -> Optional[Dict]:
        """Compute realized return from (entry_time, entry_time+horizon_sec].

        Uses the gRPC swap buffer. Joins with trades table to surface
        eventual_trade_pnl_usd if v1.4 later opened a position.
        """
        if self._grpc_stream is None:
            return None
        if entry_price_sol is None or entry_price_sol <= 0:
            return None
        recs = self._grpc_stream.get_recent_swap_records(
            mint, max_age_sec=3600.0)
        if not recs:
            return None
        future = [r for r in recs
                  if entry_time < r.timestamp <= entry_time + horizon_sec
                  and r.price_sol > 0]
        if len(future) < 3:
            return None
        last_close = float(future[-1].price_sol)
        peak_price = float(max(r.price_sol for r in future))
        trough_price = float(min(r.price_sol for r in future))
        realized_raw_15m = (last_close / entry_price_sol) - 1.0
        peak_ret_15m = (peak_price / entry_price_sol) - 1.0
        trough_ret_15m = (trough_price / entry_price_sol) - 1.0
        y_net15_emp_trail_actual = 1 if (realized_raw_15m - 0.063) > 0 else 0
        # Look up eventual trade PnL if v1.4 opened a position on this mint
        eventual_pnl = None
        try:
            rows = self.db._fetchall(
                "SELECT pnl_usd FROM trades WHERE mint_address = ? "
                "ORDER BY id DESC LIMIT 1", (mint,))
            if rows and rows[0][0] is not None:
                eventual_pnl = float(rows[0][0])
        except Exception:
            pass
        return {
            "realized_raw_15m": realized_raw_15m,
            "y_net15_emp_trail_actual": y_net15_emp_trail_actual,
            "peak_ret_15m": peak_ret_15m,
            "trough_ret_15m": trough_ret_15m,
            "eventual_trade_pnl_usd": eventual_pnl,
        }

    async def _run_shadow_event_invariant(self):
        """Shadow scanner for Event Invariant v1 Model B (Phase B research).

        Same mechanics as _run_shadow_event but:
          - model: event_invariant_v1 Model B (regressor)
          - scan window: T+3m..T+5m (narrow — only positive-EV subset
            found in holdout backtest, 30 tokens, 73% win, +$12.68)
          - cutoff: top5 (strictest)

        On first trigger inside the window, records to
        `shadow_event_invariant_evals` and stops scanning that token.
        Does NOT affect real trading.
        """
        if (not self.config.shadow_event_invariant_enabled
                or self.event_invariant_model is None
                or not self._shadow_event_invariant_pending):
            return

        cfg = self.config
        cutoff_band = cfg.shadow_event_invariant_cutoff_band
        cutoff_value = self.event_invariant_model.selection_cutoffs.get(cutoff_band)
        if cutoff_value is None:
            logger.warning(
                f"EVENT-INV-SHADOW: cutoff band {cutoff_band!r} not in model; "
                f"available: {sorted(self.event_invariant_model.selection_cutoffs.keys())}")
            return

        now = time.time()
        completed: List[str] = []
        scan_step_sec = cfg.shadow_event_scan_step_sec  # reuse flow's step (30s)
        for mint, payload in list(self._shadow_event_invariant_pending.items()):
            token = payload["token"]
            age_sec = now - token.graduation_time

            if payload["triggered"] or age_sec > cfg.shadow_event_invariant_scan_end_sec + 60:
                completed.append(mint)
                continue
            if age_sec < cfg.shadow_event_invariant_scan_start_sec:
                continue

            target_scan = int(age_sec) - (int(age_sec) % scan_step_sec)
            if target_scan <= payload["last_scan_sec"]:
                continue
            payload["last_scan_sec"] = target_scan

            try:
                n_bars = max(3, target_scan // 60 + 2)
                kline_bars, _ = self.kline_builder.build_kline(
                    mint=mint,
                    start_ts=int(token.graduation_time),
                    n_bars=n_bars, resolution=60,
                    sol_price_usd=self._sol_price_usd if self._sol_price_usd > 0 else 80.0,
                )
                if not kline_bars or len(kline_bars) < 3:
                    continue
                vf = detect_pattern_at_t(kline_bars, token.graduation_time, target_scan)
                if vf is None or vf.get("any_pattern", 0) != 1:
                    continue
                swaps_raw = self.db.get_swaps_for_token(mint)
                micro = compute_micro_at_t(
                    swaps_raw, token.graduation_time, target_scan,
                    swap_data_max_sec=self.event_invariant_model.swap_data_max_sec)
                if not micro:
                    continue
                features = {**vf, **micro}
                score = self.event_invariant_model.predict_score(features)

                if score >= cutoff_value:
                    self.db.record_shadow_event_invariant_eval(
                        mint_address=mint, symbol=token.symbol,
                        graduation_time=token.graduation_time,
                        scan_t_sec=target_scan,
                        cutoff_band=cutoff_band, cutoff_value=cutoff_value,
                        score=score, pattern=vf["pattern"],
                        entry_price_sol=vf.get("entry_price"),
                        sol_price_usd=self._sol_price_usd,
                        model_version=self.event_invariant_model.version,
                        features=features)
                    logger.info(
                        f"EVENT-INV-SHADOW: {token.symbol} trigger at T+{target_scan//60}m "
                        f"score={score:.4f} cutoff={cutoff_band}={cutoff_value:.4f} "
                        f"pattern={vf['pattern']}")
                    payload["triggered"] = True
            except Exception as e:
                logger.debug(
                    f"EVENT-INV-SHADOW: {token.symbol} scan failed at t={target_scan}s: {e}")

        for mint in completed:
            self._shadow_event_invariant_pending.pop(mint, None)

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
                info_60m = await self._fetch_gmgn_token_info(token.mint_address, token.symbol)
                security = await self._fetch_gmgn_token_security(token.mint_address, token.symbol)
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
                    gmgn_info_60m=info_60m, gmgn_security=security)
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
                            sec_early = await self._fetch_gmgn_token_security(
                                obs.token.mint_address, obs.token.symbol)
                            if sec_early:
                                self.db.update_observation_snapshot(
                                    obs.obs_id, "gmgn_security_t0", sec_early)
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
                    info = await self._fetch_gmgn_token_info(
                        obs.token.mint_address, obs.token.symbol)
                    if info:
                        self.db.update_observation_snapshot(obs.obs_id, col_name, info)
                    obs.snapshots_done.add(col_name)

                    # For t0: also fetch security snapshot immediately.
                    # This gives us honeypot/burn/renounce/tax at the moment
                    # the token first appears — before any trading or
                    # manipulation changes these flags.
                    sec = None
                    if col_name == "gmgn_info_t0":
                        obs.snapshots_done.add("gmgn_security_t0_done")
                        try:
                            sec = await self._fetch_gmgn_token_security(
                                obs.token.mint_address, obs.token.symbol)
                            if sec:
                                self.db.update_observation_snapshot(
                                    obs.obs_id, "gmgn_security_t0", sec)
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
        """T+5min shadow scoring for v4 (Hopeless Trajectory) rug filter.

        For each token in the observation pool where:
          - elapsed >= window_sec (default 300 = T+5min)
          - v4 has not scored this mint yet (per DB check)
        extract 5 features from bot's swaps table + score + log.

        SHADOW ONLY during 7-14 day validation phase.
        """
        if self.rug_filter_v4_model is None:
            return
        if not self.config.shadow_rug_v4_enabled:
            return
        window_s = int(self.config.shadow_rug_v4_window_sec)
        now = time.time()
        for mint, obs in list(self._observation_pool.items()):
            grad_t = obs.token.graduation_time
            elapsed = now - grad_t
            # Fire once per token, no earlier than window_s post-grad, and only
            # if the sample window is complete (+5s safety margin for bot lag).
            if elapsed < window_s + 5:
                continue
            if getattr(obs, "_rug_v4_scored", False):
                continue
            # Idempotency: skip if already in DB from a previous restart
            try:
                if self.db.has_shadow_rug_v4_eval(mint):
                    obs._rug_v4_scored = True
                    continue
            except Exception:
                pass
            try:
                db_path = self.db.db_path if hasattr(self.db, "db_path") else None
                if not db_path:
                    break
                # Q1 fix (2026-05-06): force in-memory swaps to DB before scoring.
                # Without this, v4 reads stale DB (last checkpoint flush ~10min ago,
                # since tick rate is 1/3.6s and flush triggers every 180 ticks).
                # See V5_6_PAIN_POINT_REPORT.md / D2 lag audit.
                try:
                    self._flush_swaps_to_db(mint, unregister=False)
                except Exception as _flush_err:
                    logger.debug(f"v4 pre-score flush failed: {_flush_err}")
                result = self.rug_filter_v4_model.score_from_sqlite(
                    db_path=db_path,
                    mint_address=mint,
                    graduation_time=grad_t,
                )
                self.db.record_shadow_rug_v4_eval(
                    mint_address=mint,
                    symbol=obs.token.symbol,
                    graduation_time=grad_t,
                    scored_at_delay_s=elapsed,
                    window_s=window_s,
                    n_swaps=result.n_swaps,
                    r1_flag=int(result.r1_flag),
                    r4_proba=(None if result.r4_proba != result.r4_proba
                              else result.r4_proba),
                    cutoff=self.rug_filter_v4_model.cutoff,
                    decision=result.decision,
                    reason=result.reason,
                    features=result.features or None,
                    model_version=self.rug_filter_v4_model.VERSION,
                )
                proba_s = (f"{result.r4_proba:.3f}"
                           if result.r4_proba == result.r4_proba else "NaN")
                logger.info(
                    f"RUG-V4-SHADOW: {obs.token.symbol} "
                    f"r1={int(result.r1_flag)} r4_proba={proba_s} "
                    f"cutoff={self.rug_filter_v4_model.cutoff:.2f} "
                    f"n_swaps={result.n_swaps} decision={result.decision}"
                )
                obs._rug_v4_scored = True
            except Exception as e:
                logger.debug(
                    f"RUG-V4-SHADOW: {obs.token.symbol} scoring failed: {e}"
                )
                obs._rug_v4_scored = True  # don't retry on hard errors

    async def _run_shadow_rug_filter_v4_2(self):
        """T+5min shadow scoring for v4.2 v0.9i hybrid (raw XGB prob).

        SHADOW ONLY — does NOT change buy/sell decisions.
        Logs to `shadow_rug_filter_v4_2_evals` for offline analysis.

        Spec: model_specs/2026-05-08_rug_filter_v4_2_SPEC.md v0.9f
        """
        if self.rug_filter_v4_2_model is None:
            return
        if not self.config.shadow_rug_v4_2_enabled:
            return
        window_s = int(self.config.shadow_rug_v4_2_window_sec)
        now = time.time()
        for mint, obs in list(self._observation_pool.items()):
            grad_t = obs.token.graduation_time
            elapsed = now - grad_t
            if elapsed < window_s + 5:
                continue
            if getattr(obs, "_rug_v4_2_scored", False):
                continue
            # Idempotency: skip if already in DB from a previous restart
            try:
                if self.db.has_shadow_rug_v4_2_eval(mint):
                    obs._rug_v4_2_scored = True
                    continue
            except Exception:
                pass
            try:
                db_path = self.db.db_path if hasattr(self.db, "db_path") else None
                if not db_path:
                    break
                # Force in-memory swaps to DB before scoring (matches v4 pattern)
                try:
                    self._flush_swaps_to_db(mint, unregister=False)
                except Exception as _flush_err:
                    logger.debug(f"v4.2 pre-score flush failed: {_flush_err}")
                result = self.rug_filter_v4_2_model.score_from_sqlite(
                    db_path=db_path,
                    mint_address=mint,
                    graduation_time=grad_t,
                )
                self.db.record_shadow_rug_v4_2_eval(
                    mint_address=mint,
                    symbol=obs.token.symbol,
                    graduation_time=grad_t,
                    scored_at_delay_s=elapsed,
                    window_s=window_s,
                    n_swaps=result.n_swaps,
                    score=(None if result.score != result.score
                            else result.score),
                    cutoff=self.rug_filter_v4_2_model.cutoff,
                    decision=result.decision,
                    reason=result.reason,
                    features=result.features or None,
                    model_version=self.rug_filter_v4_2_model.VERSION,
                )
                score_s = (f"{result.score:.4f}"
                            if result.score == result.score else "NaN")
                logger.info(
                    f"RUG-V4_2-SHADOW: {obs.token.symbol} "
                    f"score={score_s} cutoff={self.rug_filter_v4_2_model.cutoff:.4f} "
                    f"n_swaps={result.n_swaps} decision={result.decision}"
                )
                obs._rug_v4_2_scored = True
            except Exception as e:
                logger.debug(
                    f"RUG-V4_2-SHADOW: {obs.token.symbol} scoring failed: {e}"
                )
                obs._rug_v4_2_scored = True

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
                info_60m = await self._fetch_gmgn_token_info(orph["mint"], orph["symbol"])
                security = await self._fetch_gmgn_token_security(orph["mint"], orph["symbol"])
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
                    gmgn_info_60m=info_60m, gmgn_security=security)
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
                # Final poll to get latest swaps before building kline
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
        """Execution-time guardrails before sending a buy to Gateway."""
        token = candidate.token
        preflight_start = time.time()  # V3: measure preflight latency end-to-end
        logger.info(f"M3: [PREFLIGHT START] {token.symbol} model_score={candidate.model_score:.3f}")

        # === CHECK 0 (pre-snapshot): Rug filter gate ===
        # Reuses the t0 rug score already computed by the shadow path and
        # stored in shadow_rug_evals. Model was trained on t0 GMGN snapshot
        # features (32-dim, test AUC 0.7201, top5 precision 79% on rugs).
        # Only blocks when rug_filter_gate_enabled=true and score exceeds
        # the configured band's cutoff — a "soft filter" that passes the
        # strictest rejects to BLOCK and lets everything else through the
        # normal 7-check pipeline.
        if self.config.rug_filter_gate_enabled and self.rug_filter_model is not None:
            try:
                rug_eval = self.db.get_latest_shadow_rug_eval(token.mint_address)
            except Exception as e:
                rug_eval = None
                logger.debug(f"M3: rug gate lookup error {token.symbol}: {e}")
            if rug_eval:
                gate_band = self.config.rug_filter_gate_cutoff_band
                # Use the live model's cutoff (same pkl as shadow), not the
                # stored cutoff_value (which was for shadow's looser band).
                gate_cutoff = self.rug_filter_model.cutoffs.get(gate_band, 1.0)
                if rug_eval["score"] >= gate_cutoff:
                    reason = (f"rug_gate score={rug_eval['score']:.3f}>="
                              f"{gate_band}={gate_cutoff:.3f}")
                    logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                    return False, None, reason, {}
                logger.info(f"M3: [CHECK 0] {token.symbol} rug_gate ✓ "
                            f"score={rug_eval['score']:.3f} < {gate_band}={gate_cutoff:.3f}")

        # Plan B: launch GMGN snapshot and Jupiter quote concurrently —
        # they're independent network calls, so doing them in parallel saves
        # 300-800ms in the chase window between M2 decision and entry.
        snapshot_task = asyncio.create_task(
            self.discovery.get_token_market_snapshot(token.mint_address))
        quote_task = asyncio.create_task(
            self.trader.quote_buy(token.mint_address, sol_amount))

        def _cancel_quote():
            if not quote_task.done():
                quote_task.cancel()

        # === Check 1: live GMGN snapshot ===
        try:
            snapshot = await snapshot_task
        except Exception as e:
            logger.debug(f"M3: snapshot fetch error for {token.symbol}: {e}")
            snapshot = None
        if self.config.require_live_market_snapshot and not snapshot:
            reason = "live_market_snapshot_unavailable"
            logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
            _cancel_quote()
            return False, None, reason, {}
        logger.info(f"M3: [CHECK 1/7] {token.symbol} live_snapshot ✓ "
                    f"(symbol={snapshot.get('symbol','?') if snapshot else 'N/A'})")

        effective_liq = token.liquidity_usd if token.liquidity_usd > 0 else 0.0
        if snapshot:
            live_liq = float(snapshot.get("liquidity_usd") or 0.0)
            if live_liq > 0:
                effective_liq = live_liq
                token.liquidity_usd = live_liq
            biggest_pool = snapshot.get("biggest_pool_address")
            if biggest_pool:
                token.pool_address = biggest_pool

        if effective_liq < self.config.min_liquidity_usd:
            reason = (f"live_liquidity=${effective_liq:.0f}<${self.config.min_liquidity_usd:.0f}")
            logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
            self.db.record_preflight_check(
                mint_address=token.mint_address, symbol=token.symbol,
                check_name="live_liquidity", outcome="block",
                value=effective_liq, threshold=float(self.config.min_liquidity_usd),
                model_score=candidate.model_score, pool_liq_usd=effective_liq)
            _cancel_quote()
            return False, None, reason, {}
        self.db.record_preflight_check(
            mint_address=token.mint_address, symbol=token.symbol,
            check_name="live_liquidity", outcome="pass",
            value=effective_liq, threshold=float(self.config.min_liquidity_usd),
            model_score=candidate.model_score, pool_liq_usd=effective_liq)
        logger.info(f"M3: [CHECK 2/7] {token.symbol} live_liquidity ✓ "
                    f"(${effective_liq:,.0f} >= ${self.config.min_liquidity_usd:,.0f})")

        # === Check 3: swap-vs-GMGN price double-check ===
        price_check_done = False
        if snapshot and candidate.last_swap_price_sol > 0:
            swap_price_usd = candidate.last_swap_price_sol * (self._sol_price_usd if self._sol_price_usd > 0 else 80.0)
            gmgn_price = float(snapshot.get("price_usd") or 0)
            if swap_price_usd > 0 and gmgn_price > 0:
                divergence = abs(swap_price_usd - gmgn_price) / gmgn_price
                if divergence > 0.50:
                    reason = (f"price_divergence={divergence:.1%} "
                              f"(swap=${swap_price_usd:.10f} gmgn=${gmgn_price:.10f})")
                    logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                    self.db.record_preflight_check(
                        mint_address=token.mint_address, symbol=token.symbol,
                        check_name="price_divergence", outcome="block",
                        value=float(divergence), threshold=0.50,
                        model_score=candidate.model_score, pool_liq_usd=effective_liq)
                    _cancel_quote()
                    return False, None, reason, {}
                logger.info(f"M3: [CHECK 3/7] {token.symbol} price_double_check ✓ "
                            f"divergence={divergence:.1%} "
                            f"(swap=${swap_price_usd:.8f} gmgn=${gmgn_price:.8f})")
                price_check_done = True
        if not price_check_done:
            logger.info(f"M3: [CHECK 3/7] {token.symbol} price_double_check SKIP "
                        f"(snapshot={bool(snapshot)} cached_swap_price={candidate.last_swap_price_sol})")

        # Re-check extreme crash filters at execution time (ALL models).
        is_trending = getattr(token, "source", "chainstack") == "trending"
        ret_5m = float(candidate.features.get("5min_return",
                       candidate.features.get("cum_return_t0_to_entry", 0.0)) or 0.0)
        vol_5m = float(candidate.features.get("5min_volume_usd",
                       candidate.features.get("cum_volume_t0_to_entry", 0.0)) or 0.0)
        vol_liq_ratio = vol_5m / max(effective_liq, 1.0)
        cum_dd = float(candidate.features.get("cum_drawdown_t0_to_entry", 0.0) or 0.0)
        lb_ret = float(candidate.features.get("lookback_return", 0.0) or 0.0)

        if not is_trending:
            if ret_5m < self.config.max_entry_return_floor:
                reason = f"cum_return={ret_5m:.3f}<{self.config.max_entry_return_floor} (extreme crash at M3)"
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                _cancel_quote()
                return False, None, reason, {}
            if abs(cum_dd) > self.config.max_cum_drawdown_at_entry:
                reason = f"cum_drawdown={cum_dd:.3f}>{self.config.max_cum_drawdown_at_entry} (near-total crash at M3)"
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                _cancel_quote()
                return False, None, reason, {}
            if lb_ret < self.config.lookback_return_floor:
                reason = (f"lookback_return={lb_ret:.3f}<"
                          f"{self.config.lookback_return_floor} "
                          f"(>50% drop in last 3 bars at M3)")
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                _cancel_quote()
                return False, None, reason, {}
            logger.info(f"M3: [CHECK 4/7] {token.symbol} crash_filter ✓ "
                        f"cum_return={ret_5m:+.1%} cum_dd={cum_dd:+.1%} "
                        f"lb_ret={lb_ret:+.1%} vol/liq={vol_liq_ratio:.2f}")
        else:
            logger.info(f"M3: [CHECK 4/7] {token.symbol} crash_filter ✓ "
                        f"[trending] 5min_return={ret_5m:+.1%} vol/liq={vol_liq_ratio:.2f} "
                        f"(bypassed)")

        # === Check 5: Jupiter quote (awaits the task we launched in parallel with snapshot) ===
        try:
            quote = await quote_task
        except Exception as e:
            reason = f"jupiter_quote_error: {str(e)[:80]}"
            logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
            return False, None, reason, {}
        quote_id = quote.get("quoteId") or quote.get("quote_id")
        amount_in = float(quote.get("amountIn") or quote.get("amount_in") or sol_amount)
        amount_out = float(quote.get("amountOut") or quote.get("amount_out") or 0)
        route_labels = self.trader.extract_route_labels(quote)
        route_amm_keys = self.trader.extract_route_amm_keys(quote)
        route_summary = " > ".join(route_labels) if route_labels else "unknown"
        if not quote_id or amount_out <= 0:
            reason = f"invalid_quote amount_out={amount_out:.8f} quote_id={bool(quote_id)}"
            logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
            return False, None, reason, {}
        logger.info(f"M3: [CHECK 5/7] {token.symbol} jupiter_quote ✓ "
                    f"route={route_summary} amount_out={amount_out:.0f}")

        # === Entry chase filter: reject if Jupiter exec price has drifted
        # significantly above the M2 reference price (kline close at eval time).
        # Catches cases where price ran up 30%+ between M2 decision and
        # M3 execution — i.e. we'd be buying the top of a spike.
        # Live data (15 trades): avg entry slippage was +53%, with 3 trades
        # at >+100% chase leading to immediate losses. This gates that out.
        implied_exec_price = amount_in / amount_out
        m2_ref = candidate.last_swap_price_sol
        if m2_ref and m2_ref > 0:
            chase = implied_exec_price / m2_ref - 1
            if chase > self.config.max_entry_chase_pct:
                reason = (f"entry_chase={chase:+.1%}>{self.config.max_entry_chase_pct:+.1%} "
                          f"(M2_ref={m2_ref:.10f} jup_exec={implied_exec_price:.10f})")
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                self.db.record_preflight_check(
                    mint_address=token.mint_address, symbol=token.symbol,
                    check_name="entry_chase", outcome="block",
                    value=float(chase), threshold=float(self.config.max_entry_chase_pct),
                    model_score=candidate.model_score, pool_liq_usd=effective_liq)
                return False, None, reason, {}
            self.db.record_preflight_check(
                mint_address=token.mint_address, symbol=token.symbol,
                check_name="entry_chase", outcome="pass",
                value=float(chase), threshold=float(self.config.max_entry_chase_pct),
                model_score=candidate.model_score, pool_liq_usd=effective_liq)
            logger.info(f"M3: [CHECK 5.5/7] {token.symbol} entry_chase ✓ "
                        f"{chase:+.1%} (≤{self.config.max_entry_chase_pct:+.1%})")

        # === Check 6: route uses biggest pool ===
        if self.config.require_route_uses_biggest_pool:
            expected_pool = token.pool_address
            if not expected_pool:
                reason = "missing_biggest_pool_address"
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                return False, None, reason, {}
            if not route_amm_keys:
                reason = "quote_route_missing_amm_keys"
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason}")
                return False, None, reason, {}
            if expected_pool not in route_amm_keys:
                reason = f"route_missing_biggest_pool expected={expected_pool}"
                logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason} route={route_summary}")
                return False, None, reason, {}
            logger.info(f"M3: [CHECK 6/7] {token.symbol} route_uses_biggest_pool ✓ "
                        f"({expected_pool[:12]}...)")
        else:
            logger.info(f"M3: [CHECK 6/7] {token.symbol} route_uses_biggest_pool SKIP (disabled)")

        # === Check 7: price impact (local, from quote) ===
        # Plan D: removed the live Jupiter Price API / Gecko mid-price fetch.
        # That was a 500-1000ms network call that essentially re-checked what
        # CHECK 5.5 (entry_chase) already verifies via the M2 reference price.
        # We keep the price_impact check (local, from quote body) since that's
        # the Jupiter-reported AMM impact — independent signal.
        # Mid-price fields kept in preflight_meta for DB compatibility: we
        # fall back to the M2 reference (kline close at eval time) which is
        # the closest "fair" price we have.
        price_impact = self.trader.extract_quote_price_impact_pct(quote)
        if price_impact is not None and price_impact > self.config.max_buy_price_impact_pct:
            reason = f"price_impact={price_impact:.3f}>{self.config.max_buy_price_impact_pct:.3f}"
            logger.warning(f"M3: [BUY BLOCKED] {token.symbol} — {reason} route={route_summary}")
            self.db.record_preflight_check(
                mint_address=token.mint_address, symbol=token.symbol,
                check_name="price_impact", outcome="block",
                value=float(price_impact), threshold=float(self.config.max_buy_price_impact_pct),
                model_score=candidate.model_score, pool_liq_usd=effective_liq)
            return False, None, reason, {}

        # Use M2 reference as the "mid" for reporting purposes (no network call).
        mid_price = m2_ref if m2_ref and m2_ref > 0 else implied_exec_price
        mid_source = "m2_ref" if m2_ref and m2_ref > 0 else "exec_only"
        premium = (implied_exec_price / mid_price - 1) if mid_price > 0 else 0.0

        logger.info(f"M3: [CHECK 7/7] {token.symbol} impact ✓ "
                    f"impact={(price_impact or 0):.1%} chase_vs_M2={premium:+.1%} "
                    f"exec={implied_exec_price:.10f} mid={mid_price:.10f} src={mid_source}")

        preflight_latency_ms = int((time.time() - preflight_start) * 1000)
        self.db.record_preflight_check(
            mint_address=token.mint_address, symbol=token.symbol,
            check_name="all_pass", outcome="pass",
            value=float(preflight_latency_ms),
            detail=f"impact={(price_impact or 0):.3f} chase={premium:+.3f}",
            model_score=candidate.model_score, pool_liq_usd=effective_liq)
        logger.info(f"M3: [PREFLIGHT PASS] {token.symbol} all 7 checks ✓ "
                    f"(latency={preflight_latency_ms}ms) — proceeding to execute_buy")
        preflight_meta = {
            "quote_id": quote_id,
            "route_summary": route_summary,
            "amount_in_sol": amount_in,
            "amount_out_token": amount_out,
            "quote_price_impact_pct": price_impact,
            "implied_exec_price_sol": implied_exec_price,
            "quote_exec_price_sol": implied_exec_price,
            "mid_price_sol": mid_price,
            "decision_mid_price_sol": mid_price,
            "mid_source": mid_source,
            "decision_mid_source": mid_source,
            "premium_vs_mid_pct": premium,
            # V3 analysis fields
            "m2_ref_price_sol": m2_ref,
            "preflight_latency_ms": preflight_latency_ms,
            "pool_liq_at_entry_usd": effective_liq,
        }
        return True, quote, "", preflight_meta

    # Valid sizing modes — see _compute_position_size for semantics
    _SIZING_MODES_VALID = {"fixed", "slippage_budget"}

    def _compute_position_size(self, candidate: "TradeCandidate") -> Tuple[float, dict]:
        """Phase 22.G.2 capacity-aware sizing.

        Returns (size_usd, meta_dict) — meta dict is recorded to DB events
        and serialized into the buy log. Two modes:

          fixed             — legacy: uses candidate.position_size_usd override
                              if present, else config.position_size_usd.
          slippage_budget   — S_usd = clamp(β · L_sol · sol_price,
                                            min_usd, max_usd)
                              where L_sol = pool_liq_usd / sol_price / 2.
                              Theory: CPMM x*y=k → one-way slippage = ΔX/X
                              exactly (Angeris et al. 2020 §3).

        ⚠️ Stale-liquidity caveat: pool_liq comes from token.liquidity_usd
        which was captured at M1 discovery (T+0). v3.3 fires at T+7-15min
        and the pool may have shifted significantly. Step 2 enhancement
        will refresh pool_liq from on-chain reserves pre-buy.
        """
        cfg = self.config
        token = candidate.token
        override_size = getattr(candidate, "position_size_usd", None)
        if override_size is not None:
            try:
                override_size = float(override_size)
            except (TypeError, ValueError):
                override_size = None

        mode = str(getattr(cfg, "sizing_mode", "fixed") or "fixed").lower()
        if mode not in self._SIZING_MODES_VALID:
            logger.warning(
                f"M3: unknown sizing_mode={mode!r}, falling back to 'fixed'. "
                f"Valid: {sorted(self._SIZING_MODES_VALID)}")
            mode = "fixed"

        if mode == "fixed":
            sizing_usd = (override_size if override_size is not None
                          else float(cfg.position_size_usd))
            return sizing_usd, {
                "mode": "fixed",
                "size_usd": sizing_usd,
                "src_override_usd": override_size,
            }

        # mode == "slippage_budget"
        sol_price = self._sol_price_usd
        pool_liq = float(getattr(token, "liquidity_usd", 0) or 0)
        if sol_price <= 0 or pool_liq <= 0:
            sizing_usd = (override_size if override_size is not None
                          else float(cfg.position_size_usd))
            return sizing_usd, {
                "mode": "slippage_budget",
                "fallback": "fixed",
                "fallback_reason": f"sol_price={sol_price} liq={pool_liq}",
                "size_usd": sizing_usd,
            }

        # Validate config — clamp into safe ranges (do NOT silently use bad values)
        beta = float(getattr(cfg, "sizing_slippage_budget", 0.005) or 0.005)
        min_usd = float(getattr(cfg, "sizing_min_usd", 5.0) or 5.0)
        max_usd = float(getattr(cfg, "sizing_max_usd", 30.0) or 30.0)
        clamp_warn = []
        if not (0.0 < beta <= 0.05):  # 5% one-way is already extreme
            old_beta = beta
            beta = max(0.0001, min(beta, 0.05))
            clamp_warn.append(f"β {old_beta}→{beta}")
        # 2026-05-11: lowered hard floor 1.0 → 0.05 for canary/validation
        # mode. $0.05 = ~0.0005 SOL ≈ 50× Solana tx fee margin (well above
        # dust). Original $1.00 floor silently overrode user yml configs
        # < $1 (e.g. sizing_min_usd=0.1 → forced to 1.0) — caught when v5.5
        # canary at $0.10 actually traded at $1. Keep floor non-zero so
        # accidental yml typos don't trigger dust-tier swap failures.
        DUST_FLOOR_USD = 0.05
        if min_usd < DUST_FLOOR_USD:
            clamp_warn.append(f"min ${min_usd}→${DUST_FLOOR_USD:.2f} (floor)")
            min_usd = DUST_FLOOR_USD
        if max_usd < min_usd:
            clamp_warn.append(f"max ${max_usd}<min — set max=min=${min_usd}")
            max_usd = min_usd

        l_sol = pool_liq / sol_price / 2.0
        s_sol_raw = beta * l_sol
        s_usd_raw = s_sol_raw * sol_price
        s_usd_clamped = min(max(s_usd_raw, min_usd), max_usd)
        if s_usd_raw > max_usd:
            cap_reason = "max_cap"
        elif s_usd_raw < min_usd:
            cap_reason = "min_floor"
        else:
            cap_reason = "beta_binding"

        # Per-source override acts as additional hard cap if requested
        # (NOTE: default is now False — slippage_budget formula is the single
        # source of truth. Set sizing_respect_source_override_as_cap=true in
        # yml to re-enable per-source caps like BigWinner canary sizing.)
        s_usd = s_usd_clamped
        if (getattr(cfg, "sizing_respect_source_override_as_cap", False)
                and override_size is not None
                and override_size > 0
                and override_size < s_usd):
            s_usd = override_size
            cap_reason = f"src_override_cap=${override_size:.2f}"

        meta = {
            "mode": "slippage_budget",
            "beta": beta,
            "liq_usd": pool_liq,
            "sol_price_usd": sol_price,
            "l_sol": l_sol,
            "size_usd_raw": s_usd_raw,
            "size_usd_clamped": s_usd_clamped,
            "size_usd": s_usd,
            "cap_reason": cap_reason,
            "src_override_usd": override_size,
            "min_usd": min_usd,
            "max_usd": max_usd,
        }
        if clamp_warn:
            meta["config_clamp"] = "; ".join(clamp_warn)
            logger.warning(f"M3: sizing config clamped: {meta['config_clamp']}")
        return s_usd, meta

    async def _execute_buy(self, candidate: TradeCandidate) -> bool:
        """M3: register token + swap buy. Returns True on success, False on failure."""
        token = candidate.token
        try:
            # Step 1: register token in Gateway
            self._record_latency_event(token, "buy_register_start")
            logger.info(f"M3: [BUY] registering {token.symbol} ({token.mint_address}) in Gateway")
            reg_result = await self.trader.register_token(
                mint_address=token.mint_address,
                symbol=token.symbol,
                name=token.name,
                decimals=token.decimals,
            )
            logger.debug(f"M3: register result: {reg_result}")
            self._record_latency_event(token, "buy_register_done")

            # Phase 22.S.1 (2026-04-30) — Jupiter index wait, configurable.
            # Default 0.0 — token is already 3-15min old at M3, Jupiter has
            # indexed long ago. If config > 0 fall back to legacy behavior.
            # See `buy_jupiter_wait_sec` in MemeSniperConfig.
            jup_wait = float(getattr(self.config, "buy_jupiter_wait_sec", 0.0) or 0.0)
            if jup_wait > 0:
                await asyncio.sleep(jup_wait)

            # Step 2: calculate SOL amount — fail-closed if price unknown
            if self._sol_price_usd <= 0:
                await self._refresh_sol_price()
            if self._sol_price_usd <= 0:
                logger.error(f"M3: [BUY BLOCKED] SOL price unavailable — cannot size position for {token.symbol}")
                self.db.record_event("WARN", "buy", f"BUY BLOCKED {token.symbol}: SOL price unavailable")
                return False
            # Position sizing — fixed (legacy) or slippage-budget (Phase 22.G.2).
            sizing_usd, sizing_meta = self._compute_position_size(candidate)
            if sizing_usd <= 0:
                logger.error(f"M3: [BUY BLOCKED] sizing returned $0 for "
                             f"{token.symbol}: {sizing_meta}")
                self.db.record_event("WARN", "buy",
                                     f"BUY BLOCKED {token.symbol}: sizing=0 ({sizing_meta})")
                return False
            sol_amount = sizing_usd / self._sol_price_usd
            # Compact meta string for logging
            sizing_meta_str = " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sizing_meta.items() if v is not None)
            if sol_amount <= 0:
                logger.error(f"M3: invalid sol_amount={sol_amount}, sol_price={self._sol_price_usd}")
                return False

            # Step 2b: verify wallet has sufficient SOL.
            # Fee buffer scales with sol_amount: max(0.01, 0.5% of trade size).
            # At sol_amount=2.5 (≈$200 @ $80/SOL), buffer = 0.0125 (covers
            # priority fees + tip on PumpSwap with congestion).
            fee_buffer_sol = max(0.01, sol_amount * 0.005)
            sol_balance = await self.trader.get_sol_balance()
            if sol_balance is None:
                self._record_latency_event(token, "buy_balance_unavailable")
                logger.error(f"M3: [BUY BLOCKED] SOL balance unavailable — cannot verify wallet funds for {token.symbol}")
                self.db.record_event("WARN", "buy",
                                     f"BUY BLOCKED {token.symbol}: SOL balance unavailable")
                return False
            if sol_balance < sol_amount + fee_buffer_sol:
                self._record_latency_event(token, "buy_balance_insufficient", sol_balance=sol_balance)
                logger.error(f"M3: [BUY BLOCKED] insufficient SOL balance: "
                             f"{sol_balance:.4f} < {sol_amount:.4f} + {fee_buffer_sol:.4f} fee buffer")
                self.db.record_event("WARN", "buy",
                                     f"BUY BLOCKED {token.symbol}: insufficient SOL {sol_balance:.4f}")
                return False

            # Step 3: validate a live quote before sending the buy.
            quote_ok, quote, reject_reason, preflight_meta = await self._preflight_buy(candidate, sol_amount)
            if not quote_ok:
                self._record_latency_event(token, "buy_preflight_fail", reject_reason=reject_reason)
                self.db.record_event("WARN", "buy",
                                     f"BUY BLOCKED {token.symbol}: {reject_reason}")
                return False
            self._record_latency_event(token, "buy_preflight_pass", **preflight_meta)

            # Step 4: execute the exact validated quote
            _src = getattr(candidate, "entry_source", "") or "vshape"
            logger.info(f"M3: [BUY] {token.symbol} — {sol_amount:.4f} SOL (~${sizing_usd:.2f}) "
                        f"| src={_src} | model_score={candidate.model_score:.3f} "
                        f"| liq=${token.liquidity_usd:.0f} | sizing[{sizing_meta_str}]")
            # Record sizing telemetry so backtests can compare formulas later.
            try:
                self.db.record_event(
                    "INFO", "sizing",
                    f"{token.symbol} src={_src} {sizing_meta_str}")
            except Exception as _e_sz:
                logger.debug(f"M3: sizing event record failed: {_e_sz}")
            quote_id = quote.get("quoteId") or quote.get("quote_id")
            self._record_latency_event(token, "buy_submit", **{**preflight_meta, "quote_id": quote_id})
            result = await self.trader.execute_quote(quote_id)
            logger.debug(f"M3: swap_buy result: {result}")

            # Check on-chain transaction status
            tx_status = result.get("status", 1)
            if tx_status == 0:
                self._record_latency_event(token, "buy_onchain_fail", tx_status=tx_status)
                buy_tx = result.get("txHash", result.get("signature", "unknown"))
                logger.error(f"M3: [BUY FAILED ON-CHAIN] {token.symbol} — tx={buy_tx} status=0")
                self.db.record_event("ERROR", "buy",
                                     f"BUY ON-CHAIN FAIL {token.symbol} tx={buy_tx}")
                return False

            # Parse result — Gateway may nest swap data under "data" key
            swap_data = result.get("data", result)
            token_amount = float(swap_data.get("amountOut", swap_data.get("amount", 0)))
            actual_sol_spent = float(swap_data.get("amountIn", sol_amount))
            tx_hash = result.get("txHash", result.get("signature", "unknown"))

            # Sanity: reject zero-token buys (status=1 but no tokens received)
            if token_amount <= 0:
                self._record_latency_event(token, "buy_zero_tokens", tx_hash=tx_hash)
                logger.error(f"M3: [BUY ZERO TOKENS] {token.symbol} — amountOut=0, "
                             f"sol_spent={actual_sol_spent:.4f}, tx={tx_hash}")
                self.db.record_event("ERROR", "buy",
                                     f"BUY ZERO TOKENS {token.symbol} tx={tx_hash}")
                return False

            # Use the intended swap amount (sol_amount) for price, not
            # actual_sol_spent which includes priority fees.  Priority fees
            # are a committed cost already captured in sol_invested / pnl_sol,
            # but inflating entry_price_sol by 2-3× causes pnl_pct to read
            # ~-68% immediately, which fires EC EMERGENCY on every trade.
            fee_overhead_sol = actual_sol_spent - sol_amount
            swap_entry_price = sol_amount / token_amount

            # Calibrate entry_price using the same source as M4 monitoring
            # (batch API mid-price), so stop-loss isn't triggered by buy slippage.
            # TOLL case (2026-04-07): mid was 1.45x swap → entry inflated → instant -32% stop
            # Tightened calibration to ±15% (typical meme spread is < 10%, > 15% means mid is bad)
            entry_price = swap_entry_price
            try:
                bp = await self.trader.get_batch_prices(
                    [token.mint_address], self._sol_price_usd)
                mid_price = bp.get(token.mint_address)
                if mid_price and mid_price > 0:
                    ratio = mid_price / swap_entry_price
                    if ratio < 0.85 or ratio > 1.15:
                        logger.warning(f"M3: entry calibration REJECTED — mid={mid_price:.10f} "
                                       f"is {ratio:.2f}x swap={swap_entry_price:.10f} "
                                       f"(diff={(ratio-1)*100:+.0f}% > ±15%), using swap price")
                    else:
                        entry_price = mid_price
                        logger.info(f"M3: entry calibrated: swap={swap_entry_price:.10f} "
                                    f"mid={mid_price:.10f} diff={mid_price/swap_entry_price - 1:+.1%}")
            except Exception as e:
                logger.warning(f"M3: entry calibration failed, using swap price: {e}")

            position = Position(
                token=token,
                entry_price_sol=entry_price,
                token_amount=token_amount,
                entry_time=time.time(),
                entry_tx=tx_hash,
                sol_invested=actual_sol_spent,
                model_score=candidate.model_score,
                features=candidate.features,
                entry_quote_id=preflight_meta.get("quote_id"),
                entry_quote_exec_price_sol=preflight_meta.get("quote_exec_price_sol"),
                entry_decision_mid_price_sol=preflight_meta.get("decision_mid_price_sol"),
                entry_decision_mid_source=preflight_meta.get("decision_mid_source"),
                m2_ref_price_sol=preflight_meta.get("m2_ref_price_sol"),
                preflight_latency_ms=preflight_meta.get("preflight_latency_ms"),
                pool_liq_at_entry_usd=preflight_meta.get("pool_liq_at_entry_usd"),
                pool_sol_reserve_at_entry=(
                    (preflight_meta.get("pool_liq_at_entry_usd") or 0.0)
                    / max(self._sol_price_usd, 1e-9) / 2.0
                    if (preflight_meta.get("pool_liq_at_entry_usd") or 0) > 0
                       and self._sol_price_usd > 0 else None
                ),
                slippage_per_pool_ratio=(
                    actual_sol_spent / (
                        (preflight_meta.get("pool_liq_at_entry_usd") or 0.0)
                        / max(self._sol_price_usd, 1e-9) / 2.0
                    )
                    if (preflight_meta.get("pool_liq_at_entry_usd") or 0) > 0
                       and self._sol_price_usd > 0 else None
                ),
                entry_source=getattr(candidate, "entry_source", ""),
            )
            # Phase 16.3 audit-6 fix — DB-first persistence ordering.
            # Previous order (memory append → DB save) lost in-memory positions
            # if save_position raised: zombie wallet position with no record.
            # New order: persist first, then track in memory. If memory append
            # fails (very rare), DB row remains and is recovered on restart.
            try:
                self.db.save_position(position)
            except Exception as e:
                logger.error(
                    f"M3: [SAVE FAIL] {token.symbol} — DB save_position raised: {e}. "
                    f"Trade was executed (tx={tx_hash}) but persistence failed. "
                    f"Manual reconciliation may be needed.")
                self.db.record_event("ERROR", "buy",
                                     f"save_position FAIL {token.symbol}: {e}")
                # Still register in memory so M4 monitors the position; next
                # peak/trailing tick will retry save_position.
            self._positions.append(position)
            # Phase 16.3 Step 3 — bump canary counter so the live cap fires
            # even before the position has been closed (sized at buy-time,
            # not exit-time, since open-position EV ≈ deployed capital).
            if self._is_big_winner_entry_source(position.entry_source):
                self._big_winner_trade_count += 1
            self.risk._last_trade_time = time.time()
            # Subscribe pool to gRPC rug-event watcher UNCONDITIONALLY.
            # The flag `rug_event_exit_enabled` only gates the actual panic
            # sell action — events are always recorded to `rug_events` so
            # we can measure rug frequency / magnitude / timing versus M4's
            # poll-based detection before committing to live exits.
            if self._grpc_stream is not None and token.pool_address:
                self._grpc_stream.watch_pool(token.pool_address, token.mint_address)
            fill_vs_quote_slippage_pct = self._safe_relative_delta(
                swap_entry_price, preflight_meta.get("quote_exec_price_sol"))
            fill_vs_decision_mid_pct = self._safe_relative_delta(
                swap_entry_price, preflight_meta.get("decision_mid_price_sol"))
            self._record_latency_event(
                token, "buy_confirm", tx_hash=tx_hash,
                sol_spent=actual_sol_spent, token_amount=token_amount,
                fill_price_sol=swap_entry_price,
                calibrated_entry_price_sol=entry_price,
                quote_id=preflight_meta.get("quote_id"),
                quote_exec_price_sol=preflight_meta.get("quote_exec_price_sol"),
                decision_mid_price_sol=preflight_meta.get("decision_mid_price_sol"),
                decision_mid_source=preflight_meta.get("decision_mid_source"),
                fill_vs_quote_slippage_pct=fill_vs_quote_slippage_pct,
                fill_vs_decision_mid_pct=fill_vs_decision_mid_pct)

            logger.info(f"M3: [BUY OK] {token_amount:.0f} {token.symbol} @ {entry_price:.10f} SOL/token "
                        f"| swap={sol_amount:.4f} SOL fee_overhead={fee_overhead_sol:.5f} SOL"
                        f" total={actual_sol_spent:.4f} SOL | tx={tx_hash}")
            self.db.record_event("INFO", "buy",
                                 f"BUY {token.symbol} qty={token_amount:.0f} sol={actual_sol_spent:.4f} tx={tx_hash}")
            return True

        except Exception as e:
            self._record_latency_event(token, "buy_exception", error=str(e))
            logger.error(f"M3: [BUY FAIL] {token.symbol}: {e}", exc_info=True)
            self.db.record_event("ERROR", "buy", f"BUY FAIL {token.symbol}: {e}")
            return False

    async def _monitor_positions(self):
        """M4 + M5: check all positions for exit conditions."""
        if not self._positions:
            return

        now = time.time()

        # Faster polling when any position has trailing stop active
        has_trailing = (
            False
            if (self.config.ev3m_live_enabled and self.config.ev3m_live_disable_trailing)
            else any(p.trailing_activated for p in self._positions)
        )
        poll_interval = (self.config.price_poll_interval_trailing
                         if has_trailing else self.config.price_poll_interval)
        if now - self._last_monitor_time < poll_interval:
            # Still check time limits (no API call needed)
            time_exits = [pos for pos in self._positions
                          if pos.hold_seconds() >= self.config.time_limit_sec]
            for pos in time_exits:
                if self._defer_exit_for_retry(pos, "time_limit"):
                    continue
                self._record_sell_trigger_once(
                    pos, "time_limit", pnl_pct=0.0, trigger_pnl_pct=0.0,
                    trigger_price_sol=None, trigger_mid_price_sol=None,
                    trigger_mid_source=None)
                logger.info(f"M4: {pos.token.symbol} — time limit reached "
                            f"({pos.hold_seconds():.0f}s >= {self.config.time_limit_sec}s)")
                await self._execute_sell(pos, "time_limit")
            return

        self._last_monitor_time = now
        exits_to_process = []

        # Separate time-limit exits from positions needing price checks.
        # Phase 22.S.P1: skip positions currently being exited by gRPC watcher
        # to avoid double-fire / double-sell.
        price_check_positions = []
        for pos in self._positions:
            if pos.token.mint_address in self._exit_in_progress:
                continue
            hold_sec = pos.hold_seconds()
            if hold_sec >= self.config.time_limit_sec:
                if self._defer_exit_for_retry(pos, "time_limit"):
                    continue
                self._record_sell_trigger_once(
                    pos, "time_limit", pnl_pct=0.0, trigger_pnl_pct=0.0,
                    trigger_price_sol=None, trigger_mid_price_sol=None,
                    trigger_mid_source=None)
                logger.info(f"M4: {pos.token.symbol} — time limit reached ({hold_sec:.0f}s >= {self.config.time_limit_sec}s)")
                exits_to_process.append((pos, "time_limit", 0.0, None, None, None))
            else:
                price_check_positions.append(pos)

        # Fetch pool-backed (gRPC) and Jupiter prices for every position;
        # selection rules are applied per-position below.
        mints = [pos.token.mint_address for pos in price_check_positions]
        grpc_prices = (self._grpc_stream.get_swap_prices_batch(mints)
                       if (self._grpc_stream and self._grpc_stream.connected)
                       else {})
        batch_prices = (await self.trader.get_batch_prices(mints, self._sol_price_usd)
                        if mints else {})

        for pos in price_check_positions:
            hold_sec = pos.hold_seconds()

            # Grace period: still fetch price and enforce stop-loss, but defer
            # peak/trailing logic for the first 15s after entry.
            # Reduced from 60s: BabyVance went from -7% to -96% during old grace window.
            # Newly indexed tokens can report unstable early prices, so trailing
            # activation/trigger waits until the grace window has elapsed.
            trailing_grace_active = hold_sec < 15

            # gRPC pool median = ground truth (matches what a sell actually
            # hits). Jupiter Price API = fallback; prone to sandwich-tick
            # outliers (BK 2026-04-14: Jupiter reported peak +334%, pool
            # reality +47%). Peak / trail / TP triggers below require
            # price_source == "grpc_pool" so a one-off Jupiter spike cannot
            # inflate peak or force a late trail sell during the dump.
            mint = pos.token.mint_address
            grpc_sol = grpc_prices.get(mint)
            jup_px = batch_prices.get(mint)
            # UNIT CONTRACT: both pool_px and jup_px are SOL/token, same as
            # pos.entry_price_sol. Do NOT multiply by sol_price_usd here —
            # grpc stream stores price_sol (geyser_stream.py:399); multiplying
            # by sol_price_usd (~$83) produced the BFH bug where pool_px
            # looked 83× higher than jup_px, making pnl_pct ≈ +8000% and
            # spike_rejected blocking every poll.
            pool_px = grpc_sol if (grpc_sol and grpc_sol > 0) else None

            if pool_px is not None:
                current_price = pool_px
                price_source = "grpc_pool"
                if jup_px and jup_px > 0:
                    dev = abs(jup_px - pool_px) / pool_px
                    if dev > 0.30:
                        logger.info(
                            f"M4: {pos.token.symbol} — Jupiter/pool divergence "
                            f"{dev:+.1%} (jup={jup_px:.8f}, pool={pool_px:.8f}) "
                            f"— trusting pool")
            elif jup_px and jup_px > 0:
                current_price = jup_px
                price_source = "jupiter_fallback"
            else:
                current_price = None
                price_source = None

            # Forensic log: every monitor cycle, write both prices so BK-type
            # divergences can be reconstructed post-hoc. Skip if we have zero
            # signal (neither source returned a price — already logged above).
            if pool_px is not None or (jup_px and jup_px > 0):
                entry_sol = pos.entry_price_sol
                # Both pool_px and jup_px are SOL/token (same as entry_sol).
                pool_pnl = ((pool_px / entry_sol) - 1
                            if (pool_px and entry_sol > 0) else None)
                jup_pnl = ((jup_px / entry_sol) - 1
                           if (jup_px and jup_px > 0 and entry_sol > 0) else None)
                divergence = (((jup_px - pool_px) / pool_px)
                              if (pool_px and jup_px and jup_px > 0) else None)
                try:
                    self.db.record_price_probe(
                        mint_address=mint, symbol=pos.token.symbol,
                        hold_sec=hold_sec, entry_price_sol=entry_sol,
                        pool_price_sol=grpc_sol,
                        jupiter_price_usd=jup_px,
                        sol_price_usd=self._sol_price_usd,
                        divergence_pct=divergence,
                        price_source=price_source,
                        pool_pnl_pct=pool_pnl, jup_pnl_pct=jup_pnl,
                        peak_pnl_pct=pos.peak_pnl_pct,
                        trailing_activated=pos.trailing_activated)
                except Exception as e:
                    logger.debug(f"price_probe insert failed: {e}")

            # 14y shadow-exit-warn hook (rate-limited internally, OBSERVE_ONLY).
            # Always runs if enabled — independent of price-source availability.
            self._run_shadow_exit_warn_tick(pos, now)

            # If batch price unavailable, track consecutive misses.
            # DO NOT use Quote API price for PnL/trailing — it returns slippage-adjusted
            # prices that differ from mid-market by 2-5x for low-liq tokens (GOD +372% bug).
            if current_price is None:
                miss_key = pos.token.mint_address
                self._price_miss_count[miss_key] = self._price_miss_count.get(miss_key, 0) + 1
                misses = self._price_miss_count[miss_key]
                logger.warning(f"M4: batch price unavailable for {pos.token.symbol} "
                               f"(consecutive miss #{misses}, peak={pos.peak_pnl_pct*100:+.1f}%)")
                # Rug-protect: 6+ consecutive misses (~60s) while in profit.
                # At miss #5, probe Quote API to verify pool is still alive.
                # If quote succeeds → pool exists, API rate-limit is the cause → keep holding.
                # If quote fails → pool likely drained (rug) → exit.
                if pos.peak_pnl_pct >= 0.20 and misses >= 5:
                    quote_price = await self.trader.get_quote_price(
                        pos.token.mint_address, pos.token_amount * 0.01)
                    if quote_price and quote_price > 0:
                        logger.info(f"M4: {pos.token.symbol} — rug-protect probe OK "
                                    f"(quote={quote_price:.10f}, pool alive) — resetting miss counter")
                        self._price_miss_count.pop(miss_key, None)
                        continue
                    logger.warning(f"M4: {pos.token.symbol} — rug-protect exit "
                                   f"(peak={pos.peak_pnl_pct*100:+.1f}% + {misses} misses + quote probe failed)")
                    exits_to_process.append((pos, "rug_protect_blind", pos.peak_pnl_pct, None, None, None))
                    continue
                # Force sell after 30 consecutive price misses (~5min at 10s interval)
                if misses >= 30:
                    logger.warning(f"M4: {pos.token.symbol} — force exit after {misses} "
                                   f"consecutive price misses (blind position)")
                    exits_to_process.append((pos, "price_unavailable", 0.0, None, None, None))
                continue
            # Reset miss counter on successful price fetch
            self._price_miss_count.pop(pos.token.mint_address, None)
            if pos.token.mint_address not in self._first_price_seen:
                self._first_price_seen.add(pos.token.mint_address)
                self._record_latency_event(
                    pos.token, "first_price_seen",
                    hold_sec=hold_sec, entry_tx=pos.entry_tx,
                    current_price_sol=current_price)

            pnl_pct = (current_price / pos.entry_price_sol - 1) if pos.entry_price_sol > 0 else 0

            # Sanity: reject price spikes > 5x entry — likely bad data from fallback source
            if pnl_pct > 5.0:
                logger.warning(f"M4: {pos.token.symbol} — price spike rejected "
                               f"(price={current_price:.10f} vs entry={pos.entry_price_sol:.10f}, "
                               f"pnl={pnl_pct:+.0%})")
                continue

            pnl_sol = pos.sol_invested * pnl_pct

            logger.debug(f"M4: {pos.token.symbol} — price={current_price:.10f} "
                         f"entry={pos.entry_price_sol:.10f} pnl={pnl_pct:+.2%} "
                         f"peak={pos.peak_pnl_pct:+.2%} trailing={'ON' if pos.trailing_activated else 'off'} "
                         f"held={hold_sec:.0f}s")

            # Early-crash cap (Phase-1 SL from exit roadmap). If price drops
            # more than early_crash_pct within early_crash_window_sec of
            # entry, exit immediately — backtest shows -10%/120s cap cuts
            # catastrophic-loss tail across all V-shape slices.
            # Phase 25j (2026-05-04) guard: skip EC if position already
            # peaked positive — those drawdowns are recoverable volatility,
            # not rugs. See gRPC EC check above for audit details.
            if (self.config.vshape_early_crash_pct > 0
                    and hold_sec < self.config.vshape_early_crash_window_sec
                    and pnl_pct <= -self.config.vshape_early_crash_pct
                    and pos.peak_pnl_pct <= 0):
                # B6 fix (2026-05-09): require dip confirmation to defend
                # against single-tick sandwich spikes.
                # B6.1 fix (2026-05-12): bypass dip_confirmed when token is
                # clearly dead (pnl <= -50%). Audit revealed 8,194 DIP_NOT_CONFIRMED
                # log entries blocking fires on -70% to -99% tokens whose pools
                # had no swap stream activity → no confirmations possible →
                # SL/EC silently deferred → tokens held to 30min time_limit at
                # ~99% loss. 56% of 24h trades hit time_limit due to this.
                EMERGENCY_BYPASS_PNL = 0.50  # -50%: token clearly dead, fire regardless
                stream = getattr(self, "_grpc_stream", None)
                ec_confirmed = True
                if pnl_pct > -EMERGENCY_BYPASS_PNL:
                    # Normal flow: require dip confirmation
                    if stream is not None and hasattr(stream, "is_dip_confirmed"):
                        try:
                            ec_confirmed = stream.is_dip_confirmed(
                                mint, float(pos.entry_price_sol),
                                -self.config.vshape_early_crash_pct,
                                window_sec=5.0, min_confirmations=3)
                        except Exception:
                            ec_confirmed = True
                # else: pnl_pct <= -50% → emergency, skip confirmation (ec_confirmed stays True)
                if not ec_confirmed:
                    logger.info(
                        f"M4: {pos.token.symbol} — EC pnl={pnl_pct:+.2%} "
                        f"DIP_NOT_CONFIRMED (require ≥3 swaps below "
                        f"-{self.config.vshape_early_crash_pct:.0%} in 5s) — defer")
                    continue
                if pnl_pct <= -EMERGENCY_BYPASS_PNL:
                    logger.warning(
                        f"M4: {pos.token.symbol} — EC EMERGENCY pnl={pnl_pct:+.2%} "
                        f"<= -{EMERGENCY_BYPASS_PNL:.0%} — bypassing dip_confirmed")
                if self._defer_exit_for_retry(pos, "early_crash"):
                    continue
                self._record_sell_trigger_once(
                    pos, "early_crash", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                    trigger_price_sol=current_price, trigger_mid_price_sol=current_price,
                    trigger_mid_source=price_source, peak_pnl_pct=pos.peak_pnl_pct)
                logger.info(f"M4: {pos.token.symbol} — early crash triggered "
                            f"(pnl={pnl_pct:+.2%} <= -{self.config.vshape_early_crash_pct:.0%} "
                            f"within {hold_sec:.0f}s < {self.config.vshape_early_crash_window_sec}s, "
                            f"src={price_source}, dip_confirmed=true)")
                exits_to_process.append(
                    (pos, "early_crash", pnl_pct, current_price, current_price, price_source))
                continue

            # Stop loss (hard floor)
            # B6 fix (2026-05-09): require dip confirmation. Audit showed
            # 87% of SL trades recovered >30% within 10min after exit —
            # SL was firing on transient single-tick dips. Require ≥3 swaps
            # within last 5s confirming pnl <= -SL_pct before firing.
            # B6.1 fix (2026-05-12): bypass dip_confirmed when token is
            # clearly dead (pnl <= -50%). Dead pools have no swap stream →
            # no confirmations → SL silently deferred. Audit found 8,194
            # DIP_NOT_CONFIRMED entries blocking SL on -90% tokens.
            # B6.2 fix (2026-05-14): 7d audit found SL still fires 16pp late
            # on average (actual -45.8% vs spec -30%) because dead pools take
            # too long to hit the -50% emergency floor. Two changes:
            #   (a) lower emergency bypass -50% → -40% (still avoids the
            #       sandwich-tick over-fire that motivated B6)
            #   (b) time-escape: once pnl crosses -SL_pct, if dip_confirmed
            #       remains False for ≥SL_TIME_ESCAPE_SEC (10s) → fire anyway
            EMERGENCY_BYPASS_PNL = 0.40  # -40%: lowered from 0.50 per B6.2
            SL_TIME_ESCAPE_SEC = 10.0    # dead-pool fallback escape interval
            # Reset breach timer if pnl recovered above SL threshold — only
            # the most recent uninterrupted breach should count toward escape.
            if pnl_pct > -self.config.stop_loss_pct and pos.sl_first_breach_ts is not None:
                pos.sl_first_breach_ts = None
                _pos_dirty = True
            if pnl_pct <= -self.config.stop_loss_pct:
                # Record first breach timestamp for time-escape (B6.2)
                if pos.sl_first_breach_ts is None:
                    pos.sl_first_breach_ts = time.time()
                    _pos_dirty = True
                # Check for dip confirmation via raw swap stream
                stream = getattr(self, "_grpc_stream", None)
                dip_confirmed = True  # default fire (preserve old behavior on stream miss)
                time_escape_fired = False
                if pnl_pct > -EMERGENCY_BYPASS_PNL:
                    # Normal flow: require dip confirmation
                    if stream is not None and hasattr(stream, "is_dip_confirmed"):
                        try:
                            dip_confirmed = stream.is_dip_confirmed(
                                mint, float(pos.entry_price_sol),
                                -self.config.stop_loss_pct,
                                window_sec=5.0, min_confirmations=3)
                        except Exception as _e:
                            dip_confirmed = True  # safety: any error → preserve old fire
                    # B6.2: time-escape — if we've been below SL threshold
                    # for ≥10s without confirmation, fire regardless (dead pool)
                    if not dip_confirmed:
                        elapsed_below_sl = time.time() - pos.sl_first_breach_ts
                        if elapsed_below_sl >= SL_TIME_ESCAPE_SEC:
                            dip_confirmed = True
                            time_escape_fired = True
                # else: pnl_pct <= -40% → emergency, skip confirmation (dip_confirmed stays True)
                if not dip_confirmed:
                    logger.info(
                        f"M4: {pos.token.symbol} — SL pnl={pnl_pct:+.2%} "
                        f"DIP_NOT_CONFIRMED (require ≥3 swaps below "
                        f"-{self.config.stop_loss_pct:.0%} in 5s) — defer "
                        f"(time_below_sl={time.time() - pos.sl_first_breach_ts:.1f}s, "
                        f"escape in {max(0, SL_TIME_ESCAPE_SEC - (time.time() - pos.sl_first_breach_ts)):.1f}s)")
                    continue  # don't fire SL on transient dip
                if time_escape_fired:
                    logger.warning(
                        f"M4: {pos.token.symbol} — SL TIME-ESCAPE pnl={pnl_pct:+.2%} "
                        f"after {time.time() - pos.sl_first_breach_ts:.1f}s stuck below "
                        f"-{self.config.stop_loss_pct:.0%} (dead-pool fallback)")
                elif pnl_pct <= -EMERGENCY_BYPASS_PNL:
                    logger.warning(
                        f"M4: {pos.token.symbol} — SL EMERGENCY pnl={pnl_pct:+.2%} "
                        f"<= -{EMERGENCY_BYPASS_PNL:.0%} — bypassing dip_confirmed")
                if self._defer_exit_for_retry(pos, "stop_loss"):
                    continue
                self._record_sell_trigger_once(
                    pos, "stop_loss", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                    trigger_price_sol=current_price, trigger_mid_price_sol=current_price,
                    trigger_mid_source=price_source, peak_pnl_pct=pos.peak_pnl_pct)
                logger.info(f"M4: {pos.token.symbol} — stop loss triggered "
                            f"(pnl={pnl_pct:+.2%} <= -{self.config.stop_loss_pct:.0%}, "
                            f"src={price_source}, dip_confirmed=true)")
                exits_to_process.append(
                    (pos, "stop_loss", pnl_pct, current_price, current_price, price_source))
                continue

            # Hard take-profit: realized PnL crossing the TP line exits
            # immediately. This fires BEFORE trail/grace logic because a
            # parabolic pump that retraces 10% could still lose 100%+ in
            # meme-coin liquidity. take_profit_pct=0 disables this gate.
            # TP gate fix 2026-05-14: require executable peak ≥ TP × 0.5
            # before firing. 7d audit (n=16) found TP fires at avg +193%
            # with poll-peak only +1.3% — pure Jupiter Price API spikes
            # the bot can never actually execute against. Without this gate
            # every TP fire is gas-only loss. With it, only real pumps fire.
            TP_REAL_PEAK_RATIO = 0.5
            if (self.config.take_profit_pct > 0
                    and pnl_pct >= self.config.take_profit_pct
                    and price_source == "grpc_pool"):
                tp_min_real_peak = self.config.take_profit_pct * TP_REAL_PEAK_RATIO
                if pos.peak_pnl_pct_poll < tp_min_real_peak:
                    logger.info(
                        f"M4: {pos.token.symbol} — TP candidate REJECTED "
                        f"(pnl={pnl_pct:+.2%} >= {self.config.take_profit_pct:+.0%} "
                        f"but poll_peak={pos.peak_pnl_pct_poll:+.2%} < "
                        f"{tp_min_real_peak:+.0%} — likely API spike, not real)")
                    continue
                if self._defer_exit_for_retry(pos, "take_profit"):
                    continue
                self._record_sell_trigger_once(
                    pos, "take_profit", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                    trigger_price_sol=current_price, trigger_mid_price_sol=current_price,
                    trigger_mid_source=price_source, peak_pnl_pct=pos.peak_pnl_pct)
                logger.info(f"M4: {pos.token.symbol} — TAKE PROFIT triggered "
                            f"(pnl={pnl_pct:+.2%} >= {self.config.take_profit_pct:+.0%}, "
                            f"poll_peak={pos.peak_pnl_pct_poll:+.2%}, src={price_source})")
                exits_to_process.append(
                    (pos, "take_profit", pnl_pct, current_price, current_price, price_source))
                continue

            # ── v5.2 lean profit-protect soft-exit (Phase 15d, 2026-04-27) ──
            # Replaces v4_exit. v5.2 ensemble (15 features, AUC 0.91 val):
            #   p_dd_v5 = 0.5 × P(drawdown ≥5pp in next 60s) + 0.5 × (next 120s)
            # Label only fires when cur_pnl ≥ +5%, so the model already
            # encodes profit-protect. NO additional PnL/drawdown gates needed.
            # Backtest on 37 real trades (byte-aligned features post-rebuild):
            #   actual realized $-44.64 → v5.2 sim $+56.22 (+$101 / 37 trades)
            #   win rate 43% → 87%, median PnL -3.58% → +2.43%
            # Cutoff 0.30 selected by LOOCV; 80% bootstrap stable across 50
            # 75%-mints subsamples. Pre-deploy audit 22/22 PASS.
            # See outputs/v5_2_lean/v5_2_predeploy_audit.json.
            if (self.config.v5_exit_enabled
                    and price_source == "grpc_pool"
                    and hold_sec >= int(self.config.v5_exit_grace_sec)):
                v5_scores = getattr(pos, "last_v5_scores", None)
                v5_score_t = getattr(pos, "last_v5_score_t", 0.0)
                v5_age = now - float(v5_score_t) if v5_score_t else float("inf")
                if v5_scores and v5_age <= float(self.config.v5_exit_score_max_age_sec):
                    score_key = str(self.config.v5_exit_score)
                    score_val = v5_scores.get(score_key)
                    if score_val is not None and float(score_val) >= float(self.config.v5_exit_cutoff):
                        if self._defer_exit_for_retry(pos, "v5_softexit"):
                            continue
                        self._record_sell_trigger_once(
                            pos, "v5_softexit", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                            trigger_price_sol=current_price,
                            trigger_mid_price_sol=current_price,
                            trigger_mid_source=price_source,
                            peak_pnl_pct=pos.peak_pnl_pct)
                        logger.info(
                            f"M4: {pos.token.symbol} — V5.2 PROFIT-PROTECT EXIT "
                            f"({score_key}={float(score_val):.3f}>={self.config.v5_exit_cutoff:.2f}, "
                            f"pnl={pnl_pct:+.2%}, peak={pos.peak_pnl_pct:+.2%}, "
                            f"hold={hold_sec:.0f}s, age={v5_age:.0f}s)")
                        exits_to_process.append(
                            (pos, "v5_softexit", pnl_pct, current_price,
                             current_price, price_source))
                        continue

            # ── v5.3 lean profit-protect soft-exit (Phase 15e, 2026-04-29) ──
            # Trained on the historical BigWinner + V-shape entry distribution (1.57M
            # rows, 12× v5.2). val AUC 0.965/0.973 (vs v5.2 0.911/0.928).
            # 22/22 pre-deploy audit PASS. Joint sweep winner: trail_020 cutoff=0.50.
            # When v5_3_exit_shadow_only=False AND v5_exit_enabled=False:
            #   v5.3 fires real exits (replaces v5.2).
            # When shadow_only=True (default): only logs to shadow_v5_3_evals.
            #
            # 2026-05-02 FIX (Phase_24_v5_3_Production_Tuning_Investigation):
            # REMOVED `price_source == "grpc_pool"` requirement. v5.3 score is
            # computed from gRPC swap STREAM features (not fire-time price), so
            # the fire decision is robust regardless of which price feed M4 is
            # currently using. The original gate was copied from peak/trail
            # logic where price-source DOES matter (sandwich-spike risk).
            # Investigation found 1/7 v5.3 fires were missed due to this gate
            # when bot was on Jupiter fallback (65% Jupiter rate on that token).
            # Removing the gate is expected to recover ~$0.10-0.50/trade.
            if (self.config.v5_3_exit_enabled
                    and not self.config.v5_3_exit_shadow_only
                    and hold_sec >= int(self.config.v5_3_exit_grace_sec)):
                v5_3_scores = getattr(pos, "last_v5_3_scores", None)
                v5_3_score_t = getattr(pos, "last_v5_3_score_t", 0.0)
                v5_3_age = now - float(v5_3_score_t) if v5_3_score_t else float("inf")
                if v5_3_scores and v5_3_age <= float(self.config.v5_3_exit_score_max_age_sec):
                    p_v5_3 = v5_3_scores.get("p_dd_v5_3")
                    # Phase 25n (2026-05-04): peak guard — if position already up
                    # >= v5_3_exit_peak_skip_threshold (default 30%), skip v5.3
                    # fire and let trail/TP take over. v5.3 was firing
                    # prematurely on momentum-continuation tokens (TEMPO peaked
                    # 37%, X Air 27%, etc) killing further upside.
                    peak_skip_threshold = float(getattr(
                        self.config, "v5_3_exit_peak_skip_threshold", 0.30) or 0.30)
                    peak_now = float(getattr(pos, "peak_pnl_pct", 0.0) or 0.0)
                    if (p_v5_3 is not None
                            and float(p_v5_3) >= float(self.config.v5_3_exit_cutoff)
                            and peak_now < peak_skip_threshold):
                        if self._defer_exit_for_retry(pos, "v5_3_softexit"):
                            continue
                        self._record_sell_trigger_once(
                            pos, "v5_3_softexit", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                            trigger_price_sol=current_price,
                            trigger_mid_price_sol=current_price,
                            trigger_mid_source=price_source,
                            peak_pnl_pct=pos.peak_pnl_pct)
                        logger.info(
                            f"M4: {pos.token.symbol} — V5.3 PROFIT-PROTECT EXIT "
                            f"(p_dd_v5_3={float(p_v5_3):.3f}>={self.config.v5_3_exit_cutoff:.2f}, "
                            f"pnl={pnl_pct:+.2%}, peak={pos.peak_pnl_pct:+.2%}, "
                            f"hold={hold_sec:.0f}s, age={v5_3_age:.0f}s)")
                        exits_to_process.append(
                            (pos, "v5_3_softexit", pnl_pct, current_price,
                             current_price, price_source))
                        continue

            # ── v5.5.1 Chainstack-native profit-protect soft-exit (Phase 15g, 2026-05-11) ──
            # Mirrors v5.3 fire pattern. Path C 12-feature model. Bug-fix
            # retrain (BUG-1/2/3 corrected). Phase 3 audit 27/28 PASS.
            # Fire requires BOTH gates: raw_cur_pnl ≥ profit_gate (default 5%)
            # AND p_dd_v5_5 ≥ cutoff (default 0.60, prod 0.75 conservative).
            #
            # When `v5_5_exit_enabled=True` and `v5_5_exit_shadow_only=False`,
            # v5.5 fires real exits. v5.3 fire block above runs first; if v5.3
            # fired, the `continue` skips this. So v5.3 always wins ties when
            # both enabled (intentional for cutover transition).
            if (self.config.v5_5_exit_enabled
                    and not self.config.v5_5_exit_shadow_only
                    and hold_sec >= int(self.config.v5_5_exit_grace_sec)):
                v5_5_scores = getattr(pos, "last_v5_5_score", None)
                v5_5_score_t = getattr(pos, "last_v5_5_score_t", 0.0)
                v5_5_age = now - float(v5_5_score_t) if v5_5_score_t else float("inf")
                if v5_5_scores and v5_5_age <= float(self.config.v5_5_exit_score_max_age_sec):
                    p_v5_5 = v5_5_scores.get("p_dd_v5_5")
                    raw_pnl_v5_5 = v5_5_scores.get("raw_cur_pnl")
                    # Peak guard (default 1.0 = disabled; v5.5 model already
                    # has ps_peak_pnl_so_far as feature).
                    peak_skip_v55 = float(getattr(
                        self.config, "v5_5_exit_peak_skip_threshold", 1.0) or 1.0)
                    peak_now = float(getattr(pos, "peak_pnl_pct", 0.0) or 0.0)
                    profit_gate_v55 = float(self.config.v5_5_exit_profit_gate)
                    cutoff_v55 = float(self.config.v5_5_exit_cutoff)
                    if (p_v5_5 is not None
                            and raw_pnl_v5_5 is not None
                            and float(raw_pnl_v5_5) >= profit_gate_v55
                            and float(p_v5_5) >= cutoff_v55
                            and peak_now < peak_skip_v55):
                        if self._defer_exit_for_retry(pos, "v5_5_softexit"):
                            continue
                        self._record_sell_trigger_once(
                            pos, "v5_5_softexit", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                            trigger_price_sol=current_price,
                            trigger_mid_price_sol=current_price,
                            trigger_mid_source=price_source,
                            peak_pnl_pct=pos.peak_pnl_pct)
                        logger.info(
                            f"M4: {pos.token.symbol} — V5.5 PROFIT-PROTECT EXIT "
                            f"(p_dd_v5_5={float(p_v5_5):.3f}>={cutoff_v55:.2f}, "
                            f"raw_cur_pnl={float(raw_pnl_v5_5):+.2%}>={profit_gate_v55:.0%}, "
                            f"pnl={pnl_pct:+.2%}, peak={pos.peak_pnl_pct:+.2%}, "
                            f"hold={hold_sec:.0f}s, age={v5_5_age:.0f}s)")
                        exits_to_process.append(
                            (pos, "v5_5_softexit", pnl_pct, current_price,
                             current_price, price_source))
                        continue

            # ── v4.1 profit-protect soft-exit (Phase 15d, 2026-04-27 mode) ──
            # OLD MODE (rug-confirmation): fired whenever v4_score >= 0.65,
            #   typically at pnl ≈ -25% (= SL territory). Made things worse
            #   by 7pp vs no-v4 baseline because it killed bounce-back
            #   opportunities and added an extra exit-cost over what SL would
            #   have done anyway.
            # NEW MODE (profit-protect): only fire when ALL of:
            #   (1) v4_exit_enabled
            #   (2) price_source == grpc_pool (Jupiter spike protection)
            #   (3) hold_sec >= v4_exit_grace_sec (let EC handle first 30s)
            #   (4) pnl_pct >= v4_exit_min_pnl_pct (only fire when in profit)
            #   (5) v4_score >= v4_exit_cutoff (now 0.40, lower because PnL
            #       gate filters out the rug-confirmation noise that 0.65
            #       was catching)
            #   (6) drawdown_from_peak >= v4_exit_min_drawdown_pct (price
            #       has actually started reversing from the peak)
            # Backtest (sim_300, n=500): mean PnL +11.5% (vs +10.97% no-v4
            # baseline, +3.63% old-mode), v4 fires 0.8% of trades catching
            # peak exits at median +92% (winner protection, not loss cut).
            if (self.config.v4_exit_enabled
                    and price_source == "grpc_pool"
                    and hold_sec >= int(self.config.v4_exit_grace_sec)
                    and pnl_pct >= float(self.config.v4_exit_min_pnl_pct)):
                v4_scores = getattr(pos, "last_v4_scores", None)
                v4_score_t = getattr(pos, "last_v4_score_t", 0.0)
                v4_age = now - float(v4_score_t) if v4_score_t else float("inf")
                if v4_scores and v4_age <= float(self.config.v4_exit_score_max_age_sec):
                    score_key = str(self.config.v4_exit_score)
                    score_val = v4_scores.get(score_key)
                    if score_val is not None and float(score_val) >= float(self.config.v4_exit_cutoff):
                        drawdown_from_peak = float(pos.peak_pnl_pct) - float(pnl_pct)
                        if drawdown_from_peak >= float(self.config.v4_exit_min_drawdown_pct):
                            if self._defer_exit_for_retry(pos, "v4_softexit"):
                                continue
                            self._record_sell_trigger_once(
                                pos, "v4_softexit", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                                trigger_price_sol=current_price,
                                trigger_mid_price_sol=current_price,
                                trigger_mid_source=price_source,
                                peak_pnl_pct=pos.peak_pnl_pct)
                            logger.info(
                                f"M4: {pos.token.symbol} — V4 PROFIT-PROTECT EXIT "
                                f"({score_key}={float(score_val):.3f}>={self.config.v4_exit_cutoff:.2f}, "
                                f"pnl={pnl_pct:+.2%}>={self.config.v4_exit_min_pnl_pct:+.0%}, "
                                f"drawdown={drawdown_from_peak:.2%}>={self.config.v4_exit_min_drawdown_pct:.0%} "
                                f"from peak={pos.peak_pnl_pct:+.2%}, age={v4_age:.0f}s)")
                            exits_to_process.append(
                                (pos, "v4_softexit", pnl_pct, current_price,
                                 current_price, price_source))
                            continue

            if self.config.ev3m_live_enabled and self.config.ev3m_live_disable_trailing:
                continue

            if trailing_grace_active:
                logger.debug(f"M4: {pos.token.symbol} — grace period ({hold_sec:.0f}s < 15s), "
                             f"trailing logic deferred")
                continue

            # Peak only rises on pool-backed prices. Jupiter-only ticks can
            # be sandwich spikes that would inflate peak and force a delayed
            # trail sell during the dump that follows (BK postmortem).
            # Phase 16.3 Step 3 audit fix — coalesce peak + trailing-activated
            # updates into ONE save_position call. Previously two separate
            # writes could partially persist on crash.
            #
            # B5 fix (2026-05-09): peak update from current `pnl_pct` (15s
            # median-of-7 in get_swap_prices_batch) misses intra-tick spikes —
            # 5s pumps complete between polls and are diluted by median.
            # v2 sim audit showed live trail rate 4% vs sim 81% on same panel.
            # Use sandwich-filtered max from gRPC swap record buffer instead.
            _pos_dirty = False
            current_poll_peak_updated = False
            # B5 fix 2026-05-14: maintain TWO peaks:
            #   peak_pnl_pct       — observed peak (poll OR stream sandwich-max)
            #                        used only for trailing ACTIVATION
            #   peak_pnl_pct_poll  — executable peak (poll-observed only)
            #                        used for trailing drop calculation
            # Prior version overwrote peak_pnl_pct with stream sandwich-max,
            # which captures spikes the bot's price-poll never saw. Trail then
            # fired drop_from_peak against a price the bot couldn't execute
            # against — live audit (7d, n=56 trailing) showed avg 32pp late.
            if price_source == "grpc_pool" and pnl_pct > pos.peak_pnl_pct:
                pos.peak_pnl_pct = pnl_pct
                _pos_dirty = True
                current_poll_peak_updated = True
            if price_source == "grpc_pool" and pnl_pct > pos.peak_pnl_pct_poll:
                pos.peak_pnl_pct_poll = pnl_pct
                _pos_dirty = True

            # Sandwich-filtered max from raw swap stream feeds peak_pnl_pct
            # ONLY — used to activate trailing earlier on spikes the bot's
            # price feed missed. It does NOT feed peak_pnl_pct_poll, so drop
            # calculations stay anchored to executable poll-observed peaks.
            try:
                stream = getattr(self, "_grpc_stream", None)
                if stream is not None and hasattr(stream, "get_peak_price_since"):
                    peak_px = stream.get_peak_price_since(
                        mint, since_ts=pos.entry_time)
                    if peak_px is not None and pos.entry_price_sol > 0:
                        peak_pnl_from_stream = (peak_px - pos.entry_price_sol) / pos.entry_price_sol
                        if peak_pnl_from_stream > pos.peak_pnl_pct:
                            old_peak = pos.peak_pnl_pct
                            pos.peak_pnl_pct = peak_pnl_from_stream
                            _pos_dirty = True
                            if not current_poll_peak_updated:
                                logger.info(
                                    f"M4: {pos.token.symbol} — peak raised by stream "
                                    f"({old_peak:+.2%} → {peak_pnl_from_stream:+.2%}, "
                                    f"current poll pnl={pnl_pct:+.2%}, "
                                    f"poll_peak={pos.peak_pnl_pct_poll:+.2%})")
            except Exception as _b5_e:
                # Defensive — never let B5 fix break monitor loop
                logger.debug(f"M4: {pos.token.symbol} B5 peak-from-stream skipped: {_b5_e}")

            # Trailing activation: by EITHER current pnl OR (B5 fix) historical
            # peak. Without the peak-based path, B5's raised peak is silently
            # discarded for any token where current pnl is below trail_act —
            # making the B5 fix useless for spike-then-crash patterns (the
            # whole motivating case).
            if not pos.trailing_activated:
                if pnl_pct >= self.config.trailing_activation_pct:
                    pos.trailing_activated = True
                    _pos_dirty = True
                    logger.info(f"M4: {pos.token.symbol} — TRAILING STOP ACTIVATED "
                                f"(pnl={pnl_pct:+.2%} >= {self.config.trailing_activation_pct:+.0%})")
                elif pos.peak_pnl_pct >= self.config.trailing_activation_pct:
                    # B5 follow-up: peak crossed threshold (from stream), but
                    # current pnl is below. Activate trail so subsequent drop
                    # can trigger sell.
                    pos.trailing_activated = True
                    _pos_dirty = True
                    logger.info(f"M4: {pos.token.symbol} — TRAILING STOP ACTIVATED via peak "
                                f"(peak={pos.peak_pnl_pct:+.2%} >= {self.config.trailing_activation_pct:+.0%}, "
                                f"current pnl={pnl_pct:+.2%})")
            if _pos_dirty and not getattr(pos, "_removed", False):
                # BUG 3 FIX (2026-04-30): skip save if position was already
                # marked removed by _execute_sell (or _recover_phantom_sell).
                # Prevents race that re-inserts open_positions row after delete.
                self.db.save_position(pos)

            if pos.trailing_activated:
                # B5 fix 2026-05-14: drop measured against EXECUTABLE peak
                # (poll-observed), not stream sandwich-max. Live audit found
                # using stream peak made trail fire 32pp late on average
                # because the bot couldn't have sold at the stream spike.
                drop_from_peak = pos.peak_pnl_pct_poll - pnl_pct
                if drop_from_peak >= self.config.trailing_drop_pct:
                    if self._defer_exit_for_retry(pos, "trailing_stop"):
                        continue
                    self._record_sell_trigger_once(
                        pos, "trailing_stop", pnl_pct=pnl_pct, trigger_pnl_pct=pnl_pct,
                        trigger_price_sol=current_price, trigger_mid_price_sol=current_price,
                        trigger_mid_source=price_source, peak_pnl_pct=pos.peak_pnl_pct)
                    logger.info(f"M4: {pos.token.symbol} — trailing stop triggered "
                                f"(poll_peak={pos.peak_pnl_pct_poll:+.2%} "
                                f"(stream_peak={pos.peak_pnl_pct:+.2%}), now={pnl_pct:+.2%}, "
                                f"drop={drop_from_peak:.2%} >= {self.config.trailing_drop_pct:.0%}, "
                                f"src={price_source})")
                    exits_to_process.append(
                        (pos, "trailing_stop", pnl_pct, current_price, current_price, price_source))
                    continue

        # Execute exits
        for exit_item in exits_to_process:
            pos, reason, trigger_pnl, trigger_price_sol, trigger_mid_price_sol, trigger_mid_source = exit_item
            await self._execute_sell(
                pos, reason,
                trigger_pnl_pct=trigger_pnl,
                trigger_price_sol=trigger_price_sol,
                trigger_mid_price_sol=trigger_mid_price_sol,
                trigger_mid_source=trigger_mid_source)

    async def _execute_sell(self, pos: Position, reason: str, trigger_pnl_pct: float = 0.0,
                            trigger_price_sol: Optional[float] = None,
                            trigger_mid_price_sol: Optional[float] = None,
                            trigger_mid_source: Optional[str] = None):
        """M3: swap sell and record trade."""
        # Exponential backoff on repeated sell failures (30s, 60s, 120s, max 300s)
        mint = pos.token.mint_address
        retry_after = self._sell_retry_after.get(mint, 0)
        if time.time() < retry_after:
            return  # silently skip — not yet time to retry
        # Phase 16.3 Step 3 audit fix — concurrent-sell mutex.
        # rug_event_watcher and M4 monitor can both reach this path for the
        # same position. Without this guard, two Gateway swaps fire and one
        # leaves the wallet in an inconsistent state.
        if mint in self._sell_in_progress:
            logger.info(
                f"M3: {pos.token.symbol} sell already in progress "
                f"(reason={reason}) — skip duplicate")
            return
        self._sell_in_progress.add(mint)
        try:
            if trigger_price_sol is None or trigger_mid_price_sol is None:
                batch_prices = await self.trader.get_batch_prices([mint], self._sol_price_usd)
                observed_mid = batch_prices.get(mint) if batch_prices else None
                if observed_mid is not None and observed_mid > 0:
                    if trigger_price_sol is None:
                        trigger_price_sol = observed_mid
                    if trigger_mid_price_sol is None:
                        trigger_mid_price_sol = observed_mid
                    if trigger_mid_source is None:
                        trigger_mid_source = "batch_price_proxy"
            wallet_balance_ui = await self._get_wallet_token_balance_ui(mint)
            if wallet_balance_ui is not None and wallet_balance_ui <= 1e-9:
                # L2 young-position guard (postmortem 2026-04-24): a config
                # wallet_address mismatch caused 4 trades to be silently
                # pruned & lost (~$40). Never prune a recently-bought
                # position on zero-balance alone — attempt SELL regardless.
                age_sec = time.time() - pos.entry_time
                if age_sec < 300:
                    logger.error(
                        f"M3: {pos.token.symbol} wallet_balance=0 but "
                        f"age={age_sec:.0f}s < 300s — NOT pruning, attempting "
                        f"SELL anyway (possible wallet_address mismatch or "
                        f"RPC glitch — see POSTMORTEM 2026-04-24)")
                    try:
                        # 2026-04-30 fix: PRUNE_GUARD is INTENTIONAL race-protection,
                        # not an error. Demote ERROR → WARNING to reduce false alarm noise.
                        self.db.record_event(
                            "WARNING", "wallet",
                            f"PRUNE_GUARD {pos.token.symbol} "
                            f"age={age_sec:.0f}s wallet=0 — bypass prune")
                    except Exception:
                        pass
                    # fall through — do NOT return, continue with SELL attempt
                else:
                    self._prune_stale_position(
                        pos, f"{reason}_zero_wallet_balance", wallet_balance_ui)
                    return
            self._record_latency_event(
                pos.token, "sell_submit", reason=reason, trigger_pnl_pct=trigger_pnl_pct,
                trigger_price_sol=trigger_price_sol,
                trigger_mid_price_sol=trigger_mid_price_sol,
                trigger_mid_source=trigger_mid_source)
            logger.info(f"M3: [SELL] {pos.token.symbol} — reason={reason}, "
                        f"amount={pos.token_amount:.2f}, held={pos.hold_seconds():.0f}s")
            sell_slippage = 10.0 if reason in ("stop_loss", "trailing_stop") else None
            result = await self.trader.swap_sell(pos.token.mint_address, pos.token_amount,
                                                slippage_override=sell_slippage)
            logger.debug(f"M3: swap_sell result: {result}")

            # Check on-chain transaction status (status=1 means success, 0=fail)
            tx_status = result.get("status", 1)
            tx_hash = result.get("txHash", result.get("signature", "unknown"))
            if tx_status == 0:
                self._record_latency_event(
                    pos.token, "sell_onchain_fail", reason=reason, tx_hash=tx_hash)
                fails = self._sell_retry_count.get(mint, 0) + 1
                self._sell_retry_count[mint] = fails
                urgent = reason in ("stop_loss", "trailing_stop")
                backoff = min(5 * fails, 20) if urgent else min(30 * (2 ** (fails - 1)), 300)
                self._sell_retry_after[mint] = time.time() + backoff
                logger.error(f"M3: [SELL FAILED ON-CHAIN] {pos.token.symbol} — tx={tx_hash} "
                             f"status=0, retry in {backoff}s (attempt #{fails})")
                self.db.record_event("ERROR", "sell",
                                     f"SELL ON-CHAIN FAIL {pos.token.symbol} reason={reason} tx={tx_hash}")
                return

            # Parse result — Gateway may nest swap data under "data" key
            swap_data = result.get("data", result)
            sol_received = float(swap_data.get("amountOut", swap_data.get("amount", 0)))
            exit_price = sol_received / pos.token_amount if pos.token_amount > 0 else 0

            # Sanity check: if sol_received is 0 or unreasonably low, treat as failed with backoff
            if sol_received <= 0:
                self._record_latency_event(
                    pos.token, "sell_zero_output", reason=reason, tx_hash=tx_hash)
                fails = self._sell_retry_count.get(mint, 0) + 1
                self._sell_retry_count[mint] = fails
                urgent = reason in ("stop_loss", "trailing_stop")
                backoff = min(5 * fails, 20) if urgent else min(30 * (2 ** (fails - 1)), 300)
                self._sell_retry_after[mint] = time.time() + backoff
                logger.error(f"M3: [SELL SUSPICIOUS] {pos.token.symbol} — sol_received=0, "
                             f"retry in {backoff}s (attempt #{fails})")
                self.db.record_event("ERROR", "sell",
                                     f"SELL 0-RECEIVED {pos.token.symbol} reason={reason} tx={tx_hash}")
                return

            pnl_sol = sol_received - pos.sol_invested
            pnl_usd = pnl_sol * self._sol_price_usd
            pnl_pct = (pnl_sol / pos.sol_invested * 100) if pos.sol_invested > 0 else 0
            fill_vs_trigger_slippage_pct = self._safe_relative_delta(exit_price, trigger_price_sol)
            fill_vs_trigger_mid_pct = self._safe_relative_delta(exit_price, trigger_mid_price_sol)

            record = TradeRecord(
                token_symbol=pos.token.symbol,
                mint_address=pos.token.mint_address,
                entry_price_sol=pos.entry_price_sol,
                exit_price_sol=exit_price,
                token_amount=pos.token_amount,
                sol_invested=pos.sol_invested,
                sol_received=sol_received,
                pnl_sol=pnl_sol,
                pnl_usd=pnl_usd,
                hold_seconds=pos.hold_seconds(),
                exit_reason=reason,
                entry_tx=pos.entry_tx,
                exit_tx=tx_hash,
                model_score=pos.model_score,
                entry_time=pos.entry_time,
                peak_pnl_pct=pos.peak_pnl_pct,
                sol_price_usd=self._sol_price_usd,
                trigger_pnl_pct=trigger_pnl_pct,
                m2_ref_price_sol=pos.m2_ref_price_sol,
                preflight_latency_ms=pos.preflight_latency_ms,
                pool_liq_at_entry_usd=pos.pool_liq_at_entry_usd,
                pool_sol_reserve_at_entry=getattr(pos, "pool_sol_reserve_at_entry", None),
                slippage_per_pool_ratio=getattr(pos, "slippage_per_pool_ratio", None),
            )
            self._trade_log.append(record)
            if len(self._trade_log) > self._trade_log_max:
                self._trade_log = self._trade_log[-self._trade_log_max:]
            self.risk.record_trade(pnl_usd)
            self.db.record_trade(record, features=pos.features,
                                 entry_source=getattr(pos, "entry_source", ""))

            # Remove from active positions and DB
            # BUG 3 FIX (2026-04-30): _removed flag blocks concurrent save_position
            # races (M4 monitor may have stale `pos` reference and re-insert)
            pos._removed = True
            if pos in self._positions:
                self._positions.remove(pos)
            self.db.remove_position(pos.token.mint_address)
            # Unsubscribe from gRPC rug-event watcher (if subscribed).
            if self._grpc_stream is not None and pos.token.pool_address:
                self._grpc_stream.unwatch_pool(pos.token.pool_address)
            # Clear retry state and monitoring state on success
            self._sell_retry_after.pop(mint, None)
            self._sell_retry_count.pop(mint, None)
            self._sell_retry_notice_after.pop(mint, None)
            self._exit_signal_reason.pop(mint, None)
            self._price_miss_count.pop(mint, None)
            self._first_price_seen.discard(mint)

            # Final swap collector flush + unregister: trade is fully closed,
            # so we have all the swap data we need for P12-D training.
            self._cleanup_swap_keepalive(mint, reason=f"trade closed ({reason})")

            total_pnl = sum(t.pnl_usd for t in self._trade_log)
            logger.info(f"M3: [SELL OK] {pos.token.symbol} — "
                        f"pnl={pnl_sol:+.6f} SOL (${pnl_usd:+.2f}, {pnl_pct:+.1f}%) | "
                        f"reason={reason} | held={record.hold_seconds:.0f}s | "
                        f"cumulative=${total_pnl:+.2f} | tx={tx_hash}")
            self._record_latency_event(
                pos.token, "sell_confirm", reason=reason, tx_hash=tx_hash,
                trigger_pnl_pct=trigger_pnl_pct, sol_received=sol_received,
                trigger_price_sol=trigger_price_sol,
                trigger_mid_price_sol=trigger_mid_price_sol,
                trigger_mid_source=trigger_mid_source,
                fill_price_sol=exit_price,
                fill_vs_trigger_slippage_pct=fill_vs_trigger_slippage_pct,
                fill_vs_trigger_mid_pct=fill_vs_trigger_mid_pct,
                exit_reason=reason)

            self.db.record_event("INFO", "sell",
                                 f"SELL {pos.token.symbol} pnl=${pnl_usd:+.2f} reason={reason} tx={tx_hash}")

            # Blacklist token after stop_loss to prevent repeated buy-stoploss cycles
            if reason == "stop_loss":
                cooldown = self.config.token_stoploss_cooldown_sec
                self._stoploss_blacklist[pos.token.mint_address] = time.time() + cooldown
                logger.info(f"M3: {pos.token.symbol} blacklisted for {cooldown:.0f}s after stop_loss")

        except Exception as e:
            self._record_latency_event(pos.token, "sell_exception", reason=reason, error=str(e))
            fails = self._sell_retry_count.get(mint, 0) + 1
            self._sell_retry_count[mint] = fails
            urgent = reason in ("stop_loss", "trailing_stop")
            backoff = min(5 * fails, 20) if urgent else min(30 * (2 ** (fails - 1)), 300)
            self._sell_retry_after[mint] = time.time() + backoff
            err_str = str(e)

            # ========== BUG 2 FIX (2026-04-30): post-error wallet verification ==========
            # Gateway sometimes returns 400 "Transaction simulation failed" while the
            # actual on-chain swap submitted successfully (Jupiter route can change
            # between sim and execute). Without this check, bot retries indefinitely
            # despite tokens already gone. fomo case 2026-04-29 lost ~30 min of retries.
            try:
                wallet_balance = await self._get_wallet_token_balance_ui(mint)
            except Exception:
                wallet_balance = None
            if wallet_balance is not None and pos.token_amount > 0:
                drained_ratio = 1.0 - (wallet_balance / pos.token_amount)
                if drained_ratio > 0.99:  # >99% gone = swap succeeded
                    recovered = await self._recover_phantom_sell(
                        pos, reason, wallet_balance, trigger_pnl_pct,
                        trigger_price_sol, trigger_mid_price_sol, trigger_mid_source)
                    if recovered:
                        logger.warning(
                            f"M3: [SELL FAIL→RECOVERED] {pos.token.symbol} — "
                            f"Gateway error '{err_str[:80]}' but wallet drained "
                            f"({drained_ratio*100:.1f}%); trade recorded as success.")
                        self.db.record_event(
                            "WARNING", "sell",
                            f"SELL FAIL→RECOVERED {pos.token.symbol}: Gateway 400 false-error,"
                            f" wallet drained {drained_ratio*100:.1f}% — trade recorded")
                        return  # Done — position closed via recovery path
            # ============================ END BUG 2 FIX ============================

            # ========== BUG 1 FIX (2026-04-30): auto re-register on token-not-found ==========
            if "Token not found" in err_str:
                try:
                    reg_resp = await self.trader.register_token(
                        mint_address=mint,
                        symbol=pos.token.symbol,
                        name=getattr(pos.token, "name", None) or pos.token.symbol,
                        decimals=int(getattr(pos.token, "decimals", 6) or 6),
                    )
                    logger.warning(
                        f"M3: [SELL FAIL] {pos.token.symbol} — Gateway 'Token not found', "
                        f"auto-reregistered ({reg_resp.get('message', 'OK')}); "
                        f"retry in {backoff}s")
                    self._sell_retry_after[mint] = time.time() + min(backoff, 30)
                except Exception as reg_err:
                    logger.error(
                        f"M3: [SELL FAIL] {pos.token.symbol}: {e} + re-register failed: "
                        f"{reg_err} — retry in {backoff}s (attempt #{fails})")
            else:
                logger.error(f"M3: [SELL FAIL] {pos.token.symbol}: {e} — retry in {backoff}s (attempt #{fails})")
            self.db.record_event("ERROR", "sell", f"SELL FAIL {pos.token.symbol}: {e}")
        finally:
            # Phase 16.3 Step 3 audit fix — release sell-in-progress mutex
            # whether the swap succeeded, failed (with backoff), or raised.
            self._sell_in_progress.discard(mint)

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
