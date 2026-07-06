"""
MemeSniper utilities — M1 Token Discovery, M2 Signal Pipeline, M3 Gateway Trader, TradeDB.

M1 Architecture (three-tier token discovery):
  Tier 1: Chainstack on-chain polling — watch Pump.fun program for PumpSwap create_pool inner instructions
  Tier 2: GMGN token info enrichment — GET /v1/token/info for symbol, liquidity, rug_ratio, etc.
  Tier 3: GMGN kline features — existing M2 pipeline (5min kline → GBT model)

Direct HTTP calls to GMGN API and Gateway REST API.
No hummingbot connector abstraction — this is a new direct-Gateway-REST-API pattern.
"""
import asyncio
import json
import logging
import math
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from meme_sniper.L4_Signal_and_Model_Inference import ShadowGuard

try:  # production: controllers.generic.ms.models (Hummingbot loads as controllers.generic.*)
    from controllers.generic.ms.models import registry as _model_registry
except ImportError:  # in-container test channel / direct file import
    from ms.models import registry as _model_registry

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logger = logging.getLogger(__name__)

# ── Solana protocol primitives (canonical: ms.solana_protocol) ───────────────
# Production loads this module as ``controllers.generic.meme_sniper_utils``
# (sys.path root = /home/hummingbot), so the canonical package is
# ``controllers.generic.ms``. The in-container test channel does
# ``sys.path.insert(0, controllers/generic)`` and imports it flat as
# ``meme_sniper_utils``, where the package is bare ``ms``. Try the
# production-qualified path first, fall back to the flat one.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.solana_protocol import (
        PUMPFUN_PROGRAM,
        PUMPSWAP_AMM_PROGRAM,
        WSOL_MINT,
        USDC_MINT,
        USDT_MINT,
        QUOTE_TOKEN_BLACKLIST,
        PUMPSWAP_CREATE_POOL_DISC,
        PUMPSWAP_BUY_DISC,
        PUMPSWAP_SELL_DISC,
        ACCOUNT_IDX_POOL,
        ACCOUNT_IDX_CREATOR,
        ACCOUNT_IDX_BASE_MINT,
        ACCOUNT_IDX_QUOTE_MINT,
        MIN_CREATE_POOL_ACCOUNTS,
        b58decode as _b58decode,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.solana_protocol import (
        PUMPFUN_PROGRAM,
        PUMPSWAP_AMM_PROGRAM,
        WSOL_MINT,
        USDC_MINT,
        USDT_MINT,
        QUOTE_TOKEN_BLACKLIST,
        PUMPSWAP_CREATE_POOL_DISC,
        PUMPSWAP_BUY_DISC,
        PUMPSWAP_SELL_DISC,
        ACCOUNT_IDX_POOL,
        ACCOUNT_IDX_CREATOR,
        ACCOUNT_IDX_BASE_MINT,
        ACCOUNT_IDX_QUOTE_MINT,
        MIN_CREATE_POOL_ACCOUNTS,
        b58decode as _b58decode,
    )

# ── Domain types (canonical: ms.domain) ──────────────────────────────────────
# Dual-context import: production loads this module as
# ``controllers.generic.meme_sniper_utils`` (sys.path root = /home/hummingbot),
# so the canonical package is ``controllers.generic.ms``.  The in-container
# test channel inserts controllers/generic onto sys.path and imports flat as
# ``meme_sniper_utils``, where the bare ``ms`` package is visible.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.domain import (
        ChainGraduation,
        GraduatedToken,
        ObservationEntry,
        Position,
        SwapRecord,
        TradeCandidate,
        TradeRecord,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.domain import (  # type: ignore[no-redef]
        ChainGraduation,
        GraduatedToken,
        ObservationEntry,
        Position,
        SwapRecord,
        TradeCandidate,
        TradeRecord,
    )

# ── compute_swap_fields (canonical: ms.data.parse_swap) ──────────────────────
# Pure shared core extracted in Phase 3 (R8). Imported LAZILY inside
# OnChainKlineBuilder._parse_swap_from_tx (see _import_compute_swap_fields below)
# to avoid a top-level circular import: parse_swap.py imports SwapRecord from
# THIS module, so a module-level `from ...parse_swap import compute_swap_fields`
# here creates meme_sniper_utils → parse_swap → meme_sniper_utils. Under the
# production controller-loader import context that cycle raises ModuleNotFoundError
# (the `from ms.data...` fallback is unresolvable in production), which the
# `except ImportError: HAS_GEYSER = False` guard in meme_sniper.py would silently
# swallow — disabling the entire gRPC Geyser swap stream. Lazy import keeps the
# dual-context fallback while breaking the import-time cycle. Cached after first
# resolve so the per-swap hot path pays only one module-attr lookup, not a
# try/except import, on every call.
_compute_swap_fields = None  # populated on first _parse_swap_from_tx call


def _import_compute_swap_fields():
    global _compute_swap_fields
    if _compute_swap_fields is None:
        try:  # production: controllers package root on sys.path
            from controllers.generic.ms.data.parse_swap import compute_swap_fields
        except ImportError:  # test channel / flat import: controllers/generic on sys.path
            from ms.data.parse_swap import compute_swap_fields  # type: ignore[no-redef]
        _compute_swap_fields = compute_swap_fields
    return _compute_swap_fields


def _coerce_price(v) -> float:
    """Tolerant float() for GMGN price fields.

    GMGN /v1/token/info changed `price` from a flat str/number to a nested
    dict {"address": ..., "price": "0.000053", "price_1m": ..., ...}
    (observed 2026-05-14). When v is a dict, extract the inner "price";
    otherwise coerce as usual. Returns 0.0 on any failure.
    """
    if isinstance(v, dict):
        v = v.get("price") or v.get("usd_price") or 0
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


# ──────────────────────────────────────────────
# Tier 1: Solana RPC client + graduation parser
# ──────────────────────────────────────────────

# ChainGraduation moved to ms.domain.types (Phase 1 leaf #2).
# Re-imported above via try/except dual-context block.

# SolanaRPC moved to ms.transport.solana_rpc (Phase 1 leaf #3).
# Dual-context re-import preserves ``from meme_sniper_utils import SolanaRPC``.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.transport.solana_rpc import SolanaRPC
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.transport.solana_rpc import SolanaRPC  # type: ignore[no-redef]

# ── Feature-compute functions (canonical: ms.features) ───────────────────────
# Pure math helpers + entry feature builders moved to ms.features (Phase 1 leaf #4).
# Dual-context re-import preserves ``from meme_sniper_utils import compute_micro_live``
# etc. for all callers (meme_sniper.py, notebooks, tests).
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.features import (
        _body_frac_value,
        _build_confirmed_entry_frame,
        _close_anchor_value,
        _family_pass_threshold,
        _get_post_grad_anchor_open_live,
        _max_drawdown_from_path,
        _range_pct_value,
        _safe_float_value,
        _safe_range_pos_value,
        _safe_ratio_value,
        _safe_return_value,
        _series_slope,
        _slice_sum_value,
        _tail_mean_value,
        _tail_share_positive_value,
        _upper_wick_frac_value,
        compute_confirmed_entry_live_features,
        compute_micro_10m_live,
        compute_micro_live,
        compute_micro_live_full,
        compute_super_winner_event_core_features,
        compute_super_winner_event_micro_overlay_features,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.features import (  # type: ignore[no-redef]
        _body_frac_value,
        _build_confirmed_entry_frame,
        _close_anchor_value,
        _family_pass_threshold,
        _get_post_grad_anchor_open_live,
        _max_drawdown_from_path,
        _range_pct_value,
        _safe_float_value,
        _safe_range_pos_value,
        _safe_ratio_value,
        _safe_return_value,
        _series_slope,
        _slice_sum_value,
        _tail_mean_value,
        _tail_share_positive_value,
        _upper_wick_frac_value,
        compute_confirmed_entry_live_features,
        compute_micro_10m_live,
        compute_micro_live,
        compute_micro_live_full,
        compute_super_winner_event_core_features,
        compute_super_winner_event_micro_overlay_features,
    )

# ── Telemetry sinks (canonical: ms.telemetry.telemetry_sink) ─────────────────
# Production loads this module as ``controllers.generic.meme_sniper_utils``
# (sys.path root = /home/hummingbot), so the canonical package is
# ``controllers.generic.ms``. The in-container test channel does
# ``sys.path.insert(0, controllers/generic)`` and imports flat as
# ``meme_sniper_utils``, where the package is bare ``ms``. Try the
# production-qualified path first, fall back to the flat one.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.telemetry.telemetry_sink import (
        ITelemetrySink,
        NullSink,
        PostgresTelemetrySink,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.telemetry.telemetry_sink import (  # type: ignore[no-redef]
        ITelemetrySink,
        NullSink,
        PostgresTelemetrySink,
    )

# ── GatewayTrader (canonical: ms.execution.gateway_trader) ───────────────────
# M3 Gateway REST API client — token registration + Jupiter swap execution.
# _RESERVED_SYMBOLS frozenset is defined inside the class (verbatim) to prevent
# meme token symbol collision with SOL/USDC routing tokens (3 outages history).
# Production loads this module as ``controllers.generic.meme_sniper_utils``
# (sys.path root = /home/hummingbot), so the canonical package is
# ``controllers.generic.ms``. The in-container test channel inserts
# controllers/generic onto sys.path, where the bare ``ms`` package is visible.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.execution.gateway_trader import GatewayTrader  # noqa: F401
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.execution.gateway_trader import GatewayTrader  # type: ignore[no-redef]  # noqa: F401

# ── RiskManager (canonical: ms.risk.risk_manager) ────────────────────────────
# M5 risk manager — daily P&L, consecutive losses, trade count, halt logic.
# State rebuilt from TradeDB on startup; auto-resets at UTC midnight.
# Production loads this module as ``controllers.generic.meme_sniper_utils``
# (sys.path root = /home/hummingbot), so the canonical package is
# ``controllers.generic.ms``. The in-container test channel inserts
# controllers/generic onto sys.path, where the bare ``ms`` package is visible.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.risk.risk_manager import RiskManager  # noqa: F401
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.risk.risk_manager import RiskManager  # type: ignore[no-redef]  # noqa: F401


def parse_graduation_from_tx(tx_result: dict, signature: str) -> Optional[ChainGraduation]:
    """Parse a Pump.fun transaction and detect graduation via PumpSwap create_pool.

    Uses the Anchor instruction discriminator (first 8 bytes of instruction data)
    to precisely identify create_pool instructions. This replaces the old heuristic
    of "PumpSwap inner ix with >= 15 accounts" which had ~10% false positive rate
    on routed swap_buy/swap_sell instructions.

    Additional guards (QUOTE_TOKEN_BLACKLIST, quote_mint == WSOL) are retained
    as defense-in-depth.
    """
    if not tx_result:
        return None

    block_time = tx_result.get("blockTime", 0)
    slot = tx_result.get("slot", 0)
    msg = tx_result.get("transaction", {}).get("message", {})
    meta = tx_result.get("meta", {})

    # Build full account keys list (static + address table lookups)
    account_keys = []
    for ak in msg.get("accountKeys", []):
        if isinstance(ak, dict):
            account_keys.append(ak.get("pubkey", ""))
        else:
            account_keys.append(ak)
    for loaded in (meta.get("loadedAddresses") or {}).get("writable", []):
        account_keys.append(loaded)
    for loaded in (meta.get("loadedAddresses") or {}).get("readonly", []):
        account_keys.append(loaded)

    # Scan inner instructions for PumpSwap create_pool (discriminator-based)
    for inner_group in meta.get("innerInstructions", []):
        for inner_ix in inner_group.get("instructions", []):
            if inner_ix.get("programId", "") != PUMPSWAP_AMM_PROGRAM:
                continue

            # ── Primary check: instruction discriminator ──
            # PumpSwap is an Anchor program. The first 8 bytes of the
            # instruction data encode the instruction type. Only create_pool
            # is a graduation; buy/sell/deposit/withdraw are not.
            ix_data_b58 = inner_ix.get("data", "")
            if ix_data_b58:
                try:
                    ix_data = _b58decode(ix_data_b58)
                    if ix_data[:8] != PUMPSWAP_CREATE_POOL_DISC:
                        continue  # not create_pool — skip (buy/sell/etc)
                except Exception:
                    pass  # fall through to account-count heuristic

            # ── Fallback: account count heuristic (if data field missing) ──
            ix_accounts = inner_ix.get("accounts", [])
            if not ix_data_b58 and len(ix_accounts) < MIN_CREATE_POOL_ACCOUNTS:
                continue

            def resolve(idx_or_key) -> str:
                if isinstance(idx_or_key, int):
                    return account_keys[idx_or_key] if idx_or_key < len(account_keys) else ""
                return idx_or_key

            if len(ix_accounts) < 5:
                continue

            base_mint = resolve(ix_accounts[ACCOUNT_IDX_BASE_MINT])
            quote_mint = resolve(ix_accounts[ACCOUNT_IDX_QUOTE_MINT])
            pool = resolve(ix_accounts[ACCOUNT_IDX_POOL])
            creator = resolve(ix_accounts[ACCOUNT_IDX_CREATOR])

            if not base_mint:
                continue

            # Guard 1: base_mint can never be a known quote token.
            if base_mint in QUOTE_TOKEN_BLACKLIST:
                logger.debug(
                    "parse_graduation_from_tx: rejected base_mint=%s in "
                    "QUOTE_TOKEN_BLACKLIST (sig=%s)",
                    base_mint, signature[:16],
                )
                continue

            # Guard 2: real PumpSwap graduations always pair against WSOL.
            # If quote_mint is anything else, this isn't a fresh meme pool.
            if quote_mint != WSOL_MINT:
                logger.debug(
                    "parse_graduation_from_tx: rejected quote_mint=%s "
                    "(expected WSOL, sig=%s)",
                    quote_mint, signature[:16],
                )
                continue

            return ChainGraduation(
                mint=base_mint, pool=pool, creator=creator,
                quote_mint=quote_mint, signature=signature,
                block_time=block_time, slot=slot,
            )

    return None


# ──────────────────────────────────────────────
# Data classes — moved to ms.domain.types (Phase 1 leaf #2)
# ──────────────────────────────────────────────
# GraduatedToken, TradeCandidate, ObservationEntry, Position, TradeRecord,
# SwapRecord re-imported above via try/except dual-context block.

SNAPSHOT_SCHEDULE = [
    # t0: "birth certificate" snapshot ~30s after graduation.
    # GMGN needs a few seconds to index a newly graduated token; 30s gives
    # enough buffer for the API to have basic data (dev, holders, security)
    # while still capturing the token's state before any real trading.
    (30,      "gmgn_info_t0"),
    (15 * 60, "gmgn_info_15m"),
    (30 * 60, "gmgn_info_30m"),
    (45 * 60, "gmgn_info_45m"),
]


# ObservationEntry, Position, TradeRecord, SwapRecord moved to ms.domain.types
# (Phase 1 leaf #2). Re-imported above via try/except dual-context block.


# ──────────────────────────────────────────────
# On-chain Kline Builder
#   Build 1min candles from PumpSwap pool swap transactions.
#   Replaces GMGN kline dependency for newly graduated tokens.
# ──────────────────────────────────────────────


class OnChainKlineBuilder:
    """Build kline candles from on-chain swap data for newly graduated tokens.

    Supports multi-pool monitoring: a single token can have pools on PumpSwap,
    Meteora, Raydium, etc. Swaps from ALL registered pools are merged into
    unified candles, matching how Dune dex_solana.trades aggregates cross-DEX.

    After M1 discovers a graduation (with pool_address from Chainstack), this class:
    1. Registers the token's pool(s) for monitoring
    2. Periodically polls each pool's transactions via getSignaturesForAddress
    3. Parses each swap to extract price, volume, direction
    4. Builds 1-minute candles from aggregated swap data across all pools

    Key advantage over GMGN kline: available immediately (no indexing delay),
    and provides REAL trade_count + buy_sell_ratio (not scaler mean fill).
    """

    WSOL_MINT = "So11111111111111111111111111111111111111112"
    BATCH_SIZE = 20  # JSON-RPC batch size for getTransaction calls

    def __init__(self, rpc_url: str, tx_concurrency: int = 3):
        self._rpc_url = rpc_url
        self._tx_concurrency = tx_concurrency
        self._client: Optional[httpx.AsyncClient] = None
        self._pools: Dict[str, List[str]] = {}      # mint -> [pool_address, ...]
        self._swaps: Dict[str, List[SwapRecord]] = {}  # mint -> [SwapRecord, ...]
        self._seen_sigs: Dict[str, set] = {}        # mint -> set of processed tx sigs (across all pools)
        # Phase O3 (2026-05-23): reverse index pool_address → mint for O(1)
        # lookup in geyser_stream._find_registered_pools. Maintained in sync
        # with _pools by register/unregister. Per P0-B test invariant: this
        # dict has no lock — MUST be accessed from main asyncio event loop.
        self._pool_to_mint_idx: Dict[str, str] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def register(self, mint: str, pool_address: str):
        """Start tracking swaps for a token's pool. Can be called multiple times
        to add additional pools (e.g. PumpSwap first, then Meteora)."""
        if mint not in self._pools:
            self._pools[mint] = []
            self._swaps[mint] = []
            self._seen_sigs[mint] = set()
        if pool_address in self._pools[mint]:
            return
        self._pools[mint].append(pool_address)
        # Phase O3: maintain reverse index in sync with _pools
        self._pool_to_mint_idx[pool_address] = mint
        n = len(self._pools[mint])
        logger.info(f"KlineBuilder: registered {mint[:12]}... pool={pool_address[:12]}... (pool #{n})")

    def unregister(self, mint: str):
        """Stop tracking a token and free memory."""
        # Phase O3: remove all pool entries from reverse index BEFORE popping _pools
        pools_to_remove = self._pools.get(mint, [])
        for p in pools_to_remove:
            self._pool_to_mint_idx.pop(p, None)
        self._pools.pop(mint, None)
        self._swaps.pop(mint, None)
        self._seen_sigs.pop(mint, None)

    def find_mint_by_pool(self, pool: str) -> Optional[str]:
        """Phase O3 (2026-05-23): O(1) reverse lookup pool → mint.

        Thread safety: `dict.get` is atomic per Python GIL, safe from any
        thread. The `_pool_to_mint_idx` dict is mutated only by
        `register()`/`unregister()` which are called exclusively from the
        main asyncio event loop (verified by code review).

        Phase O3 v0.2 (audit hotfix): originally asserted `current_thread is
        main_thread`, but that would crash the bot if any unexpected thread
        pool callback (e.g., solders signing) ever called this. Replaced
        with soft warning log + always return the lookup result. Test
        `test_find_mint_by_pool_rejects_non_main_thread_AFTER_O3` is
        retained as DESIGN INTENT but won't fire in production.

        Returns the mint string, or None if pool not registered.
        """
        import threading
        if threading.current_thread() is not threading.main_thread():
            # Defensive log; should never happen given current call sites
            # but don't crash the bot if it does.
            logger.warning(
                "O3 invariant relaxed: find_mint_by_pool called from non-main "
                f"thread {threading.current_thread().name}; continuing (atomic dict.get)"
            )
        return self._pool_to_mint_idx.get(pool)

    def is_registered(self, mint: str) -> bool:
        return mint in self._pools

    def get_swap_count(self, mint: str) -> int:
        return len(self._swaps.get(mint, []))

    def get_swaps(self, mint: str) -> List[SwapRecord]:
        """Return all collected swaps for a token (for DB persistence)."""
        return list(self._swaps.get(mint, []))

    def get_pool_address(self, mint: str) -> Optional[str]:
        """Return the first (primary) pool address for a token."""
        pools = self._pools.get(mint, [])
        return pools[0] if pools else None

    def get_pool_count(self, mint: str) -> int:
        """Return number of registered pools for a token."""
        return len(self._pools.get(mint, []))

    def registered_mints(self) -> List[str]:
        """Return a snapshot list of all currently registered mint addresses."""
        return list(self._pools.keys())

    def pools_by_mint(self) -> Dict[str, List[str]]:
        """Return a shallow-copy snapshot of the mint→pools mapping."""
        return dict(self._pools)

    def has_swaps(self, mint: str) -> bool:
        """Return True if mint has an entry in the swap buffer (may differ from is_registered)."""
        return mint in self._swaps

    def inject_swap(self, mint: str, swap: "SwapRecord", signature: str = ""):
        """Push-based swap injection from gRPC Geyser stream.

        Safe to call from the same asyncio event loop (no threading concerns).
        Silently ignores unregistered mints.
        """
        if mint not in self._swaps:
            return
        # Phase 22.D Path B: persist signature on the SwapRecord so it lands
        # in the swaps.tx_hash column when save_swaps fires.
        if signature and not getattr(swap, "signature", ""):
            swap.signature = signature
        self._swaps[mint].append(swap)
        if signature and mint in self._seen_sigs:
            self._seen_sigs[mint].add(signature)

    async def poll_swaps(self, mint: str) -> int:
        """Fetch and parse new swap transactions across ALL pools for a token.

        Returns count of newly parsed swaps (combined from all pools).
        """
        pools = self._pools.get(mint, [])
        if not pools:
            return 0

        total_new = 0
        for pool in pools:
            count = await self._poll_single_pool(mint, pool)
            total_new += count

        return total_new

    async def backfill_swaps(self, mint: str, target_span_sec: int = 300,
                              max_pages: int = 10) -> int:
        """Backfill swap history by paging backwards through signatures.

        For trending tokens, a single getSignaturesForAddress(limit=200) may only
        cover ~45s on high-activity pools. This method pages backwards using the
        `before` cursor until swap data spans >= target_span_sec (default 5min).

        Returns total number of new swaps collected.
        """
        pools = self._pools.get(mint, [])
        if not pools:
            return 0

        total_new = 0
        for pool in pools:
            total_new += await self._backfill_single_pool(
                mint, pool, target_span_sec, max_pages)

        return total_new

    async def _backfill_single_pool(self, mint: str, pool: str,
                                     target_span_sec: int, max_pages: int) -> int:
        """Page backwards through a pool's signatures until we cover target_span_sec."""
        client = await self._get_client()
        seen = self._seen_sigs.get(mint, set())
        all_new_swaps = []
        cursor: Optional[str] = None  # `before` cursor for pagination

        for page in range(max_pages):
            # Phase 22.D Route X RC4 (2026-05-01): bumped limit 200 → 1000.
            # Solana RPC supports up to 1000 sigs per call; previous 200 cap
            # forced 5× more pages on hot tokens (3000 swaps in 30min) and
            # the 15-page hard ceiling lost the oldest swaps when sustained
            # rate exceeded 6.7/s. 1000 limit covers the median 30min token
            # in 4 pages instead of 15+.
            params: dict = {"limit": 1000}
            if cursor:
                params["before"] = cursor

            body = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [pool, params],
            }
            try:
                resp = await client.post(self._rpc_url, json=body)
                if resp.status_code == 429:
                    logger.debug(f"KlineBuilder: backfill 429 on page={page}, retrying in 2s")
                    await asyncio.sleep(2)
                    resp = await client.post(self._rpc_url, json=body)
                resp.raise_for_status()
                sigs_result = resp.json().get("result", [])
            except Exception as e:
                logger.debug(f"KlineBuilder: backfill getSignatures failed page={page}: {e}")
                break

            if not sigs_result:
                break

            # Update cursor to oldest signature in this batch (for next page)
            cursor = sigs_result[-1]["signature"]

            # Filter to new, successful signatures
            new_sigs = []
            for si in sigs_result:
                sig = si["signature"]
                if sig not in seen and si.get("err") is None:
                    new_sigs.append(sig)
                    seen.add(sig)

            if new_sigs:
                tx_results = await self._fetch_transactions_batch(new_sigs)
                for sig, tx_data in zip(new_sigs, tx_results):
                    if tx_data is None:
                        continue
                    swap = self._parse_swap_from_tx(tx_data, mint)
                    if swap:
                        # Phase 22.D Path B: attach signature for tx_hash column
                        swap.signature = sig
                        all_new_swaps.append(swap)

            # Check if we've covered enough time span
            existing = self._swaps.get(mint, [])
            all_ts = [s.timestamp for s in existing] + [s.timestamp for s in all_new_swaps]
            if all_ts:
                span = max(all_ts) - min(all_ts)
                logger.debug(f"KlineBuilder: backfill {mint[:12]}... page={page+1}, "
                             f"+{len(new_sigs)} sigs, span={span:.0f}s/{target_span_sec}s")
                if span >= target_span_sec:
                    break

        # Sort and append
        if all_new_swaps:
            all_new_swaps.sort(key=lambda s: s.timestamp)
            self._swaps[mint].extend(all_new_swaps)
            self._seen_sigs[mint] = seen

        return len(all_new_swaps)

    async def _poll_single_pool(self, mint: str, pool: str) -> int:
        """Fetch and parse new swap transactions for a single pool."""
        client = await self._get_client()

        # Step 1: Get recent signatures for the pool.
        # Phase 22.D Route X RC4 (2026-05-01): limit 200 → 1000 (see backfill).
        body = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignaturesForAddress",
            "params": [pool, {"limit": 1000}],
        }
        try:
            resp = await client.post(self._rpc_url, json=body)
            resp.raise_for_status()
            sigs_result = resp.json().get("result", [])
        except Exception as e:
            logger.debug(f"KlineBuilder: getSignatures failed for {mint[:12]}... pool={pool[:12]}...: {e}")
            return 0

        # Filter to new, successful signatures (shared seen set across all pools)
        seen = self._seen_sigs.get(mint, set())
        new_sigs = []
        for si in sigs_result:
            sig = si["signature"]
            if sig not in seen and si.get("err") is None:
                new_sigs.append(sig)
                seen.add(sig)  # mark seen immediately to avoid re-fetching on error

        if not new_sigs:
            return 0

        # Step 2: Batch-fetch transactions via JSON-RPC batching
        all_tx_results = await self._fetch_transactions_batch(new_sigs)

        # Step 3: Parse swaps from transaction results
        new_swaps = []
        for sig, tx_data in zip(new_sigs, all_tx_results):
            if tx_data is None:
                continue
            swap = self._parse_swap_from_tx(tx_data, mint)
            if swap:
                # Phase 22.D Path B: attach signature for tx_hash column
                swap.signature = sig
                new_swaps.append(swap)

        # Sort by timestamp and append
        new_swaps.sort(key=lambda s: s.timestamp)
        self._swaps[mint].extend(new_swaps)
        self._seen_sigs[mint] = seen

        return len(new_swaps)

    async def _fetch_transactions_batch(self, sigs: List[str]) -> List[Optional[dict]]:
        """Fetch multiple transactions using JSON-RPC batching for efficiency."""
        client = await self._get_client()
        results: List[Optional[dict]] = [None] * len(sigs)

        for chunk_start in range(0, len(sigs), self.BATCH_SIZE):
            chunk = sigs[chunk_start:chunk_start + self.BATCH_SIZE]
            batch = [
                {
                    "jsonrpc": "2.0", "id": i,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "jsonParsed",
                                     "maxSupportedTransactionVersion": 0}],
                }
                for i, sig in enumerate(chunk)
            ]
            try:
                resp = await client.post(self._rpc_url, json=batch, timeout=30.0)
                resp.raise_for_status()
                batch_results = resp.json()
                if isinstance(batch_results, list):
                    for r in batch_results:
                        idx = r.get("id")
                        if idx is not None and 0 <= idx < len(chunk):
                            results[chunk_start + idx] = r.get("result")
            except Exception as e:
                logger.debug(f"KlineBuilder: batch tx fetch failed: {e}")

        return results

    def _parse_swap_from_tx(self, tx_data: dict, base_mint: str) -> Optional[SwapRecord]:
        """Parse a single transaction to extract swap price/volume/direction.

        Works across all Solana AMMs (PumpSwap, Meteora DAMM, Raydium, Orca):
        1. Base token deltas from preTokenBalances/postTokenBalances
        2. SOL amount from WSOL token deltas OR native SOL lamport changes
        3. Direction from the transaction signer's base token delta
        """
        meta = tx_data.get("meta", {})
        if meta.get("err") is not None:
            return None

        block_time = tx_data.get("blockTime", 0)
        if block_time == 0:
            return None

        # --- Step 1: Build token balance deltas ---
        pre_map: Dict[int, dict] = {}
        post_map: Dict[int, dict] = {}

        for b in meta.get("preTokenBalances", []):
            idx = b.get("accountIndex")
            ui = b.get("uiTokenAmount", {})
            amount = ui.get("uiAmount")
            if amount is None:
                raw = int(ui.get("amount", "0"))
                decimals = ui.get("decimals", 0)
                amount = raw / (10 ** decimals) if decimals > 0 else float(raw)
            else:
                amount = float(amount)
            pre_map[idx] = {"mint": b.get("mint", ""), "owner": b.get("owner", ""),
                            "amount": amount}

        for b in meta.get("postTokenBalances", []):
            idx = b.get("accountIndex")
            ui = b.get("uiTokenAmount", {})
            amount = ui.get("uiAmount")
            if amount is None:
                raw = int(ui.get("amount", "0"))
                decimals = ui.get("decimals", 0)
                amount = raw / (10 ** decimals) if decimals > 0 else float(raw)
            else:
                amount = float(amount)
            post_map[idx] = {"mint": b.get("mint", ""), "owner": b.get("owner", ""),
                             "amount": amount}

        # Compute deltas grouped by owner
        all_indices = set(list(pre_map.keys()) + list(post_map.keys()))
        base_deltas: Dict[str, float] = {}  # owner -> delta
        wsol_deltas: Dict[str, float] = {}

        for idx in all_indices:
            pre = pre_map.get(idx, {})
            post = post_map.get(idx, {})
            mint_addr = post.get("mint") or pre.get("mint", "")
            owner = post.get("owner") or pre.get("owner", "")
            pre_amt = pre.get("amount", 0.0)
            post_amt = post.get("amount", 0.0)
            delta = post_amt - pre_amt

            if abs(delta) < 1e-12:
                continue

            if mint_addr == base_mint:
                base_deltas[owner] = base_deltas.get(owner, 0) + delta
            elif mint_addr == self.WSOL_MINT:
                wsol_deltas[owner] = wsol_deltas.get(owner, 0) + delta

        if not base_deltas:
            return None  # No base token movement — not a swap

        # --- Step 2-3 (format-specific): WSOL delta aggregation + native lamport ---
        # Compute from BOTH sources and take the larger value:
        # - WSOL token deltas (reliable for PumpSwap)
        # - Native SOL lamport changes (reliable for Meteora DAMM v2)
        # Meteora routes user SOL via native lamports; the WSOL token delta is just the fee.
        wsol_positive = sum(d for d in wsol_deltas.values() if d > 0)
        wsol_negative = sum(abs(d) for d in wsol_deltas.values() if d < 0)
        wsol_amount = max(wsol_positive, wsol_negative)

        # Native SOL lamport changes (user's SOL payment / receipt)
        native_sol_amount = 0.0
        pre_bal = meta.get("preBalances", [])
        post_bal = meta.get("postBalances", [])
        if pre_bal and post_bal:
            sol_received = 0.0
            sol_spent = 0.0
            for i in range(min(len(pre_bal), len(post_bal))):
                delta_lamports = post_bal[i] - pre_bal[i]
                if delta_lamports > 100_000:  # > 0.0001 SOL (ignore dust/fees)
                    sol_received += delta_lamports
                elif delta_lamports < -100_000:
                    sol_spent += abs(delta_lamports)
            native_sol_amount = max(sol_received, sol_spent) / 1e9  # lamports → SOL

        # Signer (format-specific: RPC dict path)
        signer = self._get_tx_signer(tx_data)

        # ── Common core ────────────────────────────────────────────────────────
        # Steps 2-5 (base_amount, sol_amount, is_buy, price_sol, trader) are
        # identical between gRPC and RPC parsers and live in compute_swap_fields.
        # Phase 3 (R8): dedup without altering any behaviour. Lazy import
        # (see _import_compute_swap_fields) breaks the parse_swap↔utils cycle.
        compute_swap_fields = _import_compute_swap_fields()
        fields = compute_swap_fields(
            base_deltas=base_deltas,
            wsol_amount=wsol_amount,
            native_sol_amount=native_sol_amount,
            base_mint=base_mint,
            signer=signer,
        )
        if fields is None:
            return None

        base_amount, sol_amount, is_buy, price_sol, trader = fields

        return SwapRecord(
            timestamp=block_time,
            price_sol=price_sol,
            volume_sol=sol_amount,
            is_buy=is_buy,
            base_amount=base_amount,
            trader_address=trader,
            # Additive (2026-06-28): tx slot for the pre-grad-factor shadow eval
            # (first-slot key). Default 0 when absent; every other consumer
            # ignores it, so behaviour for the live kline path is unchanged.
            slot=int(tx_data.get("slot", 0) or 0),
        )

    @staticmethod
    def _get_tx_signer(tx_data: dict) -> Optional[str]:
        """Extract the first signer (transaction initiator) from a parsed transaction."""
        msg = tx_data.get("transaction", {}).get("message", {})
        account_keys = msg.get("accountKeys", [])
        for ak in account_keys:
            if isinstance(ak, dict):
                if ak.get("signer"):
                    return ak.get("pubkey", "")
            elif isinstance(ak, str):
                # In older formats, first key is the fee payer / signer
                return ak
        return None

    # Dust filter threshold: swaps below this SOL volume are discarded
    DUST_THRESHOLD_SOL = 0.01
    # Log-space outlier threshold: ln(5) ≈ 1.609 → prices >5x from weighted median
    OUTLIER_LN_THRESHOLD = 1.609
    # VWAP close window: last N seconds of a bar for volume-weighted close
    VWAP_CLOSE_WINDOW_SEC = 10

    def build_kline(self, mint: str, start_ts: int, n_bars: int = 5,
                    resolution: int = 60, sol_price_usd: float = 1.0
                    ) -> Tuple[Optional[List[Dict]], bool]:
        """Build kline bars from collected swap data with multi-layer filtering.

        Filtering pipeline per bar:
          Layer 1 — Dust filter: remove volume_sol < DUST_THRESHOLD_SOL
          Layer 2 — Log-space sqrt-weighted median outlier filter
          Layer 3 — VWAP close (last 10s volume-weighted average)

        Returns:
            (bars, shifted) tuple.
            bars: List of kline bar dicts, or None if insufficient data.
            shifted: True if start_ts was auto-shifted away from original window.
        """
        swaps = self._swaps.get(mint, [])
        if not swaps:
            return None, False

        ts_list = sorted(s.timestamp for s in swaps)
        span_sec = ts_list[-1] - ts_list[0] if len(ts_list) > 1 else 0
        logger.debug(f"KlineBuilder: {mint[:12]}... {len(swaps)} swaps, "
                     f"span={span_sec:.0f}s, range=[{ts_list[0]}-{ts_list[-1]}], "
                     f"start_ts={start_ts}")

        shifted = False
        window_end = start_ts + n_bars * resolution
        in_window = any(start_ts <= s.timestamp < window_end for s in swaps)

        if not in_window:
            earliest = min(s.timestamp for s in swaps)
            new_start = earliest - (earliest % resolution)
            logger.debug(f"KlineBuilder: {mint[:12]}... shifting kline start "
                         f"{start_ts} → {new_start} (swap data not in original window)")
            start_ts = new_start
            shifted = True

        bars = []
        filter_stats = []  # per-bar stats for observability
        for i in range(n_bars):
            bar_start = start_ts + i * resolution
            bar_end = bar_start + resolution

            bar_swaps = [s for s in swaps if bar_start <= s.timestamp < bar_end]

            if not bar_swaps:
                if bars:
                    last_close = bars[-1]["close"]
                    bars.append({
                        "time": bar_start * 1000,
                        "open": last_close, "close": last_close,
                        "high": last_close, "low": last_close,
                        "volume": 0.0, "trades": 0, "buys": 0, "sells": 0,
                    })
                    filter_stats.append("0/0/0")
                else:
                    continue
            else:
                raw_count = len(bar_swaps)

                # Layer 1: Dust filter
                after_dust = [s for s in bar_swaps
                              if s.volume_sol >= self.DUST_THRESHOLD_SOL]
                if not after_dust:
                    after_dust = bar_swaps

                # Layer 2: Log-space sqrt-weighted median outlier filter
                filtered = self._filter_outliers_log_median(after_dust)

                dust_count = len(after_dust)
                final_count = len(filtered)
                prices = [s.price_sol * sol_price_usd for s in filtered]

                # Layer 3: VWAP close (last 10s of bar)
                close_price, used_vwap = self._compute_vwap_close(
                    filtered, bar_end, sol_price_usd)

                bars.append({
                    "time": bar_start * 1000,
                    "open": prices[0],
                    "close": close_price,
                    "high": max(prices),
                    "low": min(prices),
                    "volume": sum(s.volume_sol * sol_price_usd for s in filtered),
                    "trades": final_count,
                    "buys": sum(1 for s in filtered if s.is_buy),
                    "sells": sum(1 for s in filtered if not s.is_buy),
                })
                vwap_flag = "V" if used_vwap else "L"
                filter_stats.append(f"{raw_count}/{dust_count}/{final_count}{vwap_flag}")

        if len(bars) < n_bars:
            return None, shifted

        logger.info(f"KlineBuilder: {mint[:12]}... {n_bars} bars built, "
                    f"shifted={shifted}, "
                    f"per_bar(raw/dust/final)=[{', '.join(filter_stats)}]")

        return bars, shifted

    @staticmethod
    def _filter_outliers_log_median(swaps: List[SwapRecord]) -> List[SwapRecord]:
        """Log-space sqrt-weighted median outlier filter."""
        if len(swaps) <= 2:
            return swaps

        # Compute sqrt-volume-weighted median in log-price space
        log_prices = []
        weights = []
        for s in swaps:
            if s.price_sol > 0:
                log_prices.append(math.log(s.price_sol))
                weights.append(math.sqrt(max(s.volume_sol, 1e-9)))

        if not log_prices:
            return swaps

        # Sort by log-price, find weighted median
        paired = sorted(zip(log_prices, weights))
        total_w = sum(w for _, w in paired)
        cum = 0.0
        wmedian_log = paired[0][0]
        for lp, w in paired:
            cum += w
            if cum >= total_w * 0.5:
                wmedian_log = lp
                break

        # Filter: keep swaps within ln(5) of weighted median
        threshold = OnChainKlineBuilder.OUTLIER_LN_THRESHOLD
        filtered = []
        for s in swaps:
            if s.price_sol > 0:
                dev = abs(math.log(s.price_sol) - wmedian_log)
                if dev <= threshold:
                    filtered.append(s)

        return filtered if filtered else swaps

    @staticmethod
    def _compute_vwap_close(swaps: List[SwapRecord], bar_end: int,
                            sol_price_usd: float) -> Tuple[float, bool]:
        """Compute close price as VWAP of last 10s; fallback to last swap."""
        window_start = bar_end - OnChainKlineBuilder.VWAP_CLOSE_WINDOW_SEC
        tail = [s for s in swaps if s.timestamp >= window_start]
        if tail and sum(s.volume_sol for s in tail) > 0:
            vwap_num = sum(s.price_sol * s.volume_sol for s in tail)
            vwap_den = sum(s.volume_sol for s in tail)
            return (vwap_num / vwap_den) * sol_price_usd, True
        return swaps[-1].price_sol * sol_price_usd, False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Pre-grad bonding-curve factor SHADOW support (RPC backfill).
#   bc_pda(mint)                         — pump.fun bonding-curve PDA (re-export)
#   fetch_pregrad_tape(builder,mint,pda) — RPC-backfill the pre-grad swap tape
# OBSERVE-ONLY: used only by ms.shadow.pregrad_factors_shadow; never touches the
# live order / discovery / gRPC path. See L1_LIVE_INTEGRATION_SPEC.md §2.5.
# ──────────────────────────────────────────────────────────────────────────────

try:  # production: controllers package root on sys.path
    from controllers.generic.ms.data.pregrad_factors import (
        bc_pda,
        compute_pregrad_factors,
        score_pregrad_factors,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.data.pregrad_factors import (  # type: ignore[no-redef]
        bc_pda,
        compute_pregrad_factors,
        score_pregrad_factors,
    )


def _parse_bonding_curve_swap(tx_data: dict, mint: str, pda: str) -> Optional["SwapRecord"]:
    """PDA-precise pre-grad bonding-curve swap parse (research ``decode_tx`` parity).

    The pre-grad tape lives on the pump.fun bonding-curve PDA, where the user's
    SOL flows through the curve PDA's OWN lamport balance — not a WSOL token
    vault. The generic ``_parse_swap_from_tx`` (built for post-grad PumpSwap /
    Meteora) mis-handles this two ways, both verified against the research
    ground-truth parser on 22 real pre-grad tx (2026-06-28):

      * ``volume_sol`` over-counts 2–6 %: it sums EVERY account's native lamport
        delta, so the user's pump.fun platform fee (~1 %) + priority fee + ATA
        rent ride along; the curve only RECEIVES the swap principal.
      * NO direction gate: lacking a curve-lamport sign check, migration tx
        (curve drains ~85 SOL → PumpSwap pool: ``lam<0`` + ``tok>0``) and
        non-curve referencing tx (``lam==0``) get mis-admitted, polluting the
        tape.

    This mirrors research ``decode_tx``: take the curve PDA's own lamport delta
    as the exact (fee-free) SOL, and gate on ``lam``/``tok`` sign agreement so
    the factor inputs match the framework thresholds. Returns None for any tx
    that is not a clean curve buy/sell.
    """
    meta = tx_data.get("meta", {})
    if meta.get("err") is not None:
        return None
    block_time = tx_data.get("blockTime", 0)
    if block_time == 0:
        return None
    msg = tx_data.get("transaction", {}).get("message", {})
    keys = [k["pubkey"] if isinstance(k, dict) else k for k in msg.get("accountKeys", [])]
    if not keys or pda not in keys:
        return None
    signer = keys[0]
    bi = keys.index(pda)
    pre_bal = meta.get("preBalances", [])
    post_bal = meta.get("postBalances", [])
    if bi >= len(pre_bal) or bi >= len(post_bal):
        return None
    lam = post_bal[bi] - pre_bal[bi]  # curve PDA lamport delta = exact swap SOL

    def _signer_tok(bals: Optional[list]) -> float:
        return sum(
            float((b.get("uiTokenAmount", {}).get("uiAmount") or 0))
            for b in (bals or [])
            if b.get("mint") == mint and b.get("owner") == signer
        )

    tok = _signer_tok(meta.get("postTokenBalances")) - _signer_tok(meta.get("preTokenBalances"))
    if lam > 0 and tok > 0:
        is_buy = True
    elif lam < 0 and tok < 0:
        is_buy = False
    else:
        return None  # migration / non-curve referencing tx → exclude
    sol_amount = abs(lam) / 1e9
    base_amount = abs(tok)
    if sol_amount <= 0 or base_amount <= 0:
        return None
    return SwapRecord(
        timestamp=block_time,
        price_sol=sol_amount / base_amount,
        volume_sol=sol_amount,
        is_buy=is_buy,
        base_amount=base_amount,
        trader_address=signer,
        slot=int(tx_data.get("slot", 0) or 0),
    )


async def fetch_pregrad_tape(
    builder: "OnChainKlineBuilder",
    mint: str,
    pda: str,
    *,
    target_span_sec: int = 7 * 86400,
    max_pages: int = 60,
    stats: Optional[dict] = None,
) -> List["SwapRecord"]:
    """RPC-backfill the pre-grad bonding-curve swap tape for *mint*.

    Replicates ``OnChainKlineBuilder._backfill_single_pool`` pagination —
    ``getSignaturesForAddress(addr, {limit:1000, before:cursor})`` → batch
    ``getTransaction`` → ``_parse_swap_from_tx`` — but:

      * the paged address is the bonding-curve **PDA** (not a PumpSwap pool);
      * it RETURNS a fresh swap list and writes NOTHING to ``builder._swaps`` /
        ``builder._seen_sigs`` (pure read; no live-state mutation);
      * ``target_span_sec`` is large (7d) so pagination terminates on the EMPTY
        first-graduation page (pre-grad tape is short, < ~10 min) rather than on
        a span check;
      * reuses the builder's ``_get_client`` / ``_fetch_transactions_batch`` for
        RPC, but parses each tx via module-level ``_parse_bonding_curve_swap``
        (PDA-precise, research ``decode_tx`` parity) — NOT the generic
        ``_parse_swap_from_tx``, which on the bonding-curve PDA over-counts SOL
        by fee/priority/rent (2–6 %) and mis-admits migration / non-curve tx
        (verified vs ground truth on 22 real pre-grad tx, 2026-06-28). This
        keeps the factor inputs aligned with the framework thresholds.

    Caller wraps in try/except; any RPC failure raises out of here is swallowed
    by the shadow runner (this never blocks the main loop).
    """
    client = await builder._get_client()
    seen: set = set()
    out: List[SwapRecord] = []
    cursor: Optional[str] = None
    pages = 0

    for _page in range(max_pages):
        params: dict = {"limit": 1000}
        if cursor:
            params["before"] = cursor
        body = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignaturesForAddress",
            "params": [pda, params],
        }
        try:
            resp = await client.post(builder._rpc_url, json=body)
            if resp.status_code == 429:
                await asyncio.sleep(2)
                resp = await client.post(builder._rpc_url, json=body)
            resp.raise_for_status()
            sigs_result = resp.json().get("result", [])
        except Exception as e:
            logger.debug(f"pregrad backfill getSignatures failed page={_page} "
                         f"{mint[:10]}: {e}")
            break

        pages += 1
        if not sigs_result:
            break  # reached the first signature on the PDA → tape complete

        cursor = sigs_result[-1]["signature"]

        new_sigs = []
        for si in sigs_result:
            sig = si["signature"]
            if sig not in seen and si.get("err") is None:
                new_sigs.append(sig)
                seen.add(sig)

        if new_sigs:
            tx_results = await builder._fetch_transactions_batch(new_sigs)
            for sig, tx_data in zip(new_sigs, tx_results):
                if tx_data is None:
                    continue
                swap = _parse_bonding_curve_swap(tx_data, mint, pda)
                if swap:
                    swap.signature = sig
                    out.append(swap)

    out.sort(key=lambda s: s.timestamp)
    if stats is not None:
        stats["pages"] = pages
    return out


# ──────────────────────────────────────────────
# M1: Token Discovery (Three-tier architecture)
#   Tier 1: Chainstack on-chain polling (primary)
#   Tier 2: GMGN token info enrichment
#   Tier 3: GMGN kline features (handled by M2)
# ──────────────────────────────────────────────

class TokenDiscovery:
    """Three-tier token discovery for newly graduated PumpFun tokens.

    Tier 1 — Chainstack on-chain: poll Pump.fun program for PumpSwap create_pool
             inner instructions. Catches ~100% of graduations in real-time.
    Tier 2 — GMGN enrichment: GET /v1/token/info/{chain}/{mint} to get symbol,
             name, liquidity, price, rug_ratio, etc.

    Fallback: If Chainstack is unavailable, falls back to GMGN Trenches API
              (original method, ~4% real-time coverage).
    """

    TOKEN_INFO_PATH = "/v1/token/info"
    TOKEN_SECURITY_PATH = "/v1/token/security"
    TRENCHES_PATH = "/v1/trenches"
    TRENDING_PATH = "/v1/market/rank"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openapi.gmgn.ai",
        min_liquidity_usd: float = 5_000,
        chainstack_rpc_url: str = "",
        chainstack_batch_size: int = 50,
        chainstack_tx_concurrency: int = 3,
        trending_max_token_age_sec: int = 3600,
        instant_buy_mode: bool = False,
        public_rpc_url: str = "",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.min_liquidity_usd = min_liquidity_usd
        self.instant_buy_mode = instant_buy_mode
        self._seen_mints: Dict[str, None] = {}  # ordered dict (insertion order) as set; trimming keeps newest
        self._seen_mints_max: int = 5000
        self._gmgn_client: Optional[httpx.AsyncClient] = None

        # Tier 1: Chainstack on-chain polling
        self._rpc: Optional[SolanaRPC] = None
        self._chainstack_rpc_url = chainstack_rpc_url
        # v0.7.17: optional public RPC for Chainstack-indexing-lag fallback
        self._public_rpc_url = public_rpc_url
        self._batch_size = chainstack_batch_size
        self._tx_concurrency = chainstack_tx_concurrency
        self._last_sig: Optional[str] = None  # cursor for incremental polling
        self._chainstack_primed: bool = False
        self._chainstack_failures: int = 0  # consecutive failures → fallback
        self._chainstack_fallback_since: float = 0.0  # timestamp when fallback started
        self._chainstack_recovery_interval: float = 300.0  # retry Chainstack every 5min during fallback
        self._trending_max_token_age_sec = trending_max_token_age_sec
        self._trending_seen: Dict[str, float] = {}  # mint → timestamp, separate from _seen_mints
        self.sol_price_usd: float = 0.0  # set by controller, used for on-chain liq fallback

        # WebSocket real-time graduation detection
        self._ws_queue: asyncio.Queue = asyncio.Queue()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_connected: bool = False
        self._ws_url = (
            chainstack_rpc_url.replace("https://", "wss://").replace("http://", "ws://")
            if chainstack_rpc_url else ""
        )

        # Yellowstone gRPC Geyser stream (set by controller when configured)
        self._grpc_stream = None

    async def _get_rpc(self) -> Optional[SolanaRPC]:
        """Lazily create Chainstack RPC client. v0.7.17: forwards
        public_rpc_url so indexing-lag fallback is wired automatically
        for all callers (reconcile, orphan_scanner, creator_resolver, etc.)."""
        if not self._chainstack_rpc_url:
            return None
        if self._rpc is None:
            self._rpc = SolanaRPC(
                self._chainstack_rpc_url,
                public_rpc_url=self._public_rpc_url,
            )
        return self._rpc

    async def _get_pool_liquidity_onchain(self, pool_address: str, sol_price_usd: float) -> float:
        """Read pool's WSOL vault balance directly from chain via getTokenAccountsByOwner.
        Returns estimated USD liquidity (SOL side * 2 * price)."""
        rpc = await self._get_rpc()
        if not rpc or not pool_address:
            return 0.0
        try:
            result = await rpc.get_token_accounts_by_owner(pool_address, WSOL_MINT)
            if not result or not result.get("value"):
                return 0.0
            for acct in result["value"]:
                info = (acct.get("account", {}).get("data", {})
                        .get("parsed", {}).get("info", {}))
                ui_amount = float(info.get("tokenAmount", {}).get("uiAmount", 0) or 0)
                if ui_amount > 0:
                    return ui_amount * 2 * sol_price_usd
        except Exception as e:
            logger.debug(f"M1: on-chain liq check failed for pool {pool_address[:12]}...: {e}")
        return 0.0

    async def _get_gmgn_client(self) -> httpx.AsyncClient:
        if self._gmgn_client is None or self._gmgn_client.is_closed:
            self._gmgn_client = httpx.AsyncClient(
                headers={"X-APIKEY": self.api_key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(10.0))
        return self._gmgn_client

    def _auth_params(self) -> Dict[str, str]:
        return {"timestamp": str(int(time.time())), "client_id": str(uuid.uuid4())}

    # ── WebSocket / gRPC real-time graduation detection ─────────────────

    def set_grpc_stream(self, stream):
        """Set the GeyserPumpSwapStream reference. When active, graduation
        detection and swap collection use the gRPC stream instead of WS/HTTP."""
        self._grpc_stream = stream

    def _start_ws_listener(self):
        """Launch the WebSocket listener as a background task (idempotent).
        Skipped when gRPC stream is configured and connected."""
        if self._grpc_stream is not None:
            return  # gRPC handles graduation detection
        if not HAS_WEBSOCKETS or not self._ws_url:
            return
        if self._ws_task is not None and not self._ws_task.done():
            return
        self._ws_task = asyncio.get_event_loop().create_task(self._ws_listen_loop())

    async def _ws_listen_loop(self):
        """Connect to Chainstack WebSocket, subscribe to PumpFun logsSubscribe.

        Filters log notifications for PumpSwap AMM CPI (graduation indicator).
        Only fetches full transaction for likely graduations (~1-5/min),
        achieving 100% coverage with minimal RPC calls vs HTTP polling (~35%).
        """
        backoff = 1
        ws_seen: set = set()

        while True:
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=2 ** 22,       # 4 MB per message
                    close_timeout=5,
                ) as ws:
                    sub_msg = json.dumps({
                        "jsonrpc": "2.0", "id": 1,
                        "method": "logsSubscribe",
                        "params": [
                            {"mentions": [PUMPFUN_PROGRAM]},
                            {"commitment": "confirmed"},
                        ],
                    })
                    await ws.send(sub_msg)
                    resp = json.loads(await ws.recv())
                    sub_id = resp.get("result")
                    logger.info(f"M1-WS: subscribed to PumpFun logs (sub_id={sub_id})")
                    self._ws_connected = True
                    backoff = 1

                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                            if data.get("method") != "logsNotification":
                                continue

                            value = data["params"]["result"]["value"]
                            if value.get("err") is not None:
                                continue

                            logs = value.get("logs", [])
                            if not any(PUMPSWAP_AMM_PROGRAM in line for line in logs):
                                continue

                            sig = value["signature"]
                            if sig in ws_seen:
                                continue
                            ws_seen.add(sig)
                            if len(ws_seen) > 5000:
                                ws_seen = set(list(ws_seen)[-2500:])

                            rpc = await self._get_rpc()
                            if not rpc:
                                continue
                            tx = await rpc.get_transaction(sig)
                            grad = parse_graduation_from_tx(tx, sig)
                            if grad and grad.mint not in self._seen_mints:
                                logger.info(
                                    f"M1-WS: graduation detected — {grad.mint[:16]}... "
                                    f"pool={grad.pool[:16]}..."
                                )
                                await self._ws_queue.put(grad)
                        except Exception as e:
                            logger.debug(f"M1-WS: notification parse error: {e}")

            except asyncio.CancelledError:
                self._ws_connected = False
                logger.info("M1-WS: listener cancelled")
                return
            except Exception as e:
                self._ws_connected = False
                logger.warning(f"M1-WS: disconnected ({e}), reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _prime_chainstack_cursor(self):
        """Set the cursor to the most recent Pump.fun signature (skip historical)."""
        rpc = await self._get_rpc()
        if not rpc:
            return
        try:
            sigs = await rpc.get_signatures_for_address(PUMPFUN_PROGRAM, limit=1)
            if sigs:
                self._last_sig = sigs[0]["signature"]
                logger.info(f"M1: Chainstack cursor primed at {self._last_sig[:16]}...")
            self._chainstack_primed = True
            self._chainstack_failures = 0
            self._chainstack_fallback_since = 0.0
        except Exception as e:
            logger.error(f"M1: Chainstack prime failed: {e}")
            self._chainstack_failures += 1

    async def _poll_chainstack(self) -> List[ChainGraduation]:
        """Tier 1: Detect PumpFun graduations via gRPC (primary), WS, or HTTP (fallback).

        gRPC path (Yellowstone Geyser, zero RPC calls):
          subscribe PumpSwap AMM txs → parse create_pool discriminator
        WebSocket path (100% coverage, ~1-5 getTransaction calls/min):
          logsSubscribe → filter for PumpSwap CPI → getTransaction → parse
        HTTP fallback (~35% coverage, ~800 getTransaction calls/30s):
          getSignaturesForAddress → getTransaction on all → parse
        """
        # Priority 1: gRPC Geyser stream (writes to same _ws_queue)
        if self._grpc_stream is not None:
            if not self._grpc_stream.connected:
                self._grpc_stream.start()
            grpc_grads: List[ChainGraduation] = []
            while not self._ws_queue.empty():
                try:
                    grpc_grads.append(self._ws_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if grpc_grads:
                self._chainstack_failures = 0
                self._chainstack_fallback_since = 0.0
                return grpc_grads
            if self._grpc_stream.connected:
                return []  # gRPC live, no graduations this tick
            # gRPC disconnected — fall through to WS/HTTP

        # Priority 2: WebSocket
        self._start_ws_listener()

        # Drain WebSocket queue
        ws_grads: List[ChainGraduation] = []
        while not self._ws_queue.empty():
            try:
                ws_grads.append(self._ws_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if ws_grads:
            self._chainstack_failures = 0
            self._chainstack_fallback_since = 0.0
            return ws_grads

        if self._ws_connected:
            return []  # WS is live, just no graduations this tick

        # ── HTTP fallback (WS not available or disconnected) ──
        rpc = await self._get_rpc()
        if not rpc:
            return []

        if not self._chainstack_primed:
            await self._prime_chainstack_cursor()
            return []

        try:
            sigs = await rpc.get_signatures_for_address(
                PUMPFUN_PROGRAM,
                until=self._last_sig,
                limit=self._batch_size,
            )
        except Exception as e:
            logger.error(f"M1: Chainstack getSignatures failed: {e}")
            self._chainstack_failures += 1
            return []

        if not sigs:
            self._chainstack_failures = 0
            return []

        self._last_sig = sigs[0]["signature"]

        valid_sigs = [s for s in sigs if s.get("err") is None]
        logger.debug(f"M1: HTTP fallback fetched {len(sigs)} sigs, {len(valid_sigs)} successful")

        sem = asyncio.Semaphore(self._tx_concurrency)
        graduations: List[ChainGraduation] = []

        async def process_sig(sig_info: dict):
            sig = sig_info["signature"]
            async with sem:
                tx = await rpc.get_transaction(sig)
            grad = parse_graduation_from_tx(tx, sig)
            if grad and grad.mint not in self._seen_mints:
                graduations.append(grad)

        tasks = [process_sig(s) for s in valid_sigs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"M1: tx parse error: {r}")

        self._chainstack_failures = 0
        self._chainstack_fallback_since = 0.0
        graduations.sort(key=lambda g: g.block_time, reverse=True)
        return graduations

    # Sentinels to distinguish rejection reasons
    _FILTERED_OUT = "FILTERED"          # permanent skip (bad token quality)
    _FILTERED_LOW_LIQ = "FILTERED_LIQ"  # temporary skip (liquidity may grow)

    async def _enrich_instant(self, grad: ChainGraduation):
        """Instant-buy enrichment: zero GMGN calls, pure on-chain.

        With discriminator-based parsing in parse_graduation_from_tx, old-token
        false positives are eliminated at the parser level (swap_buy/swap_sell
        have different discriminators than create_pool). So we no longer need
        GMGN's open_timestamp for freshness verification.

        Single check: on-chain pool liquidity via RPC (~200ms).

        Returns GraduatedToken on pass, _FILTERED_LOW_LIQ if liquidity too low,
        None on RPC failure.
        """
        # On-chain liquidity check via pool WSOL vault (~200ms)
        sol_px = self.sol_price_usd if self.sol_price_usd > 0 else 80.0
        onchain_liq = await self._get_pool_liquidity_onchain(grad.pool, sol_px)

        if onchain_liq < self.min_liquidity_usd:
            logger.debug(f"M1-instant: {grad.mint[:12]}... liq=${onchain_liq:.0f} "
                         f"< ${self.min_liquidity_usd:.0f} — skipped")
            return self._FILTERED_LOW_LIQ

        # No GMGN call — use mint prefix as placeholder symbol.
        # Real name will be fetched asynchronously by M6 observation logger
        # (gmgn_info_entry snapshot) after the buy, not blocking the trade path.
        symbol = grad.mint[:6] if len(grad.mint) > 6 else grad.mint

        return GraduatedToken(
            mint_address=grad.mint,
            symbol=symbol,
            name=symbol,
            decimals=6,
            graduation_time=float(grad.block_time),
            liquidity_usd=onchain_liq,
            price_usd=0.0,  # M3 preflight will get price from Jupiter
            pool_address=grad.pool,
            source="chainstack",
        )

    async def _enrich_from_gmgn(self, grad: ChainGraduation):
        """Tier 2: Enrich a chain-detected graduation with GMGN token info.

        GET /v1/token/info?chain=sol&address={mint}
        Returns top-level: symbol, name, liquidity, price, open_timestamp
        Returns nested:    stat.top_10_holder_rate, stat.top_rat_trader_percentage, etc.

        Returns:
        - GraduatedToken on success
        - _FILTERED_OUT string if token fails quality filter (should be permanently skipped)
        - None if GMGN API failed (should be retried)
        """
        client = await self._get_gmgn_client()
        params = {**self._auth_params(), "chain": "sol", "address": grad.mint}
        try:
            resp = await client.get(
                f"{self.base_url}{self.TOKEN_INFO_PATH}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.debug(f"M1: GMGN info error for {grad.mint[:12]}...: {data.get('msg', '')}")
                return None  # API error — retryable
            token_data = data.get("data", {})
        except Exception as e:
            logger.warning(f"M1: GMGN enrichment failed for {grad.mint[:12]}...: {e}")
            return None  # Network error — retryable

        # ── 1. Age check (cheapest, no RPC) ──
        symbol = token_data.get("symbol", "UNKNOWN")
        open_timestamp = float(token_data.get("open_timestamp", 0) or 0)
        creation_timestamp = float(token_data.get("creation_timestamp", 0) or 0)
        now_ts = time.time()

        if open_timestamp <= 0 and creation_timestamp > 0:
            creation_age = now_ts - creation_timestamp
            if creation_age > 1800:
                logger.debug(f"M1: {symbol} skipped — no open_timestamp and creation "
                             f"too old ({creation_age:.0f}s, created={creation_timestamp:.0f}, "
                             f"block_ts={grad.block_time})")
                return self._FILTERED_OUT

        if open_timestamp > 0:
            graduation_time = open_timestamp
        elif grad.block_time > 0:
            graduation_time = float(grad.block_time)
        else:
            graduation_time = creation_timestamp
        if graduation_time > 0:
            real_age = now_ts - graduation_time
            if real_age > 600:
                logger.debug(f"M1: {symbol} skipped — too old "
                             f"(real_age={real_age:.0f}s, grad_time={graduation_time:.0f}, "
                             f"open_ts={open_timestamp:.0f}, block_ts={grad.block_time})")
                return self._FILTERED_OUT

        # ── 2. Liquidity check (may trigger RPC fallback) ──
        liquidity = float(token_data.get("liquidity", 0) or 0)
        if liquidity < self.min_liquidity_usd:
            sol_px = self.sol_price_usd if self.sol_price_usd > 0 else 80.0
            onchain_liq = await self._get_pool_liquidity_onchain(grad.pool, sol_px)
            if onchain_liq >= self.min_liquidity_usd:
                logger.info(f"M1: {grad.mint[:12]}... GMGN liq=${liquidity:.0f} "
                            f"but on-chain liq=${onchain_liq:.0f} — accepting")
                liquidity = onchain_liq
            else:
                logger.debug(f"M1: {grad.mint[:12]}... skipped — "
                             f"liq GMGN=${liquidity:.0f} chain=${onchain_liq:.0f} "
                             f"< ${self.min_liquidity_usd:.0f}")
                return self._FILTERED_LOW_LIQ

        # ── 3. Safety checks ──
        stat = token_data.get("stat", {})

        top_10_rate = float(stat.get("top_10_holder_rate", 0) or 0)
        if top_10_rate > 0.7:
            logger.debug(f"M1: {symbol} skipped — top_10_holder_rate={top_10_rate:.2f}")
            return self._FILTERED_OUT

        creator_rate = float(stat.get("creator_hold_rate", 0) or 0)
        if creator_rate > 0.30:
            logger.info(f"M1: {symbol} skipped — creator_hold_rate={creator_rate:.1%}")
            return self._FILTERED_OUT

        entrapment_rate = float(stat.get("top_entrapment_trader_percentage", 0) or 0)
        if entrapment_rate > 0.30:
            logger.info(f"M1: {symbol} skipped — entrapment={entrapment_rate:.1%}")
            return self._FILTERED_OUT

        holder_count = int(token_data.get("holder_count", 0) or 0)
        if holder_count < 20:
            logger.debug(f"M1: {symbol} skipped — holder_count={holder_count}")
            return self._FILTERED_OUT

        token_standard = token_data.get("standard", "")
        if str(token_standard) == "2022":
            pool_data = token_data.get("pool", {})
            fee_ratio = float(pool_data.get("fee_ratio", 0) or 0)
            if fee_ratio > 0.01:
                logger.info(f"M1: {symbol} skipped — Token-2022 transfer fee={fee_ratio:.2%}")
                return self._FILTERED_OUT

        name = token_data.get("name", symbol)
        price = _coerce_price(token_data.get("price"))

        return GraduatedToken(
            mint_address=grad.mint,
            symbol=symbol,
            name=name,
            decimals=6,
            graduation_time=graduation_time,
            liquidity_usd=liquidity,
            price_usd=price,
            pool_address=grad.pool,
            swaps_24h=None,
            buys_24h=None,
            sells_24h=None,
            _gmgn_info_raw=token_data,
        )

    async def _poll_gmgn_trenches_fallback(self) -> List[GraduatedToken]:
        """Fallback: Original GMGN Trenches API polling (~4% real-time coverage)."""
        client = await self._get_gmgn_client()
        params = {**self._auth_params(), "chain": "sol"}
        body = {
            "version": "v2",
            "completed": {
                "filters": ["offchain", "onchain"],
                "launchpad_platform": ["Pump.fun"],
                "launchpad_platform_v2": True,
                "limit": 40,
            },
        }
        try:
            resp = await client.post(
                f"{self.base_url}{self.TRENCHES_PATH}",
                params=params, json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.warning(f"GMGN trenches error: {data}")
                return []
            items = data.get("data", {}).get("completed", [])
        except Exception as e:
            logger.error(f"GMGN trenches request failed: {e}")
            return []

        new_tokens = []
        now = time.time()
        for item in items:
            mint = item.get("address", "")
            if not mint or mint in self._seen_mints:
                continue
            liquidity = float(item.get("liquidity", 0) or 0)
            if liquidity < self.min_liquidity_usd:
                continue
            grad_time = float(item.get("open_timestamp") or item.get("creation_timestamp") or 0)
            if grad_time > 0 and (now - grad_time) > 600:
                continue
            self._seen_mints.setdefault(mint)
            new_tokens.append(GraduatedToken(
                mint_address=mint,
                symbol=item.get("symbol", "UNKNOWN"),
                name=item.get("name", item.get("symbol", "UNKNOWN")),
                decimals=6,
                graduation_time=float(item.get("open_timestamp") or item.get("creation_timestamp") or now),
                liquidity_usd=liquidity,
                price_usd=_coerce_price(item.get("price")),
                swaps_24h=int(item.get("swaps_24h", 0) or 0),
                buys_24h=int(item.get("buys_24h", 0) or 0),
                sells_24h=int(item.get("sells_24h", 0) or 0),
                # grad 锚 = GMGN open_timestamp(合成,非链上 block_time)——
                # 诚实标注供链上锚消费方(xoscross)按 source 过滤(2026-07-05
                # 审计 #3;此前缺省顶着 "chainstack" 标)。
                source="trenches_fallback",
            ))
        return new_tokens

    async def poll_new_graduations(self) -> List[GraduatedToken]:
        """Poll for newly graduated tokens using three-tier architecture.

        Primary path (Chainstack available):
          1. Tier 1: Detect graduations on-chain via Chainstack RPC
          2. Tier 2: Enrich each graduation with GMGN token info
          3. Filter: min liquidity, rug_ratio, dedup via _seen_mints

        Fallback path (Chainstack unavailable or 3+ consecutive failures):
          Use GMGN Trenches API directly (original method)
        """
        # Fallback to GMGN Trenches if Chainstack is not configured or failing
        use_chainstack = (
            self._chainstack_rpc_url
            and self._chainstack_failures < 3
        )

        # Recovery: periodically retry Chainstack during fallback
        if not use_chainstack and self._chainstack_rpc_url and self._chainstack_failures >= 3:
            now = time.time()
            if now - self._chainstack_fallback_since >= self._chainstack_recovery_interval:
                logger.info(f"M1: Attempting Chainstack recovery after {self._chainstack_failures} failures")
                self._chainstack_failures = 0
                self._chainstack_primed = False  # re-prime cursor
                self._last_sig = None
                use_chainstack = True

        if not use_chainstack:
            if self._chainstack_rpc_url and self._chainstack_failures >= 3:
                if self._chainstack_fallback_since == 0:
                    self._chainstack_fallback_since = time.time()
                logger.warning(f"M1: Chainstack has {self._chainstack_failures} consecutive failures, "
                               f"falling back to GMGN Trenches")
            return await self._poll_gmgn_trenches_fallback()

        # Tier 1: On-chain detection
        chain_grads = await self._poll_chainstack()
        if not chain_grads:
            return []

        # Tier 2: Enrichment
        # In instant_buy_mode: skip GMGN API entirely, use on-chain RPC for
        # liquidity check only (~200ms vs 3-5s). Saves 5-8s per token.
        # In normal mode: full GMGN enrichment with holder/creator/sybil checks.
        enriched: List[GraduatedToken] = []
        now = time.time()
        for grad in chain_grads:
            if grad.mint in self._seen_mints:
                continue
            # Skip old graduations — in instant mode use tight 30s, otherwise 600s
            max_age = 30 if self.instant_buy_mode else 600
            if grad.block_time > 0 and (now - grad.block_time) > max_age:
                self._seen_mints.setdefault(grad.mint)
                if self.instant_buy_mode:
                    logger.debug(f"M1: {grad.mint[:12]}... skipped — too old "
                                 f"(age={now - grad.block_time:.0f}s > {max_age}s)")
                continue

            if self.instant_buy_mode:
                result = await self._enrich_instant(grad)
            else:
                result = await self._enrich_from_gmgn(grad)

            if isinstance(result, GraduatedToken):
                self._seen_mints.setdefault(grad.mint)
                enriched.append(result)
                tag = "INSTANT" if self.instant_buy_mode else "NEW"
                logger.info(
                    f"M1: [{tag}] {result.symbol} | mint={result.mint_address} | "
                    f"liq=${result.liquidity_usd:.0f} | "
                    f"age={now - grad.block_time:.1f}s | source=chainstack"
                )
            elif result == self._FILTERED_OUT:
                self._seen_mints.setdefault(grad.mint)
            elif result == self._FILTERED_LOW_LIQ:
                # Low liquidity — do NOT mark seen, will retry next poll
                pass
            else:
                logger.debug(f"M1: {grad.mint[:12]}... enrichment failed, will retry")

        # Prevent unbounded set growth
        if len(self._seen_mints) > self._seen_mints_max:
            keep_n = self._seen_mints_max // 2
            keys = list(self._seen_mints.keys())
            self._seen_mints = {k: None for k in keys[-keep_n:]}
            logger.info(f"M1: trimmed _seen_mints to {len(self._seen_mints)}")

        return enriched

    async def discover_additional_pools(self, mint: str) -> Optional[str]:
        """Query GMGN token info for biggest_pool_address.

        Returns the biggest pool address if it differs from None, or None on failure.
        Used during pending period to discover higher-liquidity pools on other DEXes
        (e.g. Meteora, Raydium) that may appear after PumpSwap graduation.
        """
        client = await self._get_gmgn_client()
        params = {**self._auth_params(), "chain": "sol", "address": mint}
        try:
            resp = await client.get(
                f"{self.base_url}{self.TOKEN_INFO_PATH}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return None
            token_data = data.get("data", {})
            biggest = token_data.get("biggest_pool_address")
            if biggest and isinstance(biggest, str) and len(biggest) > 20:
                return biggest
        except Exception as e:
            logger.debug(f"M1: pool discovery failed for {mint[:12]}...: {e}")
        return None

    async def validate_trending_token(self, mint: str, min_liq_usd: float = 50000
                                      ) -> Optional[Dict[str, Any]]:
        """Cross-validate a trending token via GMGN token info API.

        Trending API liquidity can be fake/inflated. This method fetches the real
        token info and checks:
        1. Real pool liquidity >= min_liq_usd
        2. top_10_holder_rate <= 0.70  (not whale-dominated)
        3. creator_hold_rate <= 0.30   (dev not holding majority)
        4. biggest_pool_address exists

        Returns dict with 'pool_address', 'real_liquidity', etc. on pass, or None on reject.
        """
        client = await self._get_gmgn_client()
        params = {**self._auth_params(), "chain": "sol", "address": mint}
        try:
            resp = await client.get(
                f"{self.base_url}{self.TOKEN_INFO_PATH}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.debug(f"M1b: validate {mint[:12]}... GMGN error code={data.get('code')}")
                return None
            token_data = data.get("data", {})
        except Exception as e:
            logger.debug(f"M1b: validate {mint[:12]}... request failed: {e}")
            return None

        # Extract real liquidity from pool object (not top-level which may be stale)
        pool = token_data.get("pool", {})
        pool_liq_raw = pool.get("liquidity", 0)
        # GMGN pool.liquidity is sometimes a string ratio, sometimes USD
        # Check quote_reserve_value for actual USD value
        quote_reserve_val = float(pool.get("quote_reserve_value", 0) or 0)
        # Real pool liquidity = 2 * quote_reserve_value (for AMM pools)
        real_liq = quote_reserve_val * 2 if quote_reserve_val > 0 else float(pool_liq_raw or 0)

        biggest_pool = token_data.get("biggest_pool_address")
        stat = token_data.get("stat", {})
        top_10_rate = float(stat.get("top_10_holder_rate", 0) or 0)
        creator_rate = float(stat.get("creator_hold_rate", 0) or 0)
        entrapment_rate = float(stat.get("top_entrapment_trader_percentage", 0) or 0)
        holder_count = int(token_data.get("holder_count", 0) or 0)

        symbol = token_data.get("symbol", mint[:8])

        # Reject: real liquidity too low
        if real_liq < min_liq_usd:
            logger.info(f"M1b: {symbol} REJECT — real_liq=${real_liq:.2f} "
                        f"(trending claimed much higher, token info says ${real_liq:.2f})")
            return None

        # Reject: top 10 holders own > 50%
        if top_10_rate > 0.70:
            logger.info(f"M1b: {symbol} REJECT — top_10_holder_rate={top_10_rate:.1%} "
                        f"(whale-dominated)")
            return None

        # Reject: creator still holds > 30%
        if creator_rate > 0.30:
            logger.info(f"M1b: {symbol} REJECT — creator_hold_rate={creator_rate:.1%} "
                        f"(dev holding majority)")
            return None

        # Reject: high entrapment trader percentage
        if entrapment_rate > 0.30:
            logger.info(f"M1b: {symbol} REJECT — entrapment_trader_pct={entrapment_rate:.1%}")
            return None

        # Reject: too few holders (likely fake/wash trading)
        if holder_count < 20:
            logger.info(f"M1b: {symbol} REJECT — holder_count={holder_count} (too few)")
            return None

        if not biggest_pool or len(biggest_pool) < 20:
            logger.info(f"M1b: {symbol} REJECT — no biggest_pool_address")
            return None

        # Token-2022: only reject if transfer fee extension is active
        token_standard = token_data.get("standard", "")
        if str(token_standard) == "2022":
            pool_obj = token_data.get("pool", {})
            fee_ratio = float(pool_obj.get("fee_ratio", 0) or 0)
            if fee_ratio > 0.01:
                logger.info(f"M1b: {symbol} REJECT — Token-2022 transfer fee={fee_ratio:.2%}")
                return None

        logger.info(f"M1b: {symbol} VALIDATED — real_liq=${real_liq:.0f}, "
                    f"top10={top_10_rate:.1%}, creator={creator_rate:.1%}, "
                    f"holders={holder_count}, pool={biggest_pool[:12]}...")
        return {
            "pool_address": biggest_pool,
            "real_liquidity": real_liq,
            "top_10_holder_rate": top_10_rate,
            "creator_hold_rate": creator_rate,
            "holder_count": holder_count,
        }

    async def get_token_market_snapshot(self, mint: str) -> Optional[Dict[str, Any]]:
        """Fetch current GMGN token snapshot for execution-time revalidation.

        Returns a compact dict with the live liquidity snapshot and biggest pool
        information, or None if GMGN is temporarily unavailable.
        """
        client = await self._get_gmgn_client()
        params = {**self._auth_params(), "chain": "sol", "address": mint}
        try:
            resp = await client.get(
                f"{self.base_url}{self.TOKEN_INFO_PATH}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return None
            token_data = data.get("data", {})
            stat = token_data.get("stat", {})
            # Compute real liquidity from pool reserves (top-level `liquidity` can be fake)
            pool = token_data.get("pool", {})
            quote_reserve_val = float(pool.get("quote_reserve_value", 0) or 0)
            real_liq = quote_reserve_val * 2 if quote_reserve_val > 0 else float(token_data.get("liquidity", 0) or 0)
            return {
                "symbol": token_data.get("symbol", ""),
                "liquidity_usd": real_liq,
                "price_usd": _coerce_price(token_data.get("price")),
                "biggest_pool_address": token_data.get("biggest_pool_address"),
                "top_10_holder_rate": float(stat.get("top_10_holder_rate", 0) or 0),
            }
        except Exception as e:
            logger.debug(f"M1: token snapshot failed for {mint[:12]}...: {e}")
            return None

    async def fetch_hot_rank_raw(self, max_age_sec: int = 1800) -> List[Dict]:
        """Fetch GMGN hot rank for M1 coverage cross-check (observation only).

        Filters used:
          - chain=sol, platforms=Pump.fun
          - filters: renounced (mint authority discarded), frozen (no blacklist),
            not_wash_trading (no fake volume)
          - launchpad_status=='1' (graduated to main DEX, "外盘")
          - age in [0, max_age_sec] (default 30 minutes)

        Returns raw rank items (NOT GraduatedToken) for double-check / coverage analysis.
        Does NOT touch _seen_mints or _trending_seen — pure observation, no side effects.
        """
        client = await self._get_gmgn_client()
        params = [
            ("client_id", str(uuid.uuid4())),
            ("timestamp", str(int(time.time()))),
            ("chain", "sol"),
            ("interval", "5m"),
            ("limit", "100"),
            ("order_by", "volume"),
            ("direction", "desc"),
            ("platforms", "Pump.fun"),
            ("filters", "renounced"),
            ("filters", "frozen"),
            ("filters", "not_wash_trading"),
        ]
        try:
            resp = await client.get(
                f"{self.base_url}{self.TRENDING_PATH}",
                params=params,
            )
            resp.raise_for_status()
            raw = resp.json()
            outer = raw.get("data", {})
            if isinstance(outer, dict) and "data" in outer:
                inner = outer.get("data", {})
            else:
                inner = outer
            items = inner.get("rank", [])
            if not isinstance(items, list):
                logger.debug(f"hot_rank: unexpected response shape")
                return []
        except Exception as e:
            logger.debug(f"hot_rank: request failed: {e}")
            return []

        # Client-side filter: 外盘 (launchpad_status='1') AND age <= max_age_sec
        now = time.time()
        filtered = []
        for item in items:
            if str(item.get("launchpad_status", "")) != "1":
                continue
            ots = float(item.get("open_timestamp") or item.get("creation_timestamp") or 0)
            if ots <= 0:
                continue
            age = now - ots
            if age < 0 or age > max_age_sec:
                continue
            filtered.append(item)
        return filtered

    async def poll_gmgn_trending(self, exclude_mints: Optional[set] = None) -> List[GraduatedToken]:
        """Discover high-liquidity tokens via GMGN market trending API.

        Returns tokens with liq > min_liquidity_usd, age < trending_max_token_age_sec.
        Uses separate _trending_seen set (not _seen_mints) so Chainstack-discovered
        tokens that failed initial eval (e.g. low liq) can be re-discovered when
        they appear on trending with higher liquidity.
        exclude_mints: active mints (pending + positions) to skip.
        """
        client = await self._get_gmgn_client()
        params = {
            **self._auth_params(),
            "chain": "sol",
            "interval": "5m",
            "order_by": "volume",
            "limit": "50",
        }
        try:
            resp = await client.get(
                f"{self.base_url}{self.TRENDING_PATH}",
                params=params,
            )
            logger.debug(f"M1b: GMGN trending HTTP {resp.status_code}, "
                         f"content-type={resp.headers.get('content-type', '')}")
            resp.raise_for_status()
            raw = resp.json()
            # GMGN wraps responses: {code, data: {code, data: {rank: [...]}, message, reason}}
            outer = raw.get("data", {})
            if isinstance(outer, dict) and "data" in outer:
                inner = outer.get("data", {})
            else:
                inner = outer
            if raw.get("code") != 0 and outer.get("code") != 0:
                logger.warning(f"M1b: GMGN trending error: {outer.get('message', '')}")
                return []
            items = inner.get("rank", [])
            if not isinstance(items, list):
                logger.warning(f"M1b: GMGN trending unexpected response shape: {list(inner.keys())}")
                return []
            logger.debug(f"M1b: GMGN trending returned {len(items)} tokens, "
                         f"seen_mints={len(self._seen_mints)}")
        except Exception as e:
            logger.error(f"M1b: GMGN trending request failed: {e}")
            return []

        new_tokens: List[GraduatedToken] = []
        now = time.time()
        _excl = exclude_mints or set()

        # Purge trending_seen entries older than max_age (allow re-discovery of stale entries)
        cutoff = now - self._trending_max_token_age_sec
        expired = [m for m, ts in self._trending_seen.items() if ts < cutoff]
        for m in expired:
            del self._trending_seen[m]

        for item in items:
            mint = item.get("address", "")
            if not mint:
                continue
            # Skip tokens already being tracked (pending/positions) or recently seen by trending
            if mint in _excl or mint in self._trending_seen:
                continue

            liquidity = float(item.get("liquidity", 0) or 0)
            if liquidity < self.min_liquidity_usd:
                continue

            open_ts = float(item.get("open_timestamp", 0) or 0)
            if open_ts <= 0:
                continue
            age = now - open_ts
            if age > self._trending_max_token_age_sec or age < 0:
                continue

            symbol = item.get("symbol", "UNKNOWN")

            top_10_rate = float(item.get("top_10_holder_rate", 0) or 0)
            if top_10_rate > 0.7:
                logger.debug(f"M1b: {symbol} skipped — top_10_holder={top_10_rate:.2f}")
                continue

            # Anti-sybil: high bundler_rate means most buys came from bot/sybil wallets.
            # Direct-deploy tokens (no launchpad) use sybil wallets to fake distribution,
            # so apply a stricter bundler threshold for them.
            bundler_rate = float(item.get("bundler_rate", 0) or 0)
            launchpad = item.get("launchpad_platform", "") or ""
            bundler_limit = 0.30 if not launchpad else 0.50
            if bundler_rate > bundler_limit:
                logger.info(f"M1b: {symbol} skipped — bundler_rate={bundler_rate:.2f} "
                            f"> {bundler_limit:.2f} (lp={launchpad or 'NONE'})")
                continue

            # Wash trading detection
            if item.get("is_wash_trading"):
                logger.info(f"M1b: {symbol} skipped — wash trading detected")
                continue

            # Creator must have sold (creator_close=true)
            if not item.get("creator_close", False):
                logger.debug(f"M1b: {symbol} skipped — creator still holding")
                continue

            self._trending_seen[mint] = now
            # Also add to _seen_mints so Chainstack doesn't re-discover it
            self._seen_mints.setdefault(mint)
            new_tokens.append(GraduatedToken(
                mint_address=mint,
                symbol=symbol,
                name=item.get("name", symbol),
                decimals=6,
                graduation_time=open_ts,
                liquidity_usd=liquidity,
                price_usd=_coerce_price(item.get("price")),
                pool_address=None,
                source="trending",
                _gmgn_info_raw=item,
            ))
            logger.info(
                f"M1b: [TRENDING] {symbol} | mint={mint} | "
                f"liq=${liquidity:.0f} | age={age:.0f}s | "
                f"bundler={bundler_rate:.2f} | launchpad={launchpad} | "
                f"price=${float(item.get('price', 0) or 0):.8f}"
            )

        return new_tokens

    async def close(self):
        if self._gmgn_client and not self._gmgn_client.is_closed:
            await self._gmgn_client.aclose()
        if self._rpc:
            await self._rpc.close()


# ──────────────────────────────────────────────
# M2: Signal Pipeline (GMGN kline + GBT model)
# ──────────────────────────────────────────────

class SignalPipeline:
    """Collect features from kline and predict P(alive) with GBT model.

    Supports two model versions:
      - Legacy v1 (survival_model.pkl): 6 features from GMGN 5min kline
      - P10 v1 (p10_production_model.pkl): 20 features from 1m on-chain bars
    """

    KLINE_PATH = "/v1/market/token_kline"
    LEGACY_FEATURE_NAMES = [
        "5min_return", "5min_volume_usd", "5min_trade_count",
        "5min_buy_sell_ratio", "5min_volatility", "5min_stale_pct",
    ]

    def __init__(self, model_path: str, api_key: str,
                 base_url: str = "https://openapi.gmgn.ai",
                 *, metrics: Optional[Any] = None):
        self.guarded = _model_registry.get_or_load(model_path, metrics=metrics)
        model_data = self.guarded.unwrap()
        self.model = model_data["model"]
        self.scaler = model_data.get("scaler")
        self.model_type = model_data.get("model_type", "classification")
        # Auto-detect model version
        version_str = model_data.get("version", "")
        if version_str.startswith("p11"):
            self.model_version = "p11"
            self.feature_names = model_data["feature_cols"]
            self.clip_bounds = model_data.get("clip_bounds")
            self.threshold = model_data["threshold"]
            self.lookback_bars = model_data.get("lookback_bars", 3)
            logger.info(f"SignalPipeline: loaded P11 model ({self.model_type}, "
                        f"{len(self.feature_names)} features, thr={self.threshold:.4f})")
        elif version_str.startswith("p10"):
            self.model_version = "p10"
            self.feature_names = model_data["feature_cols"]
            self.clip_bounds = model_data.get("clip_bounds")
            self.threshold = model_data.get("threshold", 0.30)
            self.lookback_bars = model_data.get("lookback_bars", 3)
            logger.info(f"SignalPipeline: loaded P10 model v1 ({len(self.feature_names)} features)")
        else:
            self.model_version = "legacy"
            self.feature_names = model_data.get("feature_names", self.LEGACY_FEATURE_NAMES)
            self.clip_bounds = None
            self.threshold = 0.50
            self.lookback_bars = 0
            logger.info(f"SignalPipeline: loaded legacy model ({len(self.feature_names)} features)")

        # Default min bars from model; caller can override via set_min_bars()
        self._min_bars = 6 if self.model_version in ("p10", "p11") else 5

        self.api_key = api_key
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    def set_min_bars(self, n: int):
        """Override minimum bar requirement (e.g. P12 uses 3 bars instead of 6)."""
        self._min_bars = n

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"X-APIKEY": self.api_key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(10.0))
        return self._client

    def _auth_params(self) -> Dict[str, str]:
        return {"timestamp": str(int(time.time())), "client_id": str(uuid.uuid4())}

    async def collect_features(self, token: GraduatedToken,
                               kline_bars: Optional[List[Dict]] = None,
                               kline_shifted: bool = False) -> Optional[Dict[str, float]]:
        """Compute features from kline bars.

        Routes to legacy or P10 feature extraction based on model version.
        When on-chain kline is used, cross-validates against GMGN kline to
        detect outlier-contaminated bars.

        If kline_shifted=True and model is P10/P11, on-chain bars are from
        a shifted time window (not graduation + 0..5min), so we fallback
        directly to GMGN graduation-window bars without cross-validation.
        """
        has_trade_data = False
        min_bars = self._min_bars
        is_fixed_window_model = self.model_version in ("p10", "p11")

        if kline_bars is not None and len(kline_bars) >= min_bars:
            bars = kline_bars[:min_bars]
            has_trade_data = True

            if kline_shifted:
                logger.warning(f"M2: {token.symbol} on-chain kline shifted — using as-is")
            # Cross-validate on-chain return vs GMGN (log-only, non-blocking)
            gmgn_bars = await self._fetch_gmgn_kline(token, n_bars=min_bars)
            if gmgn_bars and len(gmgn_bars) >= min_bars:
                xv_result = self._cross_validate_bars(
                    token.symbol, bars, gmgn_bars[:min_bars])
                if xv_result is not None:
                    bars = xv_result
                else:
                    logger.warning(f"M2: {token.symbol} XV failed — proceeding with on-chain bars")
        else:
            # Insufficient on-chain bars — try GMGN fallback
            bars = await self._fetch_gmgn_kline(token, n_bars=min_bars)
            if bars is None or len(bars) < min_bars:
                logger.warning(f"M2: {token.symbol} insufficient bars — "
                               f"on-chain={len(kline_bars) if kline_bars else 0}, "
                               f"gmgn={len(bars) if bars else 0}, need {min_bars}")
                return None
            has_trade_data = False
            logger.info(f"M2: {token.symbol} using GMGN kline fallback ({len(bars)} bars)")

        if self.model_version in ("p10", "p11"):
            return self._compute_p10_features(token, bars)
        return self._compute_legacy_features(token, bars, has_trade_data)

    # Cross-validation: compare cumulative return only (not single-bar close).
    # Single-bar close diff is misleading because:
    #   - on-chain close = VWAP of last 10s + outlier filter (single pool)
    #   - GMGN close = last trade in bar (multi-pool aggregated)
    # These are computed differently even on identical raw data.
    # cum_return is more robust because it absorbs single-bar noise.
    XV_RETURN_DIFF_THRESHOLD = 0.50  # |chain_ret - gmgn_ret| > 50pp → reject

    def _cross_validate_bars(self, symbol: str,
                             chain_bars: List[Dict],
                             gmgn_bars: List[Dict]) -> Optional[List[Dict]]:
        """Cross-validate on-chain vs GMGN by cumulative return only.

        Trust on-chain bars (P11 model is trained on the same on-chain pipeline).
        GMGN serves only as a sanity check on the overall direction/magnitude.
        Returns:
          - on-chain bars if returns agree within XV_RETURN_DIFF_THRESHOLD
          - None if returns diverge (token data is suspect, reject entirely)
        """
        try:
            chain_c0 = float(chain_bars[0].get("close", 0))
            chain_cn = float(chain_bars[-1].get("close", 0))
            chain_ret = (chain_cn / chain_c0 - 1) if chain_c0 > 0 else 0

            gmgn_c0 = float(gmgn_bars[0].get("close", 0)) if gmgn_bars else 0
            gmgn_cn = float(gmgn_bars[-1].get("close", 0)) if gmgn_bars else 0
            gmgn_ret = (gmgn_cn / gmgn_c0 - 1) if gmgn_c0 > 0 else None

            if gmgn_ret is None:
                logger.info(f"M2: {symbol} cross-validate skipped — no GMGN reference, "
                            f"chain_ret={chain_ret:+.1%}")
                return chain_bars

            ret_diff = abs(chain_ret - gmgn_ret)
            if ret_diff > self.XV_RETURN_DIFF_THRESHOLD:
                logger.warning(f"M2: {symbol} cross-validate FAIL — "
                               f"chain_ret={chain_ret:+.1%}, gmgn_ret={gmgn_ret:+.1%}, "
                               f"diff={ret_diff*100:.0f}pp > {self.XV_RETURN_DIFF_THRESHOLD*100:.0f}pp "
                               f"→ REJECT (data unreliable)")
                return None
            logger.info(f"M2: {symbol} cross-validate OK — "
                        f"chain_ret={chain_ret:+.1%}, gmgn_ret={gmgn_ret:+.1%}, "
                        f"diff={ret_diff*100:.0f}pp")
            return chain_bars
        except Exception as e:
            logger.warning(f"M2: {symbol} cross-validate error: {e}")
            return chain_bars

    def _compute_p10_features(self, token: GraduatedToken,
                              bars: List[Dict]) -> Optional[Dict[str, float]]:
        """Compute 20 P10/P11 features from kline bars.

        P10/P11 (6 bars): lookback = bars[3:6], cumulative = bars[0:6]
        P12-A  (3 bars): lookback = all bars, cumulative = all bars

        Bar dict expected keys: open, high, low, close, volume (or volume_usd), trades
        """
        if len(bars) < 3:
            logger.warning(f"P10 features: need >=3 bars for {token.symbol}, got {len(bars)}")
            return None

        n = min(len(bars), 6)
        has_trade_data = any(("trades" in b or "n_trades" in b) for b in bars[:n])
        try:
            opens = np.array([float(b.get("open", b.get("close", 0))) for b in bars[:n]])
            highs = np.array([float(b.get("high", b.get("close", 0))) for b in bars[:n]])
            lows = np.array([float(b.get("low", b.get("close", 0))) for b in bars[:n]])
            closes = np.array([float(b.get("close", 0)) for b in bars[:n]])
            vols = np.array([float(b.get("volume", b.get("volume_usd", 0)) or 0) for b in bars[:n]])
            trades = np.array([float(b.get("trades", b.get("n_trades", 0)) or 0) for b in bars[:n]])
        except (TypeError, ValueError) as e:
            logger.warning(f"P10 features: bar parse error for {token.symbol}: {e}")
            return None

        if closes[0] <= 0 or not np.all(np.isfinite(closes)):
            logger.warning(f"P10 features: invalid closes for {token.symbol}")
            return None

        # Lookback: last 3 bars (or all bars if fewer than 6)
        lb_start = max(0, n - 3)
        lb_closes = closes[lb_start:]
        lb_highs = highs[lb_start:]
        lb_lows = lows[lb_start:]
        lb_vols = vols[lb_start:]
        lb_trades = trades[lb_start:]

        feats = {}

        # === Price action (lookback) ===
        if lb_closes[0] > 0:
            feats["lookback_return"] = lb_closes[-1] / lb_closes[0] - 1
            feats["lookback_high_low"] = (lb_highs.max() - lb_lows.min()) / lb_closes[0]
            feats["lookback_max_close"] = (lb_closes.max() - lb_closes[0]) / lb_closes[0]
            feats["lookback_min_close"] = (lb_closes.min() - lb_closes[0]) / lb_closes[0]
        else:
            feats["lookback_return"] = 0
            feats["lookback_high_low"] = 0
            feats["lookback_max_close"] = 0
            feats["lookback_min_close"] = 0

        feats["last_bar_range"] = (lb_highs[-1] - lb_lows[-1]) / lb_closes[-1] if lb_closes[-1] > 0 else 0

        # === Volume (lookback) ===
        feats["lookback_volume"] = float(lb_vols.sum())
        feats["lookback_avg_volume"] = float(lb_vols.mean())
        feats["lookback_volume_slope"] = float(lb_vols[-1] - lb_vols[0])
        feats["lookback_volume_acceleration"] = float(lb_vols[-1] / max(lb_vols[0], 1))

        # === Trades (lookback) ===
        feats["lookback_trades"] = float(lb_trades.sum())
        feats["lookback_avg_trades"] = float(lb_trades.mean())
        feats["lookback_avg_trade_size"] = float(lb_vols.sum() / max(lb_trades.sum(), 1))

        # === Volatility (1m return stats over lookback) ===
        lb_rets = np.diff(lb_closes) / np.where(lb_closes[:-1] > 0, lb_closes[:-1], 1)
        lb_rets = lb_rets[np.isfinite(lb_rets)]
        feats["lookback_return_std"] = float(lb_rets.std()) if len(lb_rets) > 0 else 0
        feats["lookback_return_max"] = float(lb_rets.max()) if len(lb_rets) > 0 else 0
        feats["lookback_return_min"] = float(lb_rets.min()) if len(lb_rets) > 0 else 0

        # === Cumulative t=0 to entry ===
        if closes[0] > 0:
            feats["cum_return_t0_to_entry"] = closes[-1] / closes[0] - 1
            feats["cum_drawdown_t0_to_entry"] = lows.min() / closes[0] - 1
            feats["cum_max_t0_to_entry"] = highs.max() / closes[0] - 1
        else:
            feats["cum_return_t0_to_entry"] = 0
            feats["cum_drawdown_t0_to_entry"] = 0
            feats["cum_max_t0_to_entry"] = 0
        feats["cum_volume_t0_to_entry"] = float(vols.sum())
        feats["cum_trades_t0_to_entry"] = float(trades.sum())

        if not has_trade_data:
            logger.info(f"M2: {token.symbol} no trade data in bars — "
                        f"trade features set to 0 (model score is log-only in P12)")

        return feats

    def _compute_legacy_features(self, token: GraduatedToken, bars: List[Dict],
                                 has_trade_data: bool) -> Optional[Dict[str, float]]:
        """Original 6-feature extraction for legacy survival model."""

        closes = [float(b["close"]) for b in bars]
        volumes = [float(b.get("volume", 0) or 0) for b in bars]

        if closes[0] == 0:
            logger.warning(f"Zero close price for {token.symbol} bar 0")
            return None

        # 1. 5min_return
        ret_5min = closes[4] / closes[0] - 1

        # 2. 5min_volume_usd
        vol_5min = sum(volumes)

        # 3. 5min_trade_count
        if has_trade_data:
            trade_count = float(sum(b.get("trades", 0) for b in bars))
        else:
            # GMGN kline has no trade count — use scaler mean (neutral fill)
            trade_count = self.scaler.mean_[self.feature_names.index("5min_trade_count")]

        # 4. 5min_buy_sell_ratio
        if has_trade_data:
            total_buys = sum(b.get("buys", 0) for b in bars)
            total_sells = sum(b.get("sells", 0) for b in bars)
            total_trades = total_buys + total_sells
            buy_sell_ratio = total_buys / total_trades if total_trades > 0 else 0.5
        else:
            # GMGN kline has no buy/sell split — use scaler mean (neutral fill)
            buy_sell_ratio = self.scaler.mean_[self.feature_names.index("5min_buy_sell_ratio")]

        # 5. 5min_volatility — std of bar-to-bar pct changes
        pct_changes = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))
                       if closes[i - 1] != 0]
        volatility = float(np.std(pct_changes)) if len(pct_changes) > 1 else 0.0

        # 6. 5min_stale_pct — fraction of bars with zero trades
        if has_trade_data:
            stale_pct = sum(1 for b in bars if b.get("trades", 0) == 0) / len(bars)
        else:
            stale_pct = 0.0  # was all-zero in training

        return {
            "5min_return": ret_5min,
            "5min_volume_usd": vol_5min,
            "5min_trade_count": trade_count,
            "5min_buy_sell_ratio": buy_sell_ratio,
            "5min_volatility": volatility,
            "5min_stale_pct": stale_pct,
        }

    async def _fetch_gmgn_kline(self, token: GraduatedToken,
                               n_bars: int = 5) -> Optional[List[Dict]]:
        """Fetch 1min kline bars from GMGN API (fallback when on-chain kline unavailable)."""
        client = await self._get_client()
        grad_ts = int(token.graduation_time)
        grad_ts_ms = grad_ts * 1000
        window_ms = (n_bars + 1) * 60_000
        params = {
            **self._auth_params(),
            "chain": "sol",
            "address": token.mint_address,
            "resolution": "1m",
            "from": str(grad_ts_ms),
            "to": str(grad_ts_ms + window_ms),
        }
        try:
            resp = await client.get(
                f"{self.base_url}{self.KLINE_PATH}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                logger.warning(f"GMGN kline error for {token.symbol}: {data}")
                return None
            bars = data.get("data", {}).get("list", [])
        except Exception as e:
            logger.error(f"GMGN kline request failed for {token.symbol}: {e}")
            return None

        if len(bars) < n_bars:
            logger.warning(f"GMGN kline: only {len(bars)} bars for {token.symbol}, need {n_bars}")
            return None

        return bars[:n_bars]

    def predict_survival(self, features: Dict[str, float]) -> float:
        """Return model score (P11: P(profit) or predicted_pnl; P10: P(alive)).

        Applies clip_bounds, optional scaler, then model inference.
        """
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            logger.error(f"predict_survival: missing features {missing[:5]}{'...' if len(missing)>5 else ''} "
                         f"(model_version={self.model_version}, expected {len(self.feature_names)} keys)")
            raise KeyError(f"Missing features: {missing}")
        X = np.array([[features[f] for f in self.feature_names]], dtype=float)

        if self.clip_bounds is not None:
            for i, (lo, hi) in enumerate(self.clip_bounds):
                if not np.isnan(X[0, i]):
                    X[0, i] = np.clip(X[0, i], lo, hi)

        # XGBoost handles NaN natively; only fill for legacy/sklearn models that require it
        if self.scaler is not None:
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
            X = self.scaler.transform(X)
        else:
            X = np.where(np.isposinf(X), 1e10, X)
            X = np.where(np.isneginf(X), -1e10, X)

        if self.model_type == "regression":
            return float(self.model.predict(X)[0])
        return float(self.model.predict_proba(X)[0][1])

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# _safe_float_value, _safe_return_value, _safe_ratio_value, _safe_range_pos_value,
# _upper_wick_frac_value, _body_frac_value, _range_pct_value, _tail_mean_value,
# _tail_share_positive_value, _close_anchor_value, _slice_sum_value, _series_slope,
# _max_drawdown_from_path, _build_confirmed_entry_frame, _get_post_grad_anchor_open_live
# moved to ms.features.feature_math (Phase 1 leaf #4).
# Re-imported above via try/except dual-context block.


# compute_confirmed_entry_live_features moved to ms.features.entry_features (Phase 1 leaf #4).
# Re-imported above via try/except dual-context block.


# compute_super_winner_event_core_features moved to ms.features.entry_features (Phase 1 leaf #4).
# Re-imported above via try/except dual-context block.


# _micro_price_return, _micro_price_range, _micro_local_drawdown,
# _micro_close_pos_in_range moved to ms.features.entry_features (Phase 1 leaf #4).
# These are private helpers internal to that module (not re-exported).


# _micro_top_volume_stats, _micro_window_stats moved to ms.features.entry_features
# (Phase 1 leaf #4). Private helpers internal to that module (not re-exported).


# compute_super_winner_event_micro_overlay_features moved to ms.features.entry_features
# (Phase 1 leaf #4). Re-imported above via try/except dual-context block.


# _family_pass_threshold moved to ms.features.entry_features (Phase 1 leaf #4).
# Re-imported above via try/except dual-context block.


def _apply_threshold_pass(value: float, orientation: str, threshold: float) -> Optional[bool]:
    value = _safe_float_value(value)
    threshold = _safe_float_value(threshold)
    if not np.isfinite(value) or not np.isfinite(threshold):
        return None
    if orientation == "high":
        return bool(value >= threshold)
    if orientation == "low":
        return bool(value <= threshold)
    return None


def evaluate_super_winner_event_shadow_view(
    *,
    kline_bars: List[Dict[str, Any]],
    swaps: List[SwapRecord],
    graduation_time: float,
    sol_price_usd: float,
    scan_start_sec: int,
    scan_end_sec: int,
    scan_step_sec: int,
    threshold_map: Dict[str, Dict[str, float]],
    family_feature_map: Dict[str, List[str]],
    event_defs: Dict[str, List[str]],
    arbitration_rule: str = "earliest_then_ab_then_score",
) -> Optional[Dict[str, Any]]:
    if scan_step_sec <= 0 or scan_end_sec < scan_start_sec:
        return None

    delays = list(range(int(scan_start_sec), int(scan_end_sec) + 1, int(scan_step_sec)))
    for delay_sec in delays:
        core = compute_super_winner_event_core_features(
            kline_bars=kline_bars,
            graduation_time=graduation_time,
            delay_sec=delay_sec,
        )
        if core is None:
            continue

        family_state: Dict[str, Any] = {}
        family_pass_flags: Dict[str, bool] = {}
        for family_name, features in family_feature_map.items():
            pass_values: List[bool] = []
            available_n = 0
            total_possible_n = 0
            for feature_name in features:
                meta = threshold_map.get(feature_name)
                if not meta:
                    continue
                total_possible_n += 1
                feature_pass = _apply_threshold_pass(
                    core.get(feature_name, np.nan),
                    str(meta.get("orientation", "")),
                    float(meta.get("threshold", np.nan)),
                )
                if feature_pass is None:
                    continue
                available_n += 1
                pass_values.append(bool(feature_pass))
            pass_frac = float(sum(pass_values) / available_n) if available_n > 0 else np.nan
            family_pass = bool(
                available_n >= _family_pass_threshold(total_possible_n)
                and np.isfinite(pass_frac)
                and pass_frac >= 0.60
            )
            family_state[f"{family_name}__pass_frac_for__super_winner"] = pass_frac
            family_state[f"{family_name}__pass_for__super_winner"] = float(int(family_pass))
            family_pass_flags[family_name] = family_pass

        event_ab = bool(family_pass_flags.get("absorption", False) and family_pass_flags.get("breadth", False))
        event_abp = bool(event_ab and family_pass_flags.get("persistence", False))
        selected_event_name: Optional[str] = None
        if arbitration_rule == "earliest_then_ab_then_score":
            if event_ab:
                selected_event_name = "event_ab"
        else:
            if event_ab:
                selected_event_name = "event_ab"
        if selected_event_name is None and event_abp:
            selected_event_name = "event_abp"
        if selected_event_name is None:
            continue

        event_time_sec = int(float(graduation_time)) + int(delay_sec)
        micro = compute_super_winner_event_micro_overlay_features(
            swaps=swaps,
            event_time_sec=event_time_sec,
            sol_price_usd=sol_price_usd,
        )
        merged = {
            **core,
            **family_state,
            **micro,
            "event_name": selected_event_name,
            "event_is_abp": float(int(selected_event_name == "event_abp")),
            "event_time_sec": float(event_time_sec),
            "event_ab__flag": float(int(event_ab)),
            "event_abp__flag": float(int(event_abp)),
            "arbitration_rule": arbitration_rule,
        }
        return merged

    return None


class ConfirmedContinuationEVShadowModel:
    """Load and score the frozen Phase 6.4 EV3m live-shadow contract."""

    def __init__(self, model_path: str, *, metrics: Optional[Any] = None):
        path = Path(model_path)
        self.guarded = _model_registry.get_or_load(path, metrics=metrics)
        model_data = self.guarded.unwrap()
        self.model = model_data["model"]
        # Patch sklearn version incompatibility: SimpleImputer trained on
        # sklearn 1.7 may lack _fill_dtype when loaded on sklearn 1.8+.
        try:
            from sklearn.impute import SimpleImputer as _SI
            for step_name, step_obj in (getattr(self.model, 'named_steps', {}) or {}).items():
                if isinstance(step_obj, _SI) and not hasattr(step_obj, '_fill_dtype'):
                    step_obj._fill_dtype = step_obj.statistics_.dtype
            # Also check if model itself is a pipeline with imputer
            if hasattr(self.model, 'steps'):
                for _, step_obj in self.model.steps:
                    if isinstance(step_obj, _SI) and not hasattr(step_obj, '_fill_dtype'):
                        step_obj._fill_dtype = step_obj.statistics_.dtype
        except Exception:
            pass
        self.version = str(model_data.get("version", "ev3m_v1"))
        self.policy_name = str(model_data.get("policy_name", "confirmed_continuation_ev3m_v1"))
        self.primary_score_name = str(model_data.get("primary_score_name", "ev3m_xgboost"))
        self.feature_view = str(model_data.get("feature_view", "confirmed_entry_90s"))
        self.entry_delay_sec = int(model_data.get("entry_delay_sec", 90))
        self.feature_names = list(model_data["feature_cols"])
        self.metadata = dict(model_data.get("metadata", {}))
        self.selection_cutoffs = {
            str(k): float(v)
            for k, v in dict(model_data.get("selection_cutoffs", {})).items()
            if v is not None
        }
        logger.info(
            "ConfirmedContinuationEVShadowModel: loaded %s (%d features, bands=%s)",
            path,
            len(self.feature_names),
            ",".join(sorted(self.selection_cutoffs.keys())),
        )

    def predict_score(self, features: Dict[str, float]) -> float:
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise KeyError(f"Missing EV3m shadow features: {missing}")
        X = pd.DataFrame([{f: features.get(f, np.nan) for f in self.feature_names}], columns=self.feature_names)
        return float(self.model.predict_proba(X)[0][1])


class VShapeModel:
    """V-shape T+10m entry model (Phase 8 / 14n+14q).

    Uses 15 features across 3 layers:
      - vf_*: V-shape pattern features from 1m bars (T+0 to T+10m)
      - m10_*: Microstructure features from raw swaps (T+0 to T+10m)
      - sm_*: Smart money features from trader_records

    Entry: T+10m (600s post-graduation)
    Exit: trail_20_10 (activate at +20% peak, sell on 10% drop from peak)
    """

    def __init__(self, model_path: str, *, metrics: Optional[Any] = None):
        path = Path(model_path)
        self.guarded = _model_registry.get_or_load(path, metrics=metrics)
        model_data = self.guarded.unwrap()
        self.model = model_data["model"]
        # Patch sklearn version incompatibility
        try:
            from sklearn.impute import SimpleImputer as _SI
            if hasattr(self.model, 'steps'):
                for _, step_obj in self.model.steps:
                    if isinstance(step_obj, _SI) and not hasattr(step_obj, '_fill_dtype'):
                        step_obj._fill_dtype = step_obj.statistics_.dtype
        except Exception:
            pass
        self.version = str(model_data.get("version", "vshape_v1"))
        self.policy_name = str(model_data.get("policy_name", "vshape_t10m_trail2010"))
        self.feature_names = list(model_data["feature_cols"])
        self.entry_delay_sec = int(model_data.get("entry_delay_sec", 600))
        self.selection_cutoffs = {
            str(k): float(v)
            for k, v in dict(model_data.get("selection_cutoffs", {})).items()
            if v is not None
        }
        logger.info(
            "VShapeModel: loaded %s (%d features, bands=%s)",
            path, len(self.feature_names),
            ",".join(sorted(self.selection_cutoffs.keys())),
        )

    def predict_score(self, features: Dict[str, float]) -> float:
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            logger.warning("VShapeModel: missing features %s, using NaN", missing)
        X = pd.DataFrame(
            [{f: features.get(f, np.nan) for f in self.feature_names}],
            columns=self.feature_names,
        )
        return float(self.model.predict_proba(X)[0][1])


def detect_vshape_live(kline_bars: List[Dict[str, Any]],
                       graduation_time: float,
                       entry_offset_sec: int = 600) -> Optional[Dict[str, float]]:
    """Detect V-shape pattern from on-chain 1m bars at configurable entry offset.

    Args:
        kline_bars: List of bar dicts with keys: open, high, low, close, volume, time
        graduation_time: Unix timestamp of graduation
        entry_offset_sec: Entry decision time offset from graduation (default 600 = T+10m).
                         Pass 300 for T+5m (Phase 15b v1.6 retrain). Other values
                         require corresponding model retrain — do NOT use mid-deploy
                         drift.

    Returns:
        Dict of vf_* features, or None if insufficient data.

    Byte-parity note (Phase 15b 2026-04-26): Signature was previously hardcoded
    to 600. Both training and production MUST pass the same `entry_offset_sec`
    to maintain feature parity. v1.4r calls with 600 (default) — backwards compat.
    """
    if not kline_bars or len(kline_bars) < 3:
        return None

    # Convert to DataFrame-like structure
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

    # Post-graduation bars only
    post = [b for b in bars if b["offset"] >= -30]  # allow 30s tolerance
    if len(post) < 3:
        return None

    grad_price = post[0]["open"]
    if grad_price <= 0:
        return None

    # Pre-entry bars (strictly BEFORE entry_offset_sec — the bar at
    # offset==entry_offset_sec covers [entry_offset_sec, entry_offset_sec+60)
    # which is the future 60s post-entry, so it must NOT contribute to
    # pre-entry features. Fixed 2026-04-29 per F.5 DeepAudit v2.
    pre_entry = [b for b in post if b["offset"] < entry_offset_sec]
    if len(pre_entry) < 3:
        return None

    # Entry price at entry_offset_sec
    entry_bars = [b for b in post if b["offset"] >= entry_offset_sec]
    if not entry_bars:
        return None
    entry_price = entry_bars[0]["open"]
    if entry_price <= 0:
        return None

    closes = [b["close"] / grad_price - 1 for b in pre_entry]
    highs = [b["high"] / grad_price - 1 for b in pre_entry]
    lows = [b["low"] / grad_price - 1 for b in pre_entry]
    volumes = [b["volume"] for b in pre_entry]

    peak_ret = max(highs)
    trough_ret = min(lows)
    peak_idx = highs.index(max(highs))
    trough_idx = lows.index(min(lows))
    entry_ret = entry_price / grad_price - 1
    last_close_ret = closes[-1]

    range_total = peak_ret - trough_ret
    recovery_from_trough = last_close_ret - trough_ret
    recovery_pct = recovery_from_trough / range_total if range_total > 0.01 else 0
    drawdown_from_peak = last_close_ret - peak_ret
    peak_before_trough = peak_idx < trough_idx
    trough_before_peak = trough_idx < peak_idx

    # V-shape pattern: peak THEN dip THEN recovery
    is_vshape = (peak_ret >= 0.10 and trough_ret < peak_ret - 0.10 and
                 recovery_pct >= 0.30 and peak_before_trough)

    # Steady-up pattern: continuously up, no significant drawdown
    is_steady_up = (entry_ret > 0.10 and drawdown_from_peak > -0.10)

    # Near-high pattern: close to peak, moderate drawdown tolerance
    is_near_high = (not is_vshape and not is_steady_up and
                    drawdown_from_peak > -0.20 and entry_ret > 0 and peak_ret >= 0.05)

    # Reversal pattern (2026-04-15 addition): dip FIRST, then strong recovery.
    # Explicitly catches down-then-up tokens that the three classic patterns
    # exclude. Example: token 33KyfKkXVek7 dipped -38% then pumped to +124%
    # (final +100%) — classic winner that used to be missed.
    # Criteria:
    #   - trough occurs BEFORE peak (opposite of V-shape)
    #   - entry_ret > 0.15 (actually up overall, not just recovering to zero)
    #   - recovery_pct > 0.50 (recovered more than half of the range)
    #   - peak_ret >= 0.10 (meaningful upside after dip)
    recovery_from_trough_pct = (
        (entry_ret - trough_ret) / (peak_ret - trough_ret)
        if (peak_ret - trough_ret) > 0.01 else 0
    )
    is_reversal = (not is_vshape and not is_steady_up and not is_near_high and
                   trough_before_peak and
                   entry_ret > 0.15 and
                   recovery_from_trough_pct > 0.50 and
                   peak_ret >= 0.10)

    any_pattern = is_vshape or is_steady_up or is_near_high or is_reversal

    # ─────────────────────── v3.2 NEW PATTERNS (2026-04-29) ───────────────────────
    # Test on 1000-mint sample showed coverage gap: only 40% of decisions matched
    # current 4 patterns; the remaining 60% had 30%+ peak ≥ +50% rate.
    # Adding 3 patterns to lift frequency +23% with quality match. See
    # `reports/Pattern_Expansion_v3_2_Plan_2026-04-29.md`.

    # 5. is_cup_handle — U-shape: trough first, gentle recovery
    #    Independent of other patterns (no cascading); model learns combinations.
    is_cup_handle = 0
    if len(pre_entry) >= 5 and 0 < trough_idx < len(pre_entry) - 2:
        if -0.30 <= trough_ret < -0.05:        # mild dip
            tail_n = max(2, int(len(pre_entry) * 0.3))
            tail_closes = closes[-tail_n:]
            if (tail_closes[-1] - tail_closes[0]) >= 0.10:   # tail uptrending
                if peak_idx >= trough_idx or peak_ret < 0.10:
                    is_cup_handle = 1

    # 6. is_stair_step — multiple small pumps + shallow pullbacks
    is_stair_step = 0
    if len(pre_entry) >= 4:
        n_bumps = 0
        last_high = 0.0
        in_pullback = False
        for i in range(1, len(pre_entry)):
            if not in_pullback:
                if closes[i] > last_high + 0.05:
                    last_high = closes[i]
                    in_pullback = True
            else:
                if closes[i] < last_high - 0.03:
                    in_pullback = False
                elif closes[i] > last_high + 0.05:
                    last_high = closes[i]
                    n_bumps += 1
        if n_bumps >= 2:
            is_stair_step = 1

    # 7. is_volume_acceleration — late-window volume ≥ 2× early-window + price not crashed
    is_volume_acceleration = 0
    if len(pre_entry) >= 6:
        early_vol = sum(volumes[:3])
        late_vol = sum(volumes[-3:])
        if early_vol > 0 and (late_vol / early_vol) >= 2.0:
            if closes[-1] > -0.10:
                is_volume_acceleration = 1

    any_pattern_extended = (any_pattern or is_cup_handle or is_stair_step
                              or is_volume_acceleration)

    # Volume ratio
    if trough_idx < len(pre_entry) - 1 and trough_idx > 0:
        pre_vol = sum(volumes[:trough_idx + 1])
        post_vol = sum(volumes[trough_idx + 1:])
        recovery_vol_ratio = post_vol / pre_vol if pre_vol > 0 else 0
    else:
        recovery_vol_ratio = 0

    # Momentum
    momentum_last3 = closes[-1] - closes[-3] if len(closes) >= 3 else 0

    return {
        "is_vshape": int(is_vshape),
        "is_steady_up": int(is_steady_up),
        "is_near_high": int(is_near_high),
        "is_reversal": int(is_reversal),
        "any_pattern": int(any_pattern),
        # v3.2 new patterns (additive — old callers ignore)
        "is_cup_handle": int(is_cup_handle),
        "is_stair_step": int(is_stair_step),
        "is_volume_acceleration": int(is_volume_acceleration),
        "any_pattern_extended": int(any_pattern_extended),
        "vf_peak_ret": peak_ret,
        "vf_trough_ret": trough_ret,
        "vf_entry_ret": entry_ret,
        "vf_recovery_pct": recovery_pct,
        "vf_drawdown_from_peak": drawdown_from_peak,
        "vf_range_total": range_total,
        "vf_recovery_vol_ratio": recovery_vol_ratio,
        "vf_momentum_last3": momentum_last3,
        "entry_price_10m": entry_price,
        "grad_price": grad_price,
    }


# compute_micro_live, compute_micro_10m_live, compute_micro_live_full moved to
# ms.features.entry_features (Phase 1 leaf #4).
# Re-imported above via try/except dual-context block.


# ──────────────────────────────────────────────
# M3: Gateway Trader (direct REST API)
# ──────────────────────────────────────────────
# GatewayTrader moved to ms.execution.gateway_trader (Phase 1 leaf #6).
# Re-imported above via try/except dual-context block.

# ──────────────────────────────────────────────
# M5: Risk Manager
# ──────────────────────────────────────────────

# RiskManager moved to ms.risk.risk_manager (Phase 1 leaf #7).
# Re-imported above via try/except dual-context block.

# PostgresTelemetrySink moved to ms.telemetry.telemetry_sink (Phase 1 leaf #5).
# Re-imported above via try/except dual-context block.

# ──────────────────────────────────────────────
# Trade Database (SQLite)
# ──────────────────────────────────────────────

# ── TradeDB schema constants (canonical: ms.persistence.schema) ──────────────
# Phase 2b: DDL_* / MIGRATE_* class constants moved to schema.py.
# Dual-context import: production path = controllers.generic.ms.persistence.schema;
# test-channel path = ms.persistence.schema (flat package, controllers/generic on sys.path).
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.persistence import schema as _schema
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.persistence import schema as _schema  # type: ignore[no-redef]

# ── TradeDB connection primitives (canonical: ms.persistence.connection) ─────
# Phase 2c: SqliteConnection extracted here; dual-context import same pattern.
try:  # production
    from controllers.generic.ms.persistence import connection as _connection
except ImportError:  # test channel
    from ms.persistence import connection as _connection  # type: ignore[no-redef]

# ── Resolver cache CRUD (canonical: ms.persistence.resolver_cache) ───────────
# Phase 2d: creator_cap_hit + funder_resolution cache methods moved to
# ResolverCache; TradeDB composes + delegates.  Dual-context import same pattern.
try:  # production
    from controllers.generic.ms.persistence import resolver_cache as _resolver_cache
except ImportError:  # test channel
    from ms.persistence import resolver_cache as _resolver_cache  # type: ignore[no-redef]

# ── Diagnostic log store (canonical: ms.persistence.diagnostic_store) ────────
# Phase 2e: 4 write-only diagnostic log methods moved to DiagnosticStore;
# TradeDB composes + delegates.  Dual-context import same pattern.
try:  # production
    from controllers.generic.ms.persistence import diagnostic_store as _diagnostic_store
except ImportError:  # test channel
    from ms.persistence import diagnostic_store as _diagnostic_store  # type: ignore[no-redef]

# ── Shadow rug-filter eval store (canonical: ms.persistence.shadow_store) ─────
# Phase 2f-A: 18 rug-filter shadow-eval methods moved to ShadowStore;
# TradeDB composes + delegates.  Dual-context import same pattern.
try:  # production
    from controllers.generic.ms.persistence import shadow_store as _shadow_store
except ImportError:  # test channel
    from ms.persistence import shadow_store as _shadow_store  # type: ignore[no-redef]

# ── Operational store — trades/positions/discoveries CRUD ─────────────────────
# Phase 2g-1: trades/positions/discoveries CRUD moved to OperationalStore;
# TradeDB composes + delegates.  Dual-context import same pattern.
try:  # production
    from controllers.generic.ms.persistence import operational_store as _operational_store
except ImportError:  # test channel
    from ms.persistence import operational_store as _operational_store  # type: ignore[no-redef]


class TradeDB:
    """Persist trades and discoveries to SQLite for post-session analysis."""

    # ── Schema constants re-bound from ms.persistence.schema (Phase 2b) ────────
    # All DDL_* / MIGRATE_* strings live in schema.py (module-level).
    # TradeDB re-binds them here so all self.DDL_*/self.MIGRATE_* in __init__
    # and migrations continue to resolve without any code changes downstream.

    # Core tables
    DDL_TRADES = _schema.DDL_TRADES
    DDL_DISCOVERIES = _schema.DDL_DISCOVERIES
    DDL_EVENTS = _schema.DDL_EVENTS
    DDL_LATENCY_EVENTS = _schema.DDL_LATENCY_EVENTS
    DDL_LATENCY_EVENTS_IDX = _schema.DDL_LATENCY_EVENTS_IDX
    DDL_SWAPS = _schema.DDL_SWAPS
    DDL_SWAPS_IDX = _schema.DDL_SWAPS_IDX
    DDL_TRADER_RECORDS = _schema.DDL_TRADER_RECORDS
    DDL_OPEN_POSITIONS = _schema.DDL_OPEN_POSITIONS

    # open_positions migrations
    MIGRATE_OPEN_POSITIONS_V2 = _schema.MIGRATE_OPEN_POSITIONS_V2
    MIGRATE_OPEN_POSITIONS_V3 = _schema.MIGRATE_OPEN_POSITIONS_V3
    MIGRATE_OPEN_POSITIONS_V4 = _schema.MIGRATE_OPEN_POSITIONS_V4
    MIGRATE_OPEN_POSITIONS_V5 = _schema.MIGRATE_OPEN_POSITIONS_V5
    MIGRATE_OPEN_POSITIONS_V6 = _schema.MIGRATE_OPEN_POSITIONS_V6

    # swaps migrations
    MIGRATE_SWAPS_V2 = _schema.MIGRATE_SWAPS_V2
    MIGRATE_SWAPS_V3 = _schema.MIGRATE_SWAPS_V3

    # trades migrations
    MIGRATE_TRADES_V2 = _schema.MIGRATE_TRADES_V2
    MIGRATE_TRADES_V3 = _schema.MIGRATE_TRADES_V3
    MIGRATE_TRADES_V4 = _schema.MIGRATE_TRADES_V4
    MIGRATE_TRADES_V5 = _schema.MIGRATE_TRADES_V5
    MIGRATE_TRADES_V6 = _schema.MIGRATE_TRADES_V6  # T2.2: realized fill slippage

    # latency_events migrations (V2 is the FIRST latency migration — T2.3)
    MIGRATE_LATENCY_EVENTS_V2 = _schema.MIGRATE_LATENCY_EVENTS_V2

    # discoveries / observations migrations
    MIGRATE_DISCOVERIES_V2 = _schema.MIGRATE_DISCOVERIES_V2
    MIGRATE_DISCOVERIES_V3 = _schema.MIGRATE_DISCOVERIES_V3
    MIGRATE_OBSERVATIONS_V2 = _schema.MIGRATE_OBSERVATIONS_V2
    MIGRATE_OBSERVATIONS_V3 = _schema.MIGRATE_OBSERVATIONS_V3
    MIGRATE_OBSERVATIONS_V4 = _schema.MIGRATE_OBSERVATIONS_V4
    MIGRATE_OBSERVATIONS_V5 = _schema.MIGRATE_OBSERVATIONS_V5
    MIGRATE_OBSERVATIONS_V6 = _schema.MIGRATE_OBSERVATIONS_V6
    MIGRATE_OBSERVATIONS_V7 = _schema.MIGRATE_OBSERVATIONS_V7
    MIGRATE_OBSERVATIONS_V8 = _schema.MIGRATE_OBSERVATIONS_V8
    MIGRATE_OBSERVATIONS_V9 = _schema.MIGRATE_OBSERVATIONS_V9
    MIGRATE_OBSERVATIONS_V10 = _schema.MIGRATE_OBSERVATIONS_V10

    # shadow_exit_warn_evals migrations
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V2 = _schema.MIGRATE_SHADOW_EXIT_WARN_EVALS_V2
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V3 = _schema.MIGRATE_SHADOW_EXIT_WARN_EVALS_V3
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V4 = _schema.MIGRATE_SHADOW_EXIT_WARN_EVALS_V4
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V5 = _schema.MIGRATE_SHADOW_EXIT_WARN_EVALS_V5
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V6 = _schema.MIGRATE_SHADOW_EXIT_WARN_EVALS_V6

    # v5.x shadow exit eval tables
    DDL_SHADOW_V5_3_EVALS = _schema.DDL_SHADOW_V5_3_EVALS
    DDL_SHADOW_V5_3_EVALS_IDX = _schema.DDL_SHADOW_V5_3_EVALS_IDX
    DDL_SHADOW_V5_5_EVALS = _schema.DDL_SHADOW_V5_5_EVALS
    DDL_SHADOW_V5_5_EVALS_IDX = _schema.DDL_SHADOW_V5_5_EVALS_IDX
    DDL_SHADOW_V5_5_3_EVALS = _schema.DDL_SHADOW_V5_5_3_EVALS
    DDL_SHADOW_V5_5_3_EVALS_IDX = _schema.DDL_SHADOW_V5_5_3_EVALS_IDX
    DDL_SHADOW_V5_5_6_EVALS = _schema.DDL_SHADOW_V5_5_6_EVALS
    DDL_SHADOW_V5_5_6_EVALS_IDX = _schema.DDL_SHADOW_V5_5_6_EVALS_IDX
    DDL_SHADOW_V5_6_EVALS = _schema.DDL_SHADOW_V5_6_EVALS
    DDL_SHADOW_V5_6_EVALS_IDX = _schema.DDL_SHADOW_V5_6_EVALS_IDX
    DDL_SHADOW_CORROBORATION_EVALS = _schema.DDL_SHADOW_CORROBORATION_EVALS
    DDL_SHADOW_CORROBORATION_EVALS_IDX = _schema.DDL_SHADOW_CORROBORATION_EVALS_IDX
    MIGRATE_SHADOW_CORROBORATION_EVALS_V2 = _schema.MIGRATE_SHADOW_CORROBORATION_EVALS_V2
    DDL_SHADOW_EXIT_DECISIONS = _schema.DDL_SHADOW_EXIT_DECISIONS
    DDL_SHADOW_EXIT_DECISIONS_IDX = _schema.DDL_SHADOW_EXIT_DECISIONS_IDX
    DDL_SHADOW_V5_5_7_V2_1_EVALS = _schema.DDL_SHADOW_V5_5_7_V2_1_EVALS
    DDL_SHADOW_V5_5_7_V2_1_EVALS_IDX = _schema.DDL_SHADOW_V5_5_7_V2_1_EVALS_IDX
    DDL_SHADOW_V5_3_1_EVALS = _schema.DDL_SHADOW_V5_3_1_EVALS
    DDL_SHADOW_V5_3_1_EVALS_IDX = _schema.DDL_SHADOW_V5_3_1_EVALS_IDX

    # observation / hot_rank tables
    DDL_TOKEN_OBSERVATIONS = _schema.DDL_TOKEN_OBSERVATIONS
    DDL_TOKEN_OBSERVATIONS_IDX = _schema.DDL_TOKEN_OBSERVATIONS_IDX
    DDL_HOT_RANK_OBSERVATIONS = _schema.DDL_HOT_RANK_OBSERVATIONS
    DDL_HOT_RANK_IDX = _schema.DDL_HOT_RANK_IDX

    # policy / rug eval tables
    DDL_SHADOW_POLICY_EVALS = _schema.DDL_SHADOW_POLICY_EVALS
    DDL_SHADOW_POLICY_EVALS_IDX = _schema.DDL_SHADOW_POLICY_EVALS_IDX
    DDL_SHADOW_RUG_EVALS = _schema.DDL_SHADOW_RUG_EVALS
    DDL_SHADOW_RUG_EVALS_IDX = _schema.DDL_SHADOW_RUG_EVALS_IDX
    DDL_SHADOW_RUG_V4_EVALS = _schema.DDL_SHADOW_RUG_V4_EVALS
    DDL_SHADOW_RUG_V4_EVALS_IDX = _schema.DDL_SHADOW_RUG_V4_EVALS_IDX
    DDL_SHADOW_RUG_V4_2_EVALS = _schema.DDL_SHADOW_RUG_V4_2_EVALS
    DDL_SHADOW_RUG_V4_2_EVALS_IDX = _schema.DDL_SHADOW_RUG_V4_2_EVALS_IDX
    DDL_SHADOW_RUG_V4_3_EVALS = _schema.DDL_SHADOW_RUG_V4_3_EVALS
    DDL_SHADOW_RUG_V4_3_EVALS_IDX = _schema.DDL_SHADOW_RUG_V4_3_EVALS_IDX
    DDL_SHADOW_RUG_V4_4_EVALS = _schema.DDL_SHADOW_RUG_V4_4_EVALS
    DDL_SHADOW_RUG_V4_4_EVALS_IDX = _schema.DDL_SHADOW_RUG_V4_4_EVALS_IDX
    DDL_SHADOW_RUG_V3B_EVALS = _schema.DDL_SHADOW_RUG_V3B_EVALS
    DDL_SHADOW_RUG_V3B_EVALS_IDX = _schema.DDL_SHADOW_RUG_V3B_EVALS_IDX

    # resolver caches
    DDL_CREATOR_CAP_HIT_CACHE = _schema.DDL_CREATOR_CAP_HIT_CACHE
    DDL_CREATOR_CAP_HIT_CACHE_IDX = _schema.DDL_CREATOR_CAP_HIT_CACHE_IDX
    DDL_FUNDER_RESOLUTION_CACHE = _schema.DDL_FUNDER_RESOLUTION_CACHE
    DDL_FUNDER_RESOLUTION_CACHE_IDX = _schema.DDL_FUNDER_RESOLUTION_CACHE_IDX

    # funder-graph shadow
    DDL_SHADOW_RUG_FUNDER_V1_EVALS = _schema.DDL_SHADOW_RUG_FUNDER_V1_EVALS
    DDL_SHADOW_RUG_FUNDER_V1_EVALS_IDX = _schema.DDL_SHADOW_RUG_FUNDER_V1_EVALS_IDX

    # VWMP + cleaning-service diagnostic logs
    DDL_VWMP_SHADOW_LOG = _schema.DDL_VWMP_SHADOW_LOG
    DDL_VWMP_SHADOW_LOG_IDX = _schema.DDL_VWMP_SHADOW_LOG_IDX
    DDL_CS_PRICE_DIVERGENCE_LOG = _schema.DDL_CS_PRICE_DIVERGENCE_LOG
    DDL_CS_PRICE_DIVERGENCE_LOG_IDX = _schema.DDL_CS_PRICE_DIVERGENCE_LOG_IDX
    DDL_CS_PEAK_PHANTOM_EVENTS_LOG = _schema.DDL_CS_PEAK_PHANTOM_EVENTS_LOG
    DDL_CS_PEAK_PHANTOM_EVENTS_LOG_IDX = _schema.DDL_CS_PEAK_PHANTOM_EVENTS_LOG_IDX
    DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG = _schema.DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG
    DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG_IDX = _schema.DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG_IDX

    # event / preflight / rug event tables
    DDL_SHADOW_EVENT_EVALS = _schema.DDL_SHADOW_EVENT_EVALS
    DDL_SHADOW_EVENT_EVALS_IDX = _schema.DDL_SHADOW_EVENT_EVALS_IDX
    DDL_PREFLIGHT_CHECKS = _schema.DDL_PREFLIGHT_CHECKS
    DDL_PREFLIGHT_CHECKS_IDX = _schema.DDL_PREFLIGHT_CHECKS_IDX
    DDL_RUG_EVENTS = _schema.DDL_RUG_EVENTS
    DDL_RUG_EVENTS_IDX = _schema.DDL_RUG_EVENTS_IDX
    DDL_SHADOW_EVENT_INVARIANT_EVALS = _schema.DDL_SHADOW_EVENT_INVARIANT_EVALS
    DDL_SHADOW_EVENT_INVARIANT_EVALS_IDX = _schema.DDL_SHADOW_EVENT_INVARIANT_EVALS_IDX

    # big_winner / vshape / price-probe tables
    DDL_SHADOW_BIG_WINNER_EVALS = _schema.DDL_SHADOW_BIG_WINNER_EVALS
    DDL_SHADOW_BIG_WINNER_EVALS_IDX = _schema.DDL_SHADOW_BIG_WINNER_EVALS_IDX
    DDL_SHADOW_VSHAPE_V3_4_EVALS = _schema.DDL_SHADOW_VSHAPE_V3_4_EVALS
    DDL_SHADOW_VSHAPE_V3_4_EVALS_IDX = _schema.DDL_SHADOW_VSHAPE_V3_4_EVALS_IDX
    DDL_PRICE_PROBES = _schema.DDL_PRICE_PROBES
    DDL_PRICE_PROBES_IDX = _schema.DDL_PRICE_PROBES_IDX

    # exit-warn / sw_v2 shadow tables
    DDL_SHADOW_EXIT_WARN_EVALS = _schema.DDL_SHADOW_EXIT_WARN_EVALS
    DDL_SHADOW_EXIT_WARN_EVALS_IDX = _schema.DDL_SHADOW_EXIT_WARN_EVALS_IDX
    DDL_SHADOW_SW_V2_EVALS = _schema.DDL_SHADOW_SW_V2_EVALS
    DDL_SHADOW_SW_V2_EVALS_IDX = _schema.DDL_SHADOW_SW_V2_EVALS_IDX

    # pre-grad bonding-curve factor shadow (L1 universe rug-screen)
    DDL_SHADOW_PREGRAD_FACTORS_EVALS = _schema.DDL_SHADOW_PREGRAD_FACTORS_EVALS
    DDL_SHADOW_PREGRAD_FACTORS_EVALS_IDX = _schema.DDL_SHADOW_PREGRAD_FACTORS_EVALS_IDX
    DDL_SHADOW_XOSCROSS_ARM_EVALS = _schema.DDL_SHADOW_XOSCROSS_ARM_EVALS
    DDL_SHADOW_XOSCROSS_ARM_EVALS_IDX = _schema.DDL_SHADOW_XOSCROSS_ARM_EVALS_IDX
    MIGRATE_SHADOW_XOSCROSS_V2 = _schema.MIGRATE_SHADOW_XOSCROSS_V2

    # T1.3 schema drift guard meta-table
    DDL_SCHEMA_META = _schema.DDL_SCHEMA_META


    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(
                os.environ.get("PROJECT_DIR", "/home/hummingbot"),
                "data", "meme_sniper_trades.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path  # exposed for read-only consumers (e.g. rug_filter_v4)
        # sqlite3 with check_same_thread=False requires explicit user-side
        # serialization. Without a lock, concurrent controller callbacks can
        # interleave writes on the shared connection and destabilize the file.
        # Phase 2c: connection + lock owned by SqliteConnection; self._conn and
        # self._lock are aliases to the same objects so all downstream code
        # (PRAGMA block, DDL, migrations, 80+ methods) continues unchanged.
        self._db = _connection.SqliteConnection(db_path)
        self._conn = self._db.conn
        self._lock = self._db.lock
        self._resolver_cache = _resolver_cache.ResolverCache(self._db)
        self._diagnostic_store = _diagnostic_store.DiagnosticStore(self._db)
        self._shadow_store = _shadow_store.ShadowStore(self._db)
        self._operational_store = _operational_store.OperationalStore(self._db)
        # WAL mode corrupts repeatedly under Docker overlayfs (3 incidents in 2 days).
        # Switch to DELETE journal mode: slower writes but no WAL/SHM files that
        # overlayfs can desync. Combined with synchronous=FULL for maximum safety.
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=DELETE")
            # T1.2 — boot journal_mode assertion (warn-only; §3.A.6 / §8.2).
            # Read back the CURRENT mode: the SET above silently no-ops when other
            # connections hold the DB (the WAL-1 condition: 2026-06-02 outage).
            # MUST be warn-only per §8.3 — a hard-fail here = self-inflicted outage.
            # record_event NOT called here: the events table (DDL_EVENTS) has not
            # been created yet at this point in the ctor, so record_event would fail.
            # Telegram routing deferred to a future T3.y structured-logging item.
            try:
                _jm_row = self._conn.execute("PRAGMA journal_mode").fetchone()
                _jm = str(_jm_row[0]).lower() if _jm_row else "unknown"
                if _jm != "delete":
                    logger.warning(
                        f"⚠️ DB journal_mode={_jm!r} (expected 'delete') — WAL may be"
                        f" re-creeping; risk of lock-storm/outage (see 2026-06-02)."
                        f" DB={db_path}"
                    )
            except Exception as _jm_err:
                logger.warning(f"TradeDB journal_mode check failed (non-fatal): {_jm_err}")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            # Auto-fix index corruption on startup (caused by SIGTERM during writes).
            # REINDEX is cheap and prevents "wrong # of entries in index" errors.
            try:
                self._conn.execute("REINDEX")
            except Exception:
                pass
            self._conn.execute(self.DDL_TRADES)
            self._conn.execute(self.DDL_DISCOVERIES)
            self._conn.execute(self.DDL_EVENTS)
            self._conn.execute(self.DDL_LATENCY_EVENTS)
            for idx_stmt in self.DDL_LATENCY_EVENTS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_OPEN_POSITIONS)
            self._conn.execute(self.DDL_SWAPS)
            for idx_stmt in self.DDL_SWAPS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_TRADER_RECORDS)
            self._conn.execute(self.DDL_TOKEN_OBSERVATIONS)
            for idx_stmt in self.DDL_TOKEN_OBSERVATIONS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_HOT_RANK_OBSERVATIONS)
            for idx_stmt in self.DDL_HOT_RANK_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_POLICY_EVALS)
            for idx_stmt in self.DDL_SHADOW_POLICY_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_PRICE_PROBES)
            for idx_stmt in self.DDL_PRICE_PROBES_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_EVENT_INVARIANT_EVALS)
            for idx_stmt in self.DDL_SHADOW_EVENT_INVARIANT_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_RUG_EVENTS)
            for idx_stmt in self.DDL_RUG_EVENTS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_PREFLIGHT_CHECKS)
            for idx_stmt in self.DDL_PREFLIGHT_CHECKS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_EVENT_EVALS)
            for idx_stmt in self.DDL_SHADOW_EVENT_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_BIG_WINNER_EVALS)
            for idx_stmt in self.DDL_SHADOW_BIG_WINNER_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_VSHAPE_V3_4_EVALS)
            for idx_stmt in self.DDL_SHADOW_VSHAPE_V3_4_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_EXIT_WARN_EVALS)
            for idx_stmt in self.DDL_SHADOW_EXIT_WARN_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_SW_V2_EVALS)
            for idx_stmt in self.DDL_SHADOW_SW_V2_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_PREGRAD_FACTORS_EVALS)
            for idx_stmt in self.DDL_SHADOW_PREGRAD_FACTORS_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_XOSCROSS_ARM_EVALS)
            for idx_stmt in self.DDL_SHADOW_XOSCROSS_ARM_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V4_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V4_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V4_2_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V4_2_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V4_3_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V4_3_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V4_4_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V4_4_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V3B_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V3B_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_CREATOR_CAP_HIT_CACHE)
            for idx_stmt in self.DDL_CREATOR_CAP_HIT_CACHE_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_FUNDER_RESOLUTION_CACHE)
            for idx_stmt in self.DDL_FUNDER_RESOLUTION_CACHE_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_FUNDER_V1_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_FUNDER_V1_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_VWMP_SHADOW_LOG)
            for idx_stmt in self.DDL_VWMP_SHADOW_LOG_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_CS_PRICE_DIVERGENCE_LOG)
            for idx_stmt in self.DDL_CS_PRICE_DIVERGENCE_LOG_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_CS_PEAK_PHANTOM_EVENTS_LOG)
            for idx_stmt in self.DDL_CS_PEAK_PHANTOM_EVENTS_LOG_IDX:
                self._conn.execute(idx_stmt)
            # Phase B.4
            self._conn.execute(self.DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG)
            for idx_stmt in self.DDL_CS_V5_5_INFERENCE_DIVERGENCE_LOG_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_3_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_3_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_5_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_5_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_5_3_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_5_3_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_5_6_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_5_6_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_5_7_V2_1_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_5_7_V2_1_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_6_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_6_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_CORROBORATION_EVALS)
            for idx_stmt in self.DDL_SHADOW_CORROBORATION_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_V5_3_1_EVALS)
            for idx_stmt in self.DDL_SHADOW_V5_3_1_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_EXIT_DECISIONS)
            for idx_stmt in self.DDL_SHADOW_EXIT_DECISIONS_IDX:
                self._conn.execute(idx_stmt)
            self._migrate_open_positions()
            self._migrate_table("trades", self.MIGRATE_TRADES_V2)
            self._migrate_table("trades", self.MIGRATE_TRADES_V3)
            self._migrate_table("trades", self.MIGRATE_TRADES_V4)
            self._migrate_table("trades", self.MIGRATE_TRADES_V5)
            self._migrate_table("trades", self.MIGRATE_TRADES_V6)
            self._migrate_table("discoveries", self.MIGRATE_DISCOVERIES_V2)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V2)
            self._migrate_table("swaps", self.MIGRATE_SWAPS_V2)
            self._migrate_table("swaps", self.MIGRATE_SWAPS_V3)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V3)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V4)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V5)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V6)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V7)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V8)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V9)
            # T2.7: split embedded-value reject_reason into structured cols (FE-1).
            self._migrate_table("discoveries", self.MIGRATE_DISCOVERIES_V3)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V10)
            self._migrate_table("shadow_corroboration_evals", self.MIGRATE_SHADOW_CORROBORATION_EVALS_V2)
            # 2026-07-06 修复②:xoscross live 臂终判归因列
            self._migrate_table("shadow_xoscross_arm_evals", self.MIGRATE_SHADOW_XOSCROSS_V2)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V2)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V3)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V4)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V5)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V6)
            # T2.3 latency_events V2 — add error_category TEXT (FIRST latency migration).
            self._migrate_table("latency_events", self.MIGRATE_LATENCY_EVENTS_V2)
            # T1.3 schema drift guard — create meta-table and record version on change.
            # Placed AFTER all _migrate_table calls so the version stamp is only written
            # when migrations have actually been attempted.  The version INSERT is further
            # gated on gmgn_collection_meta being present in token_observations: if
            # MIGRATE_OBSERVATIONS_V9 silently failed (ALTER logged but swallowed), we
            # defer the stamp rather than falsely claiming the schema is at SCHEMA_VERSION.
            # Migrations re-run every boot and self-heal, so deferring is safe + truthful.
            try:
                self._conn.execute(self.DDL_SCHEMA_META)
                row = self._conn.execute(
                    "SELECT MAX(version) FROM schema_meta"
                ).fetchone()
                current_ver = row[0] if row and row[0] is not None else None
                if current_ver != _schema.SCHEMA_VERSION:
                    import time as _time
                    # Gate: only stamp if gmgn_collection_meta column is actually present.
                    _obs_cols = [
                        r[1] for r in self._conn.execute(
                            "PRAGMA table_info(token_observations)"
                        ).fetchall()
                    ]
                    if "gmgn_collection_meta" in _obs_cols:
                        self._conn.execute(
                            "INSERT INTO schema_meta(version, applied_at, note) VALUES(?, ?, ?)",
                            (_schema.SCHEMA_VERSION, _time.time(),
                             f"boot v{_schema.SCHEMA_VERSION}"),
                        )
                    else:
                        logger.warning(
                            "schema_meta: deferring version stamp — gmgn_collection_meta"
                            " column absent (migration may have failed; will retry next boot)"
                        )
            except Exception as _e:
                logger.warning(f"TradeDB schema_meta update skipped: {_e}")
            self._conn.commit()
        self._telemetry_sink: Optional[PostgresTelemetrySink] = None
        telemetry_dsn = os.environ.get("TELEMETRY_DATABASE_URL", "").strip()
        if telemetry_dsn:
            try:
                self._telemetry_sink = PostgresTelemetrySink(telemetry_dsn)
                logger.info("TradeDB telemetry sink enabled: PostgreSQL")
            except Exception as e:
                logger.warning(f"TradeDB telemetry sink unavailable: {e}")
        # Share the telemetry sink with ShadowStore so record_shadow_policy_eval
        # mirrors to PostgreSQL identically to the pre-2f TradeDB. ShadowStore is
        # constructed earlier (in the store-init block) before the sink exists, so
        # propagate the same object here after sink creation (None or live sink).
        self._shadow_store._telemetry_sink = self._telemetry_sink
        self._operational_store._telemetry_sink = self._telemetry_sink
        logger.info(f"TradeDB initialized: {db_path}")

    def _fetchall(self, sql: str, params: tuple = ()) -> list:
        return self._db.fetchall(sql, params)

    def _fetchone(self, sql: str, params: tuple = ()):
        return self._db.fetchone(sql, params)

    def _execute_commit(self, sql: str, params: tuple = ()):
        return self._db.execute_commit(sql, params)

    def _executemany_commit(self, sql: str, seq_of_params):
        return self._db.executemany_commit(sql, seq_of_params)

    def _migrate_open_positions(self):
        """Add trailing stop columns (v2), pool_address/source (v3),
        entry_source (v4), peak_pnl_pct_poll (v5), sl_first_breach_ts (v6)
        if missing."""
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V2)
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V3)
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V4)
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V5)
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V6)

    def _migrate_table(self, table: str, stmts: list):
        """Generic migration: add columns and/or auxiliary DDL (indices, etc.) if missing.

        Phase 16.3 Step 3 audit fix: failures used to silently `pass`,
        which caused INSERT to later reference missing columns and lose
        rows. Now we log loudly so on-call sees the issue, but we don't
        crash startup (the table may be locked transiently).

        Phase 22.D Route X bugfix (2026-05-01): handle non-ADD-COLUMN
        statements (e.g. CREATE INDEX IF NOT EXISTS). Previous parser
        unconditionally split on "ADD COLUMN " which raised IndexError
        on any other DDL, killing the entire migration list mid-loop.
        """
        cols = {row[1] for row in self._fetchall(f"PRAGMA table_info({table})")}
        for stmt in stmts:
            if "ADD COLUMN" in stmt:
                col_name = stmt.split("ADD COLUMN ")[1].split()[0]
                if col_name in cols:
                    continue
                desc = f"{table}.{col_name}"
            else:
                # Auxiliary DDL (CREATE INDEX, etc.) — idempotent if it
                # uses IF NOT EXISTS. We still execute under lock and
                # log, but don't gate on column-existence.
                desc = stmt[:80]
            try:
                with self._lock:
                    self._conn.execute(stmt)
                    logger.info(f"TradeDB migration: {desc} applied")
            except Exception as e:
                logger.error(
                    f"TradeDB migration FAILED: {desc}: {e} "
                    f"— INSERTs/queries referencing this may fail. "
                    f"Investigate immediately.")

    def record_trade(self, record: "TradeRecord", features: Optional[Dict] = None,
                     entry_source: str = ""):
        return self._operational_store.record_trade(record, features, entry_source)

    def record_hot_rank_observation(self, mint: str, symbol: str, age_sec: int,
                                     liquidity_usd: float, volume_5m: float,
                                     swaps_5m: int, smart_degen_count: int,
                                     rank_position: int, match_status: str,
                                     m1_first_seen: Optional[float] = None,
                                     m1_passed: Optional[bool] = None):
        return self._operational_store.record_hot_rank_observation(mint, symbol, age_sec, liquidity_usd, volume_5m, swaps_5m, smart_degen_count, rank_position, match_status, m1_first_seen, m1_passed)

    def lookup_discovery(self, mint: str) -> Optional[Tuple[float, int]]:
        return self._operational_store.lookup_discovery(mint)

    def record_discovery(self, token: "GraduatedToken", p_alive: Optional[float],
                         passed: bool, reject_reason: str = "",
                         features: Optional[Dict] = None):
        return self._operational_store.record_discovery(token, p_alive, passed, reject_reason, features)

    def save_swaps(self, mint_address: str, pool_address: str,
                   swaps: List["SwapRecord"]):
        return self._operational_store.save_swaps(mint_address, pool_address, swaps)

    def record_observation(self, token: "GraduatedToken",
                           model_score: Optional[float], model_passed: Optional[bool],
                           reject_reason: str = "",
                           features: Optional[Dict] = None,
                           kline_6m: Optional[List[Dict]] = None,
                           gmgn_info_entry: Optional[Dict] = None,
                           gmgn_collection_meta: Optional[Dict] = None) -> int:
        return self._operational_store.record_observation(token, model_score, model_passed, reject_reason, features, kline_6m, gmgn_info_entry, gmgn_collection_meta)

    def complete_observation(self, obs_id: int,
                             kline_30m: Optional[List[Dict]] = None,
                             trade_pnl_usd: Optional[float] = None,
                             trade_exit_reason: Optional[str] = None,
                             gmgn_info_60m: Optional[Dict] = None,
                             gmgn_security: Optional[Dict] = None,
                             gmgn_collection_meta: Optional[Dict] = None):
        return self._operational_store.complete_observation(obs_id, kline_30m, trade_pnl_usd, trade_exit_reason, gmgn_info_60m, gmgn_security, gmgn_collection_meta)

    def update_observation_snapshot(self, obs_id: int, column: str, data: Dict,
                                     gmgn_collection_meta: Optional[Dict] = None):
        return self._operational_store.update_observation_snapshot(obs_id, column, data, gmgn_collection_meta)

    # ── Trader Record Management ──────────────────────────────────────

    def update_trader_records(self, mint_address: str, is_winner: bool):
        return self._operational_store.update_trader_records(mint_address, is_winner)

    def get_trader_record(self, trader_address: str) -> Optional[Dict]:
        return self._operational_store.get_trader_record(trader_address)

    def get_trader_records_batch(self, addresses: List[str]) -> Dict[str, Dict]:
        return self._operational_store.get_trader_records_batch(addresses)

    def get_trader_records_count(self) -> int:
        return self._operational_store.get_trader_records_count()

    def compute_smart_money_features(self, mint_address: str,
                                      min_history: int = 3,
                                      smart_threshold: float = 0.5) -> Dict[str, float]:
        return self._operational_store.compute_smart_money_features(mint_address, min_history, smart_threshold)

    def get_observation_gmgn_t0(self, mint_address: str) -> Optional[Dict]:
        return self._operational_store.get_observation_gmgn_t0(mint_address)

    def get_swaps_for_token(self, mint_address: str) -> List[Dict]:
        return self._operational_store.get_swaps_for_token(mint_address)

    def get_swaps_df_for_token(self, mint_address: str):
        return self._operational_store.get_swaps_df_for_token(mint_address)

    def load_orphan_observations(self, buffer_sec: float = 62 * 60) -> List[Dict]:
        return self._operational_store.load_orphan_observations(buffer_sec)

    def load_pending_observations(self) -> List[Dict]:
        return self._operational_store.load_pending_observations()

    def record_event(self, level: str, module: str, message: str):
        return self._operational_store.record_event(level, module, message)

    def record_latency_event(self, mint_address: str, symbol: str,
                             graduation_time: Optional[float], event_name: str,
                             age_sec: Optional[float] = None,
                             metadata: Optional[Dict] = None,
                             error_category: Optional[str] = None):
        return self._operational_store.record_latency_event(
            mint_address, symbol, graduation_time, event_name, age_sec, metadata,
            error_category=error_category,
        )

    def get_latest_shadow_rug_eval(self, mint_address: str) -> Optional[Dict]:
        return self._shadow_store.get_latest_shadow_rug_eval(mint_address)

    def record_shadow_rug_eval(self, *, mint_address: str, symbol: Optional[str],
                                graduation_time: Optional[float],
                                snapshot_delay_sec: Optional[float],
                                score: float, cutoff_band: str,
                                cutoff_value: float, would_reject: bool,
                                model_version: Optional[str],
                                features: Optional[Dict]):
        return self._shadow_store.record_shadow_rug_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, snapshot_delay_sec=snapshot_delay_sec, score=score, cutoff_band=cutoff_band, cutoff_value=cutoff_value, would_reject=would_reject, model_version=model_version, features=features)

    def has_shadow_rug_v4_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_v4_eval(mint_address)

    def get_shadow_rug_v4_decision(self, mint_address: str) -> Optional[Tuple[str, Optional[int], Optional[float]]]:
        return self._shadow_store.get_shadow_rug_v4_decision(mint_address)

    def record_shadow_rug_v4_eval(self, *, mint_address: str, symbol: Optional[str],
                                   graduation_time: Optional[float],
                                   scored_at_delay_s: Optional[float],
                                   window_s: int, n_swaps: int,
                                   r1_flag: int, r4_proba: Optional[float],
                                   cutoff: float,
                                   decision: str, reason: str,
                                   features: Optional[Dict],
                                   model_version: Optional[str]):
        return self._shadow_store.record_shadow_rug_v4_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, window_s=window_s, n_swaps=n_swaps, r1_flag=r1_flag, r4_proba=r4_proba, cutoff=cutoff, decision=decision, reason=reason, features=features, model_version=model_version)

    # ===== v4.2 v0.9i hybrid shadow methods =====
    def has_shadow_rug_v4_2_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_v4_2_eval(mint_address)

    def record_shadow_rug_v4_2_eval(self, *, mint_address: str, symbol: Optional[str],
                                      graduation_time: Optional[float],
                                      scored_at_delay_s: Optional[float],
                                      window_s: int, n_swaps: int,
                                      score: Optional[float],
                                      cutoff: float,
                                      decision: str, reason: str,
                                      features: Optional[Dict],
                                      model_version: Optional[str]):
        return self._shadow_store.record_shadow_rug_v4_2_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, window_s=window_s, n_swaps=n_swaps, score=score, cutoff=cutoff, decision=decision, reason=reason, features=features, model_version=model_version)

    # ===== v4.3 cross-source-aligned shadow methods =====
    def has_shadow_rug_v4_3_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_v4_3_eval(mint_address)

    def get_latest_shadow_rug_v4_3_eval(self, mint_address: str) -> Optional[Dict[str, Any]]:
        return self._shadow_store.get_latest_shadow_rug_v4_3_eval(mint_address)

    def record_shadow_rug_v4_3_eval(self, *, mint_address: str, symbol: Optional[str],
                                      graduation_time: Optional[float],
                                      scored_at_delay_s: Optional[float],
                                      window_s: int, n_swaps: int,
                                      score: Optional[float],
                                      cutoff: float,
                                      decision: str, reason: str,
                                      features: Optional[Dict],
                                      model_version: Optional[str]):
        return self._shadow_store.record_shadow_rug_v4_3_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, window_s=window_s, n_swaps=n_swaps, score=score, cutoff=cutoff, decision=decision, reason=reason, features=features, model_version=model_version)

    # ───────────────────────────────────────────────────────────────────
    # v4.4 cs L1+L2 cleaning A/B vs v4.3 (2026-05-22, spec v0.6)
    # Schema mirrors v4.3 1:1. Separate table for clean A/B comparison.
    # ───────────────────────────────────────────────────────────────────
    def has_shadow_rug_v4_4_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_v4_4_eval(mint_address)

    def get_latest_shadow_rug_v4_4_eval(self, mint_address: str) -> Optional[Dict[str, Any]]:
        return self._shadow_store.get_latest_shadow_rug_v4_4_eval(mint_address)

    def record_shadow_rug_v4_4_eval(self, *, mint_address: str, symbol: Optional[str],
                                      graduation_time: Optional[float],
                                      scored_at_delay_s: Optional[float],
                                      window_s: int, n_swaps: int,
                                      score: Optional[float],
                                      cutoff: float,
                                      decision: str, reason: str,
                                      features: Optional[Dict],
                                      model_version: Optional[str]):
        return self._shadow_store.record_shadow_rug_v4_4_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, window_s=window_s, n_swaps=n_swaps, score=score, cutoff=cutoff, decision=decision, reason=reason, features=features, model_version=model_version)

    # ───────────────────────────────────────────────────────────────────
    # v3b Tier-1 shadow methods (2026-05-17)
    # ───────────────────────────────────────────────────────────────────
    def has_shadow_rug_v3b_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_v3b_eval(mint_address)

    def get_latest_shadow_rug_v3b_eval(self, mint_address: str) -> Optional[Dict[str, Any]]:
        return self._shadow_store.get_latest_shadow_rug_v3b_eval(mint_address)

    def record_shadow_rug_v3b_eval(self, *, mint_address: str,
                                     symbol: Optional[str],
                                     graduation_time: Optional[float],
                                     scored_at_delay_s: Optional[float],
                                     F0: Optional[int], F7: Optional[int],
                                     cap_hit: Optional[bool],
                                     score: Optional[float],
                                     decision: str, reason: str,
                                     n_swaps: int,
                                     mode: str,
                                     model_version: Optional[str]):
        return self._shadow_store.record_shadow_rug_v3b_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, F0=F0, F7=F7, cap_hit=cap_hit, score=score, decision=decision, reason=reason, n_swaps=n_swaps, mode=mode, model_version=model_version)

    # ───────────────────────────────────────────────────────────────────
    # Resolver cache — Phase 2d: methods + constants moved to
    # ms.persistence.resolver_cache.ResolverCache; TradeDB composes +
    # delegates.  Signatures are verbatim (keyword-only params preserved).
    # ───────────────────────────────────────────────────────────────────

    # FUNDER_STATUS_* re-bound as TradeDB class attributes so all external
    # callers (rug_filter_funder_v1.py: db.FUNDER_STATUS_* and
    # TradeDB.FUNDER_STATUS_*) continue to resolve unchanged.
    FUNDER_STATUS_NOT_RESOLVED = _resolver_cache.ResolverCache.FUNDER_STATUS_NOT_RESOLVED
    FUNDER_STATUS_NO_TRANSFER  = _resolver_cache.ResolverCache.FUNDER_STATUS_NO_TRANSFER
    FUNDER_STATUS_ERROR        = _resolver_cache.ResolverCache.FUNDER_STATUS_ERROR
    FUNDER_STATUS_FOUND        = _resolver_cache.ResolverCache.FUNDER_STATUS_FOUND

    def has_creator_resolved(self, mint_address: str) -> bool:
        return self._resolver_cache.has_creator_resolved(mint_address)

    def record_creator_resolution(self, *, mint_address: str,
                                    creator: Optional[str],
                                    creation_ts: Optional[float],
                                    n_sig_pages: int,
                                    cap_hit: Optional[bool],
                                    error: Optional[str]):
        return self._resolver_cache.record_creator_resolution(
            mint_address=mint_address, creator=creator,
            creation_ts=creation_ts, n_sig_pages=n_sig_pages,
            cap_hit=cap_hit, error=error)

    def lookup_creator_cap_hit(self, mint_address: str) -> Optional[bool]:
        return self._resolver_cache.lookup_creator_cap_hit(mint_address)

    def has_funder_resolved(self, wallet_address: str) -> bool:
        return self._resolver_cache.has_funder_resolved(wallet_address)

    def record_funder_resolution(self, *, wallet_address: str,
                                    funder: Optional[str],
                                    amount_sol: Optional[float],
                                    fund_ts: Optional[float],
                                    error: Optional[str]):
        return self._resolver_cache.record_funder_resolution(
            wallet_address=wallet_address, funder=funder,
            amount_sol=amount_sol, fund_ts=fund_ts, error=error)

    def lookup_funder(self, wallet_address: str) -> Optional[str]:
        return self._resolver_cache.lookup_funder(wallet_address)

    def lookup_funder_status(self, wallet_address: str
                              ) -> tuple[Optional[str], str]:
        return self._resolver_cache.lookup_funder_status(wallet_address)

    def lookup_funder_full(self, wallet_address: str) -> Optional[dict]:
        return self._resolver_cache.lookup_funder_full(wallet_address)

    # ─────────────────────────────────────────────────────────────────
    # Shadow rug filter (funder-graph v1) eval log — Phase 1A step 5
    # Spec: 2026-05-18_funder_only_rug_filter_SPEC.md §6
    # ─────────────────────────────────────────────────────────────────

    def has_shadow_rug_funder_v1_eval(self, mint_address: str) -> bool:
        return self._shadow_store.has_shadow_rug_funder_v1_eval(mint_address)

    def record_shadow_rug_funder_v1_eval(self, *,
                                          mint_address: str,
                                          symbol: Optional[str],
                                          graduation_time: Optional[float],
                                          scored_at_delay_s: float,
                                          creator_address: Optional[str],
                                          funder_address: Optional[str],
                                          funder_status: str,
                                          F15: Optional[int],
                                          F16: Optional[int],
                                          F17: Optional[float],
                                          matched_root: Optional[str],
                                          decision: int,
                                          decision_str: str,
                                          reason: Optional[str]):
        return self._shadow_store.record_shadow_rug_funder_v1_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, creator_address=creator_address, funder_address=funder_address, funder_status=funder_status, F15=F15, F16=F16, F17=F17, matched_root=matched_root, decision=decision, decision_str=decision_str, reason=reason)

    def log_vwmp_shadow(self, *, timestamp: float, mint_address: str,
                          grpc_price: Optional[float],
                          vwmp_price: Optional[float],
                          vwmp_buffer_size: Optional[int],
                          vwmp_qualified_count: Optional[int],
                          vwmp_status: Optional[str],
                          grpc_pnl_pct: Optional[float],
                          vwmp_pnl_pct: Optional[float],
                          price_source_used: str):
        return self._diagnostic_store.log_vwmp_shadow(timestamp=timestamp, mint_address=mint_address, grpc_price=grpc_price, vwmp_price=vwmp_price, vwmp_buffer_size=vwmp_buffer_size, vwmp_qualified_count=vwmp_qualified_count, vwmp_status=vwmp_status, grpc_pnl_pct=grpc_pnl_pct, vwmp_pnl_pct=vwmp_pnl_pct, price_source_used=price_source_used)

    def log_cs_price_divergence(self, *, timestamp: float, mint_address: str,
                                  pool_price_sol: Optional[float],
                                  inline_vwmp_price: Optional[float],
                                  inline_vwmp_status: Optional[str],
                                  inline_returned_null: int,
                                  cs_authoritative_price: Optional[float],
                                  cs_returned_null: int,
                                  cs_latency_ms: Optional[float],
                                  divergence_pct: Optional[float],
                                  flagged_5pct: int,
                                  flagged_2pct: int,
                                  flagged_1pct: int,
                                  current_price_used: Optional[float],
                                  notes: Optional[str] = None):
        return self._diagnostic_store.log_cs_price_divergence(timestamp=timestamp, mint_address=mint_address, pool_price_sol=pool_price_sol, inline_vwmp_price=inline_vwmp_price, inline_vwmp_status=inline_vwmp_status, inline_returned_null=inline_returned_null, cs_authoritative_price=cs_authoritative_price, cs_returned_null=cs_returned_null, cs_latency_ms=cs_latency_ms, divergence_pct=divergence_pct, flagged_5pct=flagged_5pct, flagged_2pct=flagged_2pct, flagged_1pct=flagged_1pct, current_price_used=current_price_used, notes=notes)

    def log_cs_peak_phantom_event(self, *, timestamp: float, mint_address: str,
                                    symbol: Optional[str],
                                    poll_price_sol: Optional[float],
                                    poll_pnl_pct: Optional[float],
                                    prior_peak_poll: Optional[float],
                                    cs_authoritative_price: Optional[float],
                                    cs_pnl_pct: Optional[float],
                                    divergence_pct: Optional[float],
                                    phantom_guard_passed: int,
                                    cs_gate_passed: int,
                                    action: str,
                                    stream_peak_pnl: Optional[float],
                                    stream_confirm_ratio: Optional[float],
                                    cs_warmup_complete: int,
                                    cs_buffer_size: Optional[int],
                                    notes: Optional[str] = None):
        return self._diagnostic_store.log_cs_peak_phantom_event(timestamp=timestamp, mint_address=mint_address, symbol=symbol, poll_price_sol=poll_price_sol, poll_pnl_pct=poll_pnl_pct, prior_peak_poll=prior_peak_poll, cs_authoritative_price=cs_authoritative_price, cs_pnl_pct=cs_pnl_pct, divergence_pct=divergence_pct, phantom_guard_passed=phantom_guard_passed, cs_gate_passed=cs_gate_passed, action=action, stream_peak_pnl=stream_peak_pnl, stream_confirm_ratio=stream_confirm_ratio, cs_warmup_complete=cs_warmup_complete, cs_buffer_size=cs_buffer_size, notes=notes)

    def log_b4_inference_divergence(self, *, model_id: str, mint_address: str,
                                      symbol: Optional[str],
                                      entry_t: float, decision_t: float,
                                      features_old: Optional[dict],
                                      score_old: Optional[float],
                                      old_compute_ok: int,
                                      old_latency_ms: Optional[float],
                                      features_new: Optional[dict],
                                      score_new: Optional[float],
                                      new_compute_ok: int,
                                      new_latency_ms: Optional[float],
                                      decision_score_cutoff: float = 0.5,
                                      notes: Optional[str] = None):
        return self._diagnostic_store.log_b4_inference_divergence(model_id=model_id, mint_address=mint_address, symbol=symbol, entry_t=entry_t, decision_t=decision_t, features_old=features_old, score_old=score_old, old_compute_ok=old_compute_ok, old_latency_ms=old_latency_ms, features_new=features_new, score_new=score_new, new_compute_ok=new_compute_ok, new_latency_ms=new_latency_ms, decision_score_cutoff=decision_score_cutoff, notes=notes)

    def record_shadow_event_eval(self, *, mint_address: str, symbol: Optional[str],
                                  graduation_time: Optional[float],
                                  scan_t_sec: int,
                                  cutoff_band: str, cutoff_value: float,
                                  score: float, pattern: str,
                                  entry_price_sol: Optional[float],
                                  sol_price_usd: Optional[float],
                                  model_version: Optional[str],
                                  features: Optional[Dict]):
        return self._shadow_store.record_shadow_event_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scan_t_sec=scan_t_sec, cutoff_band=cutoff_band, cutoff_value=cutoff_value, score=score, pattern=pattern, entry_price_sol=entry_price_sol, sol_price_usd=sol_price_usd, model_version=model_version, features=features)

    def record_shadow_big_winner_eval(self, *, mint_address: str,
                                        symbol: Optional[str],
                                        graduation_time: Optional[float],
                                        scan_t_sec: int,
                                        cutoff_value: float,
                                        score: float,
                                        decision_pass: bool,
                                        n_swaps: int,
                                        entry_price_sol: Optional[float],
                                        sol_price_usd: Optional[float],
                                        model_version: str = "big_winner_v2",
                                        features_json: Optional[str] = None):
        return self._shadow_store.record_shadow_big_winner_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scan_t_sec=scan_t_sec, cutoff_value=cutoff_value, score=score, decision_pass=decision_pass, n_swaps=n_swaps, entry_price_sol=entry_price_sol, sol_price_usd=sol_price_usd, model_version=model_version, features_json=features_json)

    def record_shadow_vshape_v3_4_eval(self, *, mint_address: str,
                                         symbol: Optional[str],
                                         graduation_time: Optional[float],
                                         scan_t_sec: int,
                                         cutoff_value: float,
                                         score: float,
                                         decision_pass: bool,
                                         ood_pass: Optional[bool] = None,
                                         pattern_detected: Optional[bool] = None,
                                         n_swaps: int = 0,
                                         entry_price_sol: Optional[float] = None,
                                         sol_price_usd: Optional[float] = None,
                                         model_version: str = "vshape_v3_4",
                                         features_json: Optional[str] = None):
        return self._shadow_store.record_shadow_vshape_v3_4_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scan_t_sec=scan_t_sec, cutoff_value=cutoff_value, score=score, decision_pass=decision_pass, ood_pass=ood_pass, pattern_detected=pattern_detected, n_swaps=n_swaps, entry_price_sol=entry_price_sol, sol_price_usd=sol_price_usd, model_version=model_version, features_json=features_json)

    def record_preflight_check(self, *, mint_address: str,
                               symbol: Optional[str], check_name: str,
                               outcome: str, value: Optional[float] = None,
                               threshold: Optional[float] = None,
                               detail: Optional[str] = None,
                               model_score: Optional[float] = None,
                               pool_liq_usd: Optional[float] = None):
        return self._operational_store.record_preflight_check(mint_address=mint_address, symbol=symbol, check_name=check_name, outcome=outcome, value=value, threshold=threshold, detail=detail, model_score=model_score, pool_liq_usd=pool_liq_usd)

    def record_rug_event(self, *, mint_address: str,
                         pool_address: Optional[str],
                         sol_amount: float, price_sol: float,
                         detected_at: float, action_taken: str,
                         signature: Optional[str] = None,
                         triggered_pnl_pct: Optional[float] = None):
        return self._operational_store.record_rug_event(mint_address=mint_address, pool_address=pool_address, sol_amount=sol_amount, price_sol=price_sol, detected_at=detected_at, action_taken=action_taken, signature=signature, triggered_pnl_pct=triggered_pnl_pct)

    def record_shadow_event_invariant_eval(self, *, mint_address: str,
                                           symbol: Optional[str],
                                           graduation_time: Optional[float],
                                           scan_t_sec: int,
                                           cutoff_band: str, cutoff_value: float,
                                           score: float, pattern: str,
                                           entry_price_sol: Optional[float],
                                           sol_price_usd: Optional[float],
                                           model_version: Optional[str],
                                           features: Optional[Dict]):
        return self._shadow_store.record_shadow_event_invariant_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, scan_t_sec=scan_t_sec, cutoff_band=cutoff_band, cutoff_value=cutoff_value, score=score, pattern=pattern, entry_price_sol=entry_price_sol, sol_price_usd=sol_price_usd, model_version=model_version, features=features)

    def record_price_probe(self, *, mint_address: str, symbol: Optional[str],
                            hold_sec: Optional[float],
                            entry_price_sol: Optional[float],
                            pool_price_sol: Optional[float],
                            jupiter_price_usd: Optional[float],
                            sol_price_usd: Optional[float],
                            divergence_pct: Optional[float],
                            price_source: Optional[str],
                            pool_pnl_pct: Optional[float],
                            jup_pnl_pct: Optional[float],
                            peak_pnl_pct: Optional[float],
                            trailing_activated: bool):
        return self._operational_store.record_price_probe(mint_address=mint_address, symbol=symbol, hold_sec=hold_sec, entry_price_sol=entry_price_sol, pool_price_sol=pool_price_sol, jupiter_price_usd=jupiter_price_usd, sol_price_usd=sol_price_usd, divergence_pct=divergence_pct, price_source=price_source, pool_pnl_pct=pool_pnl_pct, jup_pnl_pct=jup_pnl_pct, peak_pnl_pct=peak_pnl_pct, trailing_activated=trailing_activated)

    # --- v5.3 shadow eval (Phase 15e, 2026-04-29) -----------------------------

    def record_shadow_v5_3_eval(self, *, mint_address: str,
                                  position_id: Optional[int],
                                  hold_sec: int,
                                  p_dd_60s: Optional[float],
                                  p_dd_120s: Optional[float],
                                  p_dd_v5_3: Optional[float],
                                  cutoff_value: float,
                                  would_fire: int,
                                  pos_pnl_pct: Optional[float],
                                  peak_pnl_pct: Optional[float],
                                  sol_price_usd: Optional[float],
                                  fail_reason: Optional[str] = None):
        return self._shadow_store.record_shadow_v5_3_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_60s=p_dd_60s, p_dd_120s=p_dd_120s, p_dd_v5_3=p_dd_v5_3, cutoff_value=cutoff_value, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason)

    def record_shadow_v5_5_eval(self, *, mint_address: str,
                                  position_id: Optional[int],
                                  hold_sec: int,
                                  p_dd_v5_5: Optional[float],
                                  raw_cur_pnl: Optional[float],
                                  cutoff_value: float,
                                  profit_gate: float,
                                  would_fire: int,
                                  pos_pnl_pct: Optional[float],
                                  peak_pnl_pct: Optional[float],
                                  sol_price_usd: Optional[float],
                                  fail_reason: Optional[str] = None):
        return self._shadow_store.record_shadow_v5_5_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_5=p_dd_v5_5, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason)

    def record_shadow_v5_5_3_eval(self, *, mint_address: str,
                                    position_id: Optional[int],
                                    hold_sec: int,
                                    p_dd_v5_5: Optional[float],
                                    raw_cur_pnl: Optional[float],
                                    cutoff_value: float,
                                    profit_gate: float,
                                    would_fire: int,
                                    pos_pnl_pct: Optional[float],
                                    peak_pnl_pct: Optional[float],
                                    sol_price_usd: Optional[float],
                                    fail_reason: Optional[str] = None,
                                    n_outlier_clips_applied: int = 0):
        return self._shadow_store.record_shadow_v5_5_3_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_5=p_dd_v5_5, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason, n_outlier_clips_applied=n_outlier_clips_applied)

    def record_shadow_v5_5_6_eval(self, *, mint_address: str,
                                    position_id: Optional[int],
                                    hold_sec: int,
                                    p_dd_v5_5: Optional[float],
                                    raw_cur_pnl: Optional[float],
                                    cutoff_value: float,
                                    profit_gate: float,
                                    would_fire: int,
                                    pos_pnl_pct: Optional[float],
                                    peak_pnl_pct: Optional[float],
                                    sol_price_usd: Optional[float],
                                    fail_reason: Optional[str] = None,
                                    n_outlier_clips_applied: int = 0):
        return self._shadow_store.record_shadow_v5_5_6_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_5=p_dd_v5_5, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason, n_outlier_clips_applied=n_outlier_clips_applied)

    def record_shadow_v5_6_eval(self, *, mint_address: str,
                                  position_id: Optional[int],
                                  hold_sec: int,
                                  p_dd_v5_5: Optional[float],
                                  raw_cur_pnl: Optional[float],
                                  cutoff_value: float,
                                  profit_gate: float,
                                  would_fire: int,
                                  pos_pnl_pct: Optional[float],
                                  peak_pnl_pct: Optional[float],
                                  sol_price_usd: Optional[float],
                                  fail_reason: Optional[str] = None,
                                  n_outlier_clips_applied: int = 0):
        return self._shadow_store.record_shadow_v5_6_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_5=p_dd_v5_5, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason, n_outlier_clips_applied=n_outlier_clips_applied)

    def record_shadow_corroboration_eval(self, *, mint_address: str,
                                          position_id: Optional[int],
                                          exit_path: str,
                                          hold_sec: Optional[int],
                                          pnl_pct: Optional[float],
                                          peak_pnl_pct: Optional[float],
                                          peak_pnl_pct_poll: Optional[float],
                                          drop_from_peak: Optional[float],
                                          n_swaps_window: Optional[int],
                                          would_fire_now: int,
                                          peak_confirmed: Optional[int],
                                          level_confirmed: Optional[int],
                                          would_fire_with_peak_confirm: Optional[int],
                                          would_fire_with_level_confirm: Optional[int],
                                          would_fire_with_both: Optional[int],
                                          window_sec: Optional[float],
                                          min_confirmations: Optional[int],
                                          peak_tol_pct: Optional[float],
                                          vol_floor_sol: Optional[float] = None):
        return self._shadow_store.record_shadow_corroboration_eval(mint_address=mint_address, position_id=position_id, exit_path=exit_path, hold_sec=hold_sec, pnl_pct=pnl_pct, peak_pnl_pct=peak_pnl_pct, peak_pnl_pct_poll=peak_pnl_pct_poll, drop_from_peak=drop_from_peak, n_swaps_window=n_swaps_window, would_fire_now=would_fire_now, peak_confirmed=peak_confirmed, level_confirmed=level_confirmed, would_fire_with_peak_confirm=would_fire_with_peak_confirm, would_fire_with_level_confirm=would_fire_with_level_confirm, would_fire_with_both=would_fire_with_both, window_sec=window_sec, min_confirmations=min_confirmations, peak_tol_pct=peak_tol_pct, vol_floor_sol=vol_floor_sol)

    def record_exit_decision(self, *, mint_address: str,
                             position_id: Optional[int],
                             hold_sec: Optional[int],
                             entry_price_sol: Optional[float],
                             current_price: Optional[float],
                             pnl_pct: Optional[float],
                             peak_pnl_pct: Optional[float],
                             peak_pnl_pct_poll: Optional[float],
                             trail_active: Optional[int],
                             pool_price_sol: Optional[float],
                             n_swaps_window: Optional[int],
                             is_fire_tick: int = 0,
                             fire_reason: Optional[str] = None):
        # E1 live-shadow exit-decision log delegate (W1, 2026-06-08). TradeDB facade ->
        # ShadowStore, mirroring record_shadow_corroboration_eval above. Without this
        # delegate ctrl.db.record_exit_decision raises AttributeError (swallowed by the
        # caller's try/except) -> shadow_exit_decisions stays empty. See test_e1_tradedb_integration.
        return self._shadow_store.record_exit_decision(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, entry_price_sol=entry_price_sol, current_price=current_price, pnl_pct=pnl_pct, peak_pnl_pct=peak_pnl_pct, peak_pnl_pct_poll=peak_pnl_pct_poll, trail_active=trail_active, pool_price_sol=pool_price_sol, n_swaps_window=n_swaps_window, is_fire_tick=is_fire_tick, fire_reason=fire_reason)

    def record_shadow_v5_5_7_v2_1_eval(self, *, mint_address: str,
                                          position_id: Optional[int],
                                          hold_sec: int,
                                          p_dd_v5_5: Optional[float],
                                          raw_cur_pnl: Optional[float],
                                          cutoff_value: float,
                                          profit_gate: float,
                                          would_fire: int,
                                          pos_pnl_pct: Optional[float],
                                          peak_pnl_pct: Optional[float],
                                          sol_price_usd: Optional[float],
                                          fail_reason: Optional[str] = None,
                                          n_outlier_clips_applied: int = 0):
        return self._shadow_store.record_shadow_v5_5_7_v2_1_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_5=p_dd_v5_5, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason, n_outlier_clips_applied=n_outlier_clips_applied)

    def record_shadow_v5_3_1_eval(self, *, mint_address: str,
                                    position_id: Optional[int],
                                    hold_sec: int,
                                    p_dd_v5_3_1: Optional[float],
                                    raw_cur_pnl: Optional[float],
                                    cutoff_value: float,
                                    profit_gate: float,
                                    would_fire: int,
                                    pos_pnl_pct: Optional[float],
                                    peak_pnl_pct: Optional[float],
                                    sol_price_usd: Optional[float],
                                    fail_reason: Optional[str] = None):
        return self._shadow_store.record_shadow_v5_3_1_eval(mint_address=mint_address, position_id=position_id, hold_sec=hold_sec, p_dd_v5_3_1=p_dd_v5_3_1, raw_cur_pnl=raw_cur_pnl, cutoff_value=cutoff_value, profit_gate=profit_gate, would_fire=would_fire, pos_pnl_pct=pos_pnl_pct, peak_pnl_pct=peak_pnl_pct, sol_price_usd=sol_price_usd, fail_reason=fail_reason)

    # --- 14y shadow-exit-warn -------------------------------------------------

    def record_shadow_exit_warn_eval(self, *, position_id: Optional[int],
                                     mint_address: str,
                                     symbol: Optional[str],
                                     dt_from_entry: int,
                                     dt_from_grad: Optional[int],
                                     p_drop_raw: float,
                                     p_rug_raw: float,
                                     mid_price_sol: Optional[float],
                                     n_cache_swaps: Optional[int],
                                     feature_window_count: Optional[int],
                                     features_json: Optional[str] = None,
                                     p_rug_live_v1: Optional[float] = None,
                                     p_rug_v3: Optional[float] = None,
                                     p_rug_v3_window: Optional[int] = None,
                                     p_rug_v4: Optional[float] = None,
                                     p_rug_v4_y60s: Optional[float] = None,
                                     p_rug_v4_y120s: Optional[float] = None,
                                     p_dd_v5: Optional[float] = None,
                                     p_dd_v5_60s: Optional[float] = None,
                                     p_dd_v5_120s: Optional[float] = None,
                                     v4_fail_reason: Optional[str] = None,
                                     v5_fail_reason: Optional[str] = None):
        return self._shadow_store.record_shadow_exit_warn_eval(position_id=position_id, mint_address=mint_address, symbol=symbol, dt_from_entry=dt_from_entry, dt_from_grad=dt_from_grad, p_drop_raw=p_drop_raw, p_rug_raw=p_rug_raw, mid_price_sol=mid_price_sol, n_cache_swaps=n_cache_swaps, feature_window_count=feature_window_count, features_json=features_json, p_rug_live_v1=p_rug_live_v1, p_rug_v3=p_rug_v3, p_rug_v3_window=p_rug_v3_window, p_rug_v4=p_rug_v4, p_rug_v4_y60s=p_rug_v4_y60s, p_rug_v4_y120s=p_rug_v4_y120s, p_dd_v5=p_dd_v5, p_dd_v5_60s=p_dd_v5_60s, p_dd_v5_120s=p_dd_v5_120s, v4_fail_reason=v4_fail_reason, v5_fail_reason=v5_fail_reason)

    def fetch_unresolved_shadow_exit_warn_evals(self, older_than_sec: float = 60.0,
                                                limit: int = 1000) -> list:
        return self._shadow_store.fetch_unresolved_shadow_exit_warn_evals(older_than_sec=older_than_sec, limit=limit)

    def resolve_shadow_exit_warn_eval(self, *, row_id: int,
                                      realized_drop_60: Optional[float],
                                      realized_max_sell_60: Optional[float],
                                      y_drop_actual: Optional[int],
                                      y_rug_actual: Optional[int]):
        return self._shadow_store.resolve_shadow_exit_warn_eval(row_id=row_id, realized_drop_60=realized_drop_60, realized_max_sell_60=realized_max_sell_60, y_drop_actual=y_drop_actual, y_rug_actual=y_rug_actual)

    def count_shadow_exit_warn_evals(self, resolved: Optional[bool] = None) -> int:
        return self._shadow_store.count_shadow_exit_warn_evals(resolved=resolved)

    # ── 19c SW v2 shadow (F1 entry model at T+180s) ────────────────────────

    def record_shadow_sw_v2_eval(self, *, mint_address: str,
                                 symbol: Optional[str],
                                 graduation_time: float,
                                 dt_from_grad: int,
                                 entry_offset_sec: Optional[float],
                                 entry_price_sol: Optional[float],
                                 grad_price_sol: Optional[float],
                                 n_bars_pre_entry: Optional[int],
                                 n_swaps_used: Optional[int],
                                 feature_ready: bool,
                                 f1_score: Optional[float],
                                 features_json: Optional[str],
                                 v14_candidate: bool = False,
                                 v14_entered: bool = False,
                                 v14_score: Optional[float] = None,
                                 v14_pattern: Optional[str] = None,
                                 rug_filter_score: Optional[float] = None):
        return self._shadow_store.record_shadow_sw_v2_eval(mint_address=mint_address, symbol=symbol, graduation_time=graduation_time, dt_from_grad=dt_from_grad, entry_offset_sec=entry_offset_sec, entry_price_sol=entry_price_sol, grad_price_sol=grad_price_sol, n_bars_pre_entry=n_bars_pre_entry, n_swaps_used=n_swaps_used, feature_ready=feature_ready, f1_score=f1_score, features_json=features_json, v14_candidate=v14_candidate, v14_entered=v14_entered, v14_score=v14_score, v14_pattern=v14_pattern, rug_filter_score=rug_filter_score)

    def fetch_unresolved_shadow_sw_v2_evals(self, older_than_sec: float = 900.0,
                                            limit: int = 500) -> list:
        return self._shadow_store.fetch_unresolved_shadow_sw_v2_evals(older_than_sec=older_than_sec, limit=limit)

    def resolve_shadow_sw_v2_eval(self, *, row_id: int,
                                  realized_raw_15m: Optional[float],
                                  y_net15_emp_trail_actual: Optional[int],
                                  peak_ret_15m: Optional[float],
                                  trough_ret_15m: Optional[float],
                                  eventual_trade_pnl_usd: Optional[float]):
        return self._shadow_store.resolve_shadow_sw_v2_eval(row_id=row_id, realized_raw_15m=realized_raw_15m, y_net15_emp_trail_actual=y_net15_emp_trail_actual, peak_ret_15m=peak_ret_15m, trough_ret_15m=trough_ret_15m, eventual_trade_pnl_usd=eventual_trade_pnl_usd)

    def count_shadow_sw_v2_evals(self, resolved: Optional[bool] = None) -> int:
        return self._shadow_store.count_shadow_sw_v2_evals(resolved=resolved)

    def record_shadow_pregrad_factors_eval(self, *, mint_address: str,
                                           graduation_time: Optional[float],
                                           scored_at_delay_s: Optional[float],
                                           n_pregrad_swaps: Optional[int],
                                           rpc_pages: Optional[int],
                                           bc_first_slot_vol_pct: Optional[float],
                                           pg_late20_sell_share: Optional[float],
                                           bc_churn_share: Optional[float],
                                           l1_score: Optional[int],
                                           l1_pass: Optional[int]):
        return self._shadow_store.record_shadow_pregrad_factors_eval(mint_address=mint_address, graduation_time=graduation_time, scored_at_delay_s=scored_at_delay_s, n_pregrad_swaps=n_pregrad_swaps, rpc_pages=rpc_pages, bc_first_slot_vol_pct=bc_first_slot_vol_pct, pg_late20_sell_share=pg_late20_sell_share, bc_churn_share=bc_churn_share, l1_score=l1_score, l1_pass=l1_pass)

    def record_shadow_xoscross_arm_eval(self, *, arm, mint_address, graduation_time,
                                        crossing_ts, crossing_rank, swap_rate_180s,
                                        quantile_line, decision, l1_pass,
                                        hypo_entry_price_sol, features_json=None,
                                        live_action=None):
        return self._shadow_store.record_shadow_xoscross_arm_eval(arm=arm, mint_address=mint_address, graduation_time=graduation_time, crossing_ts=crossing_ts, crossing_rank=crossing_rank, swap_rate_180s=swap_rate_180s, quantile_line=quantile_line, decision=decision, l1_pass=l1_pass, hypo_entry_price_sol=hypo_entry_price_sol, features_json=features_json, live_action=live_action)

    def fetch_xoscross_recent_rates(self, *, arm, limit=500):
        return self._shadow_store.fetch_xoscross_recent_rates(arm=arm, limit=limit)

    def update_xoscross_live_action(self, *, mint_address, action):
        return self._shadow_store.update_xoscross_live_action(mint_address=mint_address, action=action)

    def count_shadow_pregrad_factors_evals(self) -> int:
        return self._shadow_store.count_shadow_pregrad_factors_evals()

    def record_shadow_policy_eval(self, obs_id: Optional[int],
                                  mint_address: str,
                                  symbol: str,
                                  graduation_time: Optional[float],
                                  feature_delay_sec: int,
                                  stage1_name: Optional[str],
                                  stage1_score: Optional[float],
                                  stage1_threshold: Optional[float],
                                  stage1_passed: Optional[bool],
                                  stage2_rule_name: str,
                                  stage2_passed: Optional[bool],
                                  feature_source: str,
                                  feature_view: str,
                                  range_1to3m: Optional[float],
                                  drawdown_1to3m: Optional[float],
                                  total_volume_3m: Optional[float],
                                  total_trades_3m: Optional[float],
                                  metadata: Optional[Dict] = None):
        return self._shadow_store.record_shadow_policy_eval(obs_id, mint_address, symbol, graduation_time, feature_delay_sec, stage1_name, stage1_score, stage1_threshold, stage1_passed, stage2_rule_name, stage2_passed, feature_source, feature_view, range_1to3m, drawdown_1to3m, total_volume_3m, total_trades_3m, metadata)

    def save_position(self, pos: "Position"):
        return self._operational_store.save_position(pos)

    def remove_position(self, mint_address: str):
        return self._operational_store.remove_position(mint_address)

    def load_open_positions(self) -> List["Position"]:
        return self._operational_store.load_open_positions()

    def count_trades_by_entry_source(self, entry_source: str) -> int:
        return self._operational_store.count_trades_by_entry_source(entry_source)

    def get_recent_stoploss_mints(self, cooldown_sec: float) -> Dict[str, float]:
        return self._operational_store.get_recent_stoploss_mints(cooldown_sec)

    def get_trades_for_date(self, date_str: str) -> List[Dict]:
        return self._operational_store.get_trades_for_date(date_str)

    def load_all_trades(self) -> List[TradeRecord]:
        return self._operational_store.load_all_trades()

    def close(self):
        with self._lock:
            self._conn.close()
        if self._telemetry_sink is not None:
            self._telemetry_sink.close()
