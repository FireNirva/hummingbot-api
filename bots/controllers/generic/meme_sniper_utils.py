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
import joblib
import numpy as np
import pandas as pd

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logger = logging.getLogger(__name__)

# ── Solana Constants ──────────────────────────────────────────────────────
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPSWAP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"

# Tokens that can NEVER be a "newly graduated" base_mint. Any candidate where
# the parser thinks one of these is the graduated token is by definition a
# false positive — the parser misread a routed swap_buy/swap_sell as
# create_pool. See step1_dune_vs_bot_sanity_check report (2026-04-09): bot's
# WS source had 17 false positives in 30h, of which 8 were USDC/WSOL.
QUOTE_TOKEN_BLACKLIST = frozenset({WSOL_MINT, USDC_MINT, USDT_MINT})

# PumpSwap Anchor instruction discriminators (sha256("global:<name>")[:8]).
# Used to distinguish create_pool (graduation) from swap_buy/swap_sell.
# Source: PumpSwap IDL + empirical verification.
PUMPSWAP_CREATE_POOL_DISC = bytes([233, 146, 209, 142, 207, 104, 64, 188])
PUMPSWAP_BUY_DISC = bytes([102, 6, 61, 18, 1, 218, 235, 234])
PUMPSWAP_SELL_DISC = bytes([51, 230, 133, 164, 1, 127, 131, 173])

# Inline base58 decoder (avoids external dependency in Docker container)
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def _b58decode(s: str) -> bytes:
    n = 0
    for c in s.encode():
        n = n * 58 + _B58_ALPHABET.index(c)
    result = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    # Preserve leading zeros
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + result

# PumpSwap create_pool inner instruction account layout (verified empirically):
#   [0] pool address (PDA)
#   [1] global_config
#   [2] authority / creator
#   [3] base_mint         <-- the graduated token
#   [4] quote_mint        <-- WSOL
ACCOUNT_IDX_POOL = 0
ACCOUNT_IDX_CREATOR = 2
ACCOUNT_IDX_BASE_MINT = 3
ACCOUNT_IDX_QUOTE_MINT = 4
MIN_CREATE_POOL_ACCOUNTS = 15  # create_pool has 22 accounts; log/emit has 1-2


# ──────────────────────────────────────────────
# Tier 1: Solana RPC client + graduation parser
# ──────────────────────────────────────────────

@dataclass
class ChainGraduation:
    """Raw graduation detected on-chain (before GMGN enrichment)."""
    mint: str           # base_mint (the graduated token)
    pool: str           # PumpSwap pool address
    creator: str        # who triggered the graduation
    quote_mint: str     # typically WSOL
    signature: str      # transaction signature
    block_time: int     # unix timestamp
    slot: int


class SolanaRPC:
    """Lightweight async Solana JSON-RPC client for Chainstack."""

    def __init__(self, rpc_url: str, timeout: float = 30.0):
        self.rpc_url = rpc_url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._req_id = 0

    async def _call(self, method: str, params: list) -> dict:
        self._req_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        for attempt in range(3):
            try:
                resp = await self._client.post(self.rpc_url, json=payload)
                data = resp.json()
                if "error" in data:
                    err = data["error"]
                    if err.get("code") == 429 or resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.warning(f"RPC rate limited, retrying in {wait}s")
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"RPC error: {err}")
                return data.get("result")
            except httpx.RequestError as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("RPC max retries exceeded")

    async def get_signatures_for_address(
        self, address: str, before: Optional[str] = None,
        until: Optional[str] = None, limit: int = 100,
    ) -> list:
        opts: dict = {"limit": min(limit, 1000)}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        return await self._call("getSignaturesForAddress", [address, opts])

    async def get_transaction(self, signature: str) -> Optional[dict]:
        return await self._call("getTransaction", [
            signature,
            {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
        ])

    async def get_token_accounts_by_owner(self, owner: str, mint: str) -> dict:
        # Try SPL Token first, then Token-2022 (PumpFun migrated to Token-2022)
        result = await self._call("getTokenAccountsByOwner", [
            owner,
            {"mint": mint},
            {"encoding": "jsonParsed"},
        ])
        if result and result.get("value"):
            return result
        # Fallback: query Token-2022 program and filter to target mint
        t22 = await self._call("getTokenAccountsByOwner", [
            owner,
            {"programId": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"},
            {"encoding": "jsonParsed"},
        ])
        if t22 and t22.get("value"):
            t22["value"] = [
                a for a in t22["value"]
                if a.get("account", {}).get("data", {}).get("parsed", {})
                    .get("info", {}).get("mint") == mint
            ]
        return t22

    async def close(self):
        await self._client.aclose()


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
# Data classes
# ──────────────────────────────────────────────

@dataclass
class GraduatedToken:
    mint_address: str
    symbol: str
    name: str
    decimals: int
    graduation_time: float  # unix timestamp
    liquidity_usd: float
    price_usd: float
    pool_address: Optional[str] = None  # PumpSwap pool address (from Chainstack Tier 1)
    source: str = "chainstack"  # "chainstack" or "trending"
    # Trenches API activity fields (from M1 discovery, None if unavailable)
    swaps_24h: Optional[int] = None
    buys_24h: Optional[int] = None
    sells_24h: Optional[int] = None
    # Raw GMGN token_info response captured at M1 enrichment (for research DB)
    _gmgn_info_raw: Optional[Dict] = None


@dataclass
class TradeCandidate:
    token: GraduatedToken
    model_score: float
    features: Dict[str, float]
    queued_at: float = 0.0  # unix timestamp when queued (0 = not queued yet)
    last_swap_price_sol: float = 0.0  # cached at M2 time before flush, used by M3 double-check
    buy_fail_count: int = 0
    buy_last_fail_time: float = 0.0


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


@dataclass
class ObservationEntry:
    token: GraduatedToken
    obs_id: int  # row id in token_observations table
    expire_time: float  # unix timestamp when to fetch GMGN kline and finalize
    snapshots_done: set = None  # columns already fetched

    def __post_init__(self):
        if self.snapshots_done is None:
            self.snapshots_done = set()


@dataclass
class Position:
    token: GraduatedToken
    entry_price_sol: float
    token_amount: float
    entry_time: float
    entry_tx: str
    sol_invested: float
    model_score: float = 0.0
    features: Optional[Dict[str, float]] = None
    entry_quote_id: Optional[str] = None
    entry_quote_exec_price_sol: Optional[float] = None
    entry_decision_mid_price_sol: Optional[float] = None
    entry_decision_mid_source: Optional[str] = None
    peak_pnl_pct: float = 0.0
    trailing_activated: bool = False
    # V3 execution-quality fields (from M2/M3 preflight)
    m2_ref_price_sol: Optional[float] = None
    preflight_latency_ms: Optional[int] = None
    pool_liq_at_entry_usd: Optional[float] = None

    def hold_seconds(self) -> float:
        return time.time() - self.entry_time


@dataclass
class TradeRecord:
    token_symbol: str
    mint_address: str
    entry_price_sol: float
    exit_price_sol: float
    token_amount: float
    sol_invested: float
    sol_received: float
    pnl_sol: float
    pnl_usd: float
    hold_seconds: float
    exit_reason: str  # "stop_loss" | "time_limit" | "manual"
    entry_tx: str
    exit_tx: str
    model_score: float
    timestamp: float = field(default_factory=time.time)
    entry_time: float = 0.0
    peak_pnl_pct: float = 0.0
    sol_price_usd: float = 0.0
    trigger_pnl_pct: float = 0.0  # PnL% when exit was detected (before swap slippage)
    # V3 analysis fields — populated from M2/M3 state, used to dissect
    # execution quality (entry chase, preflight latency, pool depth).
    m2_ref_price_sol: Optional[float] = None
    preflight_latency_ms: Optional[int] = None
    pool_liq_at_entry_usd: Optional[float] = None


# ──────────────────────────────────────────────
# On-chain Kline Builder
#   Build 1min candles from PumpSwap pool swap transactions.
#   Replaces GMGN kline dependency for newly graduated tokens.
# ──────────────────────────────────────────────

@dataclass
class SwapRecord:
    """A single parsed swap from a PumpSwap pool transaction."""
    timestamp: int       # block time (unix seconds)
    price_sol: float     # effective price per token in SOL
    volume_sol: float    # trade volume in SOL
    is_buy: bool         # True if user bought the base token
    base_amount: float   # token amount traded (human units)
    trader_address: str = ""  # signer / trader used for microstructure overlays


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
        n = len(self._pools[mint])
        logger.info(f"KlineBuilder: registered {mint[:12]}... pool={pool_address[:12]}... (pool #{n})")

    def unregister(self, mint: str):
        """Stop tracking a token and free memory."""
        self._pools.pop(mint, None)
        self._swaps.pop(mint, None)
        self._seen_sigs.pop(mint, None)

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

    def inject_swap(self, mint: str, swap: "SwapRecord", signature: str = ""):
        """Push-based swap injection from gRPC Geyser stream.

        Safe to call from the same asyncio event loop (no threading concerns).
        Silently ignores unregistered mints.
        """
        if mint not in self._swaps:
            return
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
            params: dict = {"limit": 200}
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
                for tx_data in tx_results:
                    if tx_data is None:
                        continue
                    swap = self._parse_swap_from_tx(tx_data, mint)
                    if swap:
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

        # Step 1: Get recent signatures for the pool
        body = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getSignaturesForAddress",
            "params": [pool, {"limit": 200}],
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
        for tx_data in all_tx_results:
            if tx_data is None:
                continue
            swap = self._parse_swap_from_tx(tx_data, mint)
            if swap:
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

        # --- Step 2: Determine base_amount (always from token balances) ---
        # Total positive base deltas = total base received (by buyers or pool)
        base_received = sum(d for d in base_deltas.values() if d > 0)
        if base_received <= 0:
            return None

        base_amount = base_received

        # --- Step 3: Determine SOL amount ---
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

        # Use the larger value — handles both PumpSwap (WSOL) and Meteora (native SOL)
        sol_amount = max(wsol_amount, native_sol_amount)

        if sol_amount <= 0:
            return None

        # --- Step 4: Determine direction (buy vs sell) ---
        # Find the transaction signer (the user who initiated the swap)
        signer = self._get_tx_signer(tx_data)

        if signer and signer in base_deltas:
            # Signer's base delta: positive = bought base, negative = sold base
            is_buy = base_deltas[signer] > 0
        else:
            # Fallback: find the largest positive base delta owner (the buyer)
            # If they're also the largest SOL spender, it's a buy
            largest_base_receiver = max(base_deltas.items(), key=lambda x: x[1])
            is_buy = largest_base_receiver[1] > 0

        price_sol = sol_amount / base_amount
        volume_sol = sol_amount

        return SwapRecord(
            timestamp=block_time,
            price_sol=price_sol,
            volume_sol=volume_sol,
            is_buy=is_buy,
            base_amount=base_amount,
            trader_address=signer or "",
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
        """Lazily create Chainstack RPC client."""
        if not self._chainstack_rpc_url:
            return None
        if self._rpc is None:
            self._rpc = SolanaRPC(self._chainstack_rpc_url)
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
        price = float(token_data.get("price", 0) or 0)

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
                price_usd=float(item.get("price", 0) or 0),
                swaps_24h=int(item.get("swaps_24h", 0) or 0),
                buys_24h=int(item.get("buys_24h", 0) or 0),
                sells_24h=int(item.get("sells_24h", 0) or 0),
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
                "price_usd": float(token_data.get("price", 0) or 0),
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
                price_usd=float(item.get("price", 0) or 0),
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
                 base_url: str = "https://openapi.gmgn.ai"):
        model_data = joblib.load(model_path)
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


def _safe_float_value(x: Any) -> float:
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def _safe_return_value(price: float, base_price: float) -> float:
    if any(pd.isna(v) for v in [price, base_price]):
        return np.nan
    if price <= 0 or base_price <= 0:
        return np.nan
    return float(price / base_price - 1.0)


def _safe_ratio_value(num: float, den: float) -> float:
    if any(pd.isna(v) for v in [num, den]):
        return np.nan
    if den == 0:
        return np.nan
    return float(num / den)


def _safe_range_pos_value(value: float, low: float, high: float) -> float:
    if any(pd.isna(v) for v in [value, low, high]):
        return np.nan
    if high <= low:
        return np.nan
    return float((value - low) / (high - low))


def _upper_wick_frac_value(open_: float, close: float, high: float, low: float) -> float:
    if any(pd.isna(v) for v in [open_, close, high, low]):
        return np.nan
    if high <= low:
        return np.nan
    return float((high - max(open_, close)) / (high - low))


def _body_frac_value(open_: float, close: float, high: float, low: float) -> float:
    if any(pd.isna(v) for v in [open_, close, high, low]):
        return np.nan
    if high <= low:
        return np.nan
    return float(abs(close - open_) / (high - low))


def _range_pct_value(open_: float, high: float, low: float) -> float:
    if any(pd.isna(v) for v in [open_, high, low]):
        return np.nan
    if open_ <= 0:
        return np.nan
    return float((high - low) / open_)


def _tail_mean_value(values: np.ndarray, n: int) -> float:
    if len(values) == 0:
        return np.nan
    tail = values[max(0, len(values) - n):]
    if len(tail) == 0:
        return np.nan
    tail = tail[np.isfinite(tail)]
    if len(tail) == 0:
        return np.nan
    return float(np.mean(tail))


def _tail_share_positive_value(values: np.ndarray, n: int) -> float:
    if len(values) == 0:
        return np.nan
    tail = values[max(0, len(values) - n):]
    tail = tail[np.isfinite(tail)]
    if len(tail) == 0:
        return np.nan
    return float((tail > 0).mean())


def _close_anchor_value(closes: np.ndarray, last_idx: int, bars_back: int, fallback: float) -> float:
    anchor_idx = last_idx - bars_back
    if anchor_idx >= 0:
        return _safe_float_value(closes[anchor_idx])
    return _safe_float_value(fallback)


def _slice_sum_value(values: np.ndarray, start: int, end: int) -> float:
    if end <= start:
        return np.nan
    return float(np.nansum(values[start:end]))


def _series_slope(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return np.nan
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x[mask], arr[mask], 1)[0])


def _max_drawdown_from_path(rets: List[float]) -> float:
    arr = np.asarray(rets, dtype=float)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return np.nan
    vals = arr[mask]
    peaks = np.maximum.accumulate(vals)
    drawdowns = vals - peaks
    return float(drawdowns.min())


def _build_confirmed_entry_frame(kline_bars: List[Dict[str, Any]],
                                 graduation_time: float) -> Optional[pd.DataFrame]:
    if not kline_bars:
        return None
    df = pd.DataFrame(kline_bars).copy()
    if df.empty or "time" not in df.columns:
        return None
    df["bar_time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="ms", utc=True)
    df = df[df["bar_time"].notna()].copy()
    if df.empty:
        return None
    df = df.sort_values("bar_time").reset_index(drop=True)
    df["open"] = pd.to_numeric(df.get("open"), errors="coerce")
    df["high"] = pd.to_numeric(df.get("high"), errors="coerce")
    df["low"] = pd.to_numeric(df.get("low"), errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["volume_usd"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["n_trades"] = pd.to_numeric(df.get("trades"), errors="coerce")
    df["relative_minute"] = np.arange(len(df), dtype=int)
    grad_ts = pd.to_datetime(float(graduation_time), unit="s", utc=True)
    df["bar_start_offset_sec"] = (df["bar_time"] - grad_ts).dt.total_seconds().astype(float)
    df["bar_end_offset_sec"] = df["bar_start_offset_sec"] + 60.0
    df["contains_graduation"] = (
        (df["bar_start_offset_sec"] <= 0.0) & (df["bar_end_offset_sec"] > 0.0)
    )
    return df


def _get_post_grad_anchor_open_live(df: pd.DataFrame) -> Tuple[float, int]:
    anchor_idx = 1 if len(df) >= 2 and bool(df.iloc[0].get("contains_graduation", False)) else 0
    if len(df) <= anchor_idx:
        return np.nan, anchor_idx
    anchor_open = pd.to_numeric(pd.Series([df.iloc[anchor_idx]["open"]]), errors="coerce").iloc[0]
    return (float(anchor_open) if pd.notna(anchor_open) else np.nan), anchor_idx


def compute_confirmed_entry_live_features(kline_bars: List[Dict[str, Any]],
                                          graduation_time: float,
                                          delay_sec: int = 90) -> Optional[Dict[str, float]]:
    df = _build_confirmed_entry_frame(kline_bars, graduation_time)
    if df is None or df.empty:
        return None

    entry_candidates = df[pd.to_numeric(df["bar_start_offset_sec"], errors="coerce") >= float(delay_sec)].copy()
    if entry_candidates.empty:
        return None
    entry_row = entry_candidates.sort_values(["bar_start_offset_sec", "bar_time"]).iloc[0]
    entry_price = _safe_float_value(entry_row["open"])
    entry_offset_sec = _safe_float_value(entry_row["bar_start_offset_sec"])
    entry_relative_minute = int(entry_row["relative_minute"])
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_offset_sec):
        return None

    anchor_open, anchor_idx = _get_post_grad_anchor_open_live(df)
    if not np.isfinite(anchor_open) or anchor_open <= 0:
        return None

    hist = df[pd.to_numeric(df["bar_end_offset_sec"], errors="coerce") <= float(entry_offset_sec)].copy()
    if hist.empty:
        return None
    hist = hist[pd.to_numeric(hist["relative_minute"], errors="coerce") >= float(anchor_idx)].copy()
    hist = hist.reset_index(drop=True)
    n_hist = int(len(hist))
    if n_hist <= 0:
        return None

    close_rets = [_safe_return_value(v, anchor_open) for v in hist["close"].tolist()]
    low_rets = [_safe_return_value(v, anchor_open) for v in hist["low"].tolist()]
    high_rets = [_safe_return_value(v, anchor_open) for v in hist["high"].tolist()]

    total_volume = float(pd.to_numeric(hist["volume_usd"], errors="coerce").fillna(0.0).sum())
    total_trades = float(pd.to_numeric(hist["n_trades"], errors="coerce").fillna(0.0).sum())
    last_row = hist.iloc[-1]
    last_close_ret = _safe_float_value(close_rets[-1])
    last_low_ret = _safe_float_value(low_rets[-1])
    min_close_ret = float(np.nanmin(close_rets))
    max_close_ret = float(np.nanmax(close_rets))
    min_intrabar_low_ret = float(np.nanmin(low_rets))

    return {
        "entry_price": entry_price,
        "avg_trade_size_to_entry": total_volume / total_trades if total_trades > 0 else np.nan,
        "last_bar_volume": _safe_float_value(last_row["volume_usd"]),
        "avg_volume_per_bar_to_entry": total_volume / n_hist if n_hist > 0 else np.nan,
        "min_close_ret_to_entry": min_close_ret,
        "total_volume_to_entry": total_volume,
        "total_trades_to_entry": total_trades,
        "hist_last_close_ret": last_close_ret,
        "hist_last_low_ret": last_low_ret,
        "area_under_close_path_to_entry": float(np.nansum(close_rets)),
        "min_intrabar_low_ret_to_entry": min_intrabar_low_ret,
        "max_close_ret_to_entry": max_close_ret,
        "last_bar_trades": _safe_float_value(last_row["n_trades"]),
        "close_range_to_entry": max_close_ret - min_close_ret,
        "close_drawdown_to_entry": _max_drawdown_from_path(close_rets),
        "close_ret_slope_to_entry": _series_slope(close_rets),
        "entry_offset_sec": entry_offset_sec,
        "entry_relative_minute": float(entry_relative_minute),
        "hist_completed_postgrad_bars": float(n_hist),
    }


def compute_super_winner_event_core_features(
    kline_bars: List[Dict[str, Any]],
    graduation_time: float,
    delay_sec: int,
) -> Optional[Dict[str, float]]:
    df = _build_confirmed_entry_frame(kline_bars, graduation_time)
    if df is None or df.empty:
        return None

    entry_candidates = df[pd.to_numeric(df["bar_start_offset_sec"], errors="coerce") >= float(delay_sec)].copy()
    if entry_candidates.empty:
        return None
    entry_row = entry_candidates.sort_values(["bar_start_offset_sec", "bar_time"]).iloc[0]
    entry_price = _safe_float_value(entry_row["open"])
    entry_offset_sec = _safe_float_value(entry_row["bar_start_offset_sec"])
    entry_relative_minute = _safe_float_value(entry_row["relative_minute"])
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(entry_offset_sec):
        return None

    grad_price, anchor_idx = _get_post_grad_anchor_open_live(df)
    if not np.isfinite(grad_price) or grad_price <= 0:
        return None

    hist = df[pd.to_numeric(df["bar_end_offset_sec"], errors="coerce") <= float(entry_offset_sec)].copy()
    if hist.empty:
        return None
    hist = hist[pd.to_numeric(hist["relative_minute"], errors="coerce") >= float(anchor_idx)].copy().reset_index(drop=True)
    if hist.empty:
        return None

    opens = pd.to_numeric(hist["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(hist["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(hist["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(hist["close"], errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(hist["volume_usd"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    trades = pd.to_numeric(hist["n_trades"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid_prices = np.isfinite(opens) & np.isfinite(highs) & np.isfinite(lows) & np.isfinite(closes)
    if not valid_prices.any():
        return None

    n_obs = len(hist)
    last_idx = n_obs - 1
    prev_idx = max(0, last_idx - 1)

    peak_idx = int(np.nanargmax(highs))
    trough_idx = int(np.nanargmin(lows))
    peak_high = float(np.nanmax(highs))
    trough_low = float(np.nanmin(lows))
    peak_close = float(np.nanmax(closes))
    last_close = float(closes[last_idx])
    prev_close = float(closes[prev_idx])
    last_open = float(opens[last_idx])
    last_high = float(highs[last_idx])
    last_low = float(lows[last_idx])

    bar_rets = np.array([_safe_return_value(c, o) for c, o in zip(closes, opens)], dtype=float)
    range_pcts = np.array([_range_pct_value(o, h, l) for o, h, l in zip(opens, highs, lows)], dtype=float)
    upper_wicks = np.array([_upper_wick_frac_value(o, c, h, l) for o, c, h, l in zip(opens, closes, highs, lows)], dtype=float)
    body_fracs = np.array([_body_frac_value(o, c, h, l) for o, c, h, l in zip(opens, closes, highs, lows)], dtype=float)
    close_in_bar_range = np.array([_safe_range_pos_value(c, l, h) for c, l, h in zip(closes, lows, highs)], dtype=float)

    last2_start = max(0, n_obs - 2)
    prev2_start = max(0, n_obs - 4)
    prev2_end = max(0, n_obs - 2)

    vol_last2 = _slice_sum_value(volume, last2_start, n_obs)
    vol_prev2 = _slice_sum_value(volume, prev2_start, prev2_end)
    trades_last2 = _slice_sum_value(trades, last2_start, n_obs)
    trades_prev2 = _slice_sum_value(trades, prev2_start, prev2_end)

    recovery_den = peak_high - trough_low
    recovery_strength = np.nan
    if np.isfinite(recovery_den) and recovery_den > 0:
        recovery_strength = float((last_close - trough_low) / recovery_den)

    close_1m_anchor = _close_anchor_value(closes, last_idx, 1, grad_price)
    close_2m_anchor = _close_anchor_value(closes, last_idx, 2, grad_price)
    close_3m_anchor = _close_anchor_value(closes, last_idx, 3, grad_price)
    close_prev_step_anchor = _close_anchor_value(closes, last_idx, 2, grad_price)

    last1m_close_ret = _safe_return_value(last_close, close_1m_anchor)
    last2m_close_ret = _safe_return_value(last_close, close_2m_anchor)
    last3m_close_ret = _safe_return_value(last_close, close_3m_anchor)
    prev1m_close_ret = _safe_return_value(prev_close, close_prev_step_anchor)
    confirm_last2_close_slope = (
        last1m_close_ret - prev1m_close_ret
        if np.isfinite(last1m_close_ret) and np.isfinite(prev1m_close_ret)
        else np.nan
    )

    if n_obs >= 2:
        prev2_avg = float(np.nanmean(volume[max(0, n_obs - 3):max(0, n_obs - 1)]))
    else:
        prev2_avg = np.nan
    exhaustion_spike = _safe_ratio_value(float(volume[last_idx]), prev2_avg)

    return {
        "event_delay_sec": float(delay_sec),
        "event_entry_relative_minute": float(entry_relative_minute),
        "event_entry_ret_from_grad": _safe_return_value(entry_price, grad_price),
        "path_close_ret_last1m": last1m_close_ret,
        "path_close_ret_last2m": last2m_close_ret,
        "path_close_ret_last3m": last3m_close_ret,
        "path_drawdown_from_peak_to_entry": _safe_return_value(entry_price, peak_high),
        "path_last_close_vs_trough": _safe_return_value(last_close, trough_low),
        "path_recovery_from_trough_to_entry": _safe_return_value(entry_price, trough_low),
        "confirm_breakout_vs_peak_close": _safe_return_value(last_close, peak_close),
        "confirm_recovery_strength": recovery_strength,
        "flow_obs_trades": float(np.nansum(trades)),
        "flow_last_bar_trades": float(trades[last_idx]),
        "flow_last_bar_trades_share_of_obs": _safe_ratio_value(float(trades[last_idx]), float(np.nansum(trades))),
        "flow_trades_last2m_vs_prev2m": _safe_ratio_value(trades_last2, trades_prev2),
        "flow_vol_last2m_share_of_obs": _safe_ratio_value(vol_last2, float(np.nansum(volume))),
        "flow_obs_volume_per_bar": _safe_ratio_value(float(np.nansum(volume)), float(n_obs)),
        "confirm_close_in_last_bar_range": close_in_bar_range[last_idx],
        "confirm_close_near_pre_entry_high": _safe_ratio_value(last_close, peak_high),
        "reject_exhaustion_volume_spike": exhaustion_spike,
        "reject_last2_body_frac_mean": _tail_mean_value(body_fracs, 2),
        "reject_last2_upper_wick_mean": _tail_mean_value(upper_wicks, 2),
        "reject_last_bar_body_frac": body_fracs[last_idx],
        "reject_last_bar_red_flag": float(int(bar_rets[last_idx] < 0)) if np.isfinite(bar_rets[last_idx]) else np.nan,
        "reject_last_bar_upper_wick_frac": upper_wicks[last_idx],
        "reject_range_instability_last3": _tail_mean_value(range_pcts, 3),
        "reject_weak_close_after_spike": (
            exhaustion_spike * (1.0 - close_in_bar_range[last_idx])
            if np.isfinite(exhaustion_spike) and np.isfinite(close_in_bar_range[last_idx])
            else np.nan
        ),
        "confirm_last2_close_slope": confirm_last2_close_slope,
        "confirm_last2_green_share": _tail_share_positive_value(bar_rets, 2),
        "confirm_last_bar_green_flag": float(int(bar_rets[last_idx] > 0)) if np.isfinite(bar_rets[last_idx]) else np.nan,
        "confirm_positive_bar_share_last3": _tail_share_positive_value(bar_rets, 3),
        "flow_last_bar_volume_vs_prev2m_avg": _safe_ratio_value(float(volume[last_idx]), prev2_avg),
        "entry_price": entry_price,
        "entry_offset_sec": float(entry_offset_sec),
        "grad_price": float(grad_price),
        "hist_completed_postgrad_bars": float(n_obs),
        "path_bars_since_peak": float(n_obs - 1 - peak_idx),
        "path_bars_since_trough": float(n_obs - 1 - trough_idx),
    }


def _micro_price_return(prices: pd.Series) -> float:
    px = pd.to_numeric(prices, errors="coerce").dropna()
    if len(px) < 2:
        return np.nan
    first = _safe_float_value(px.iloc[0])
    last = _safe_float_value(px.iloc[-1])
    if not np.isfinite(first) or not np.isfinite(last) or first == 0:
        return np.nan
    return float(last / first - 1.0)


def _micro_price_range(prices: pd.Series) -> float:
    px = pd.to_numeric(prices, errors="coerce").dropna()
    if len(px) < 2:
        return np.nan
    low = _safe_float_value(px.min())
    high = _safe_float_value(px.max())
    if not np.isfinite(low) or not np.isfinite(high) or low == 0:
        return np.nan
    return float(high / low - 1.0)


def _micro_local_drawdown(prices: pd.Series) -> float:
    px = pd.to_numeric(prices, errors="coerce").dropna()
    if len(px) < 2:
        return np.nan
    values = px.astype(float).to_numpy()
    cummax = np.maximum.accumulate(values)
    dd = values / cummax - 1.0
    return float(dd.min()) if len(dd) else np.nan


def _micro_close_pos_in_range(prices: pd.Series) -> float:
    px = pd.to_numeric(prices, errors="coerce").dropna()
    if len(px) < 2:
        return np.nan
    low = _safe_float_value(px.min())
    high = _safe_float_value(px.max())
    last = _safe_float_value(px.iloc[-1])
    if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(last):
        return np.nan
    if high <= low:
        return 1.0
    return float((last - low) / (high - low))


def _micro_top_volume_stats(sub: pd.DataFrame) -> Dict[str, float]:
    out = {
        "top1_volume_usd": np.nan,
        "top3_volume_usd": np.nan,
        "top1_share": np.nan,
        "top3_share": np.nan,
        "hhi": np.nan,
    }
    if sub.empty or "trader_address" not in sub.columns:
        return out
    grouped = (
        sub.groupby("trader_address", dropna=True)["usd_amount"]
        .sum()
        .astype(float)
        .sort_values(ascending=False)
    )
    if grouped.empty:
        return out
    total = float(grouped.sum())
    shares = grouped / total if total > 0 else grouped * np.nan
    out["top1_volume_usd"] = float(grouped.iloc[0])
    out["top3_volume_usd"] = float(grouped.head(3).sum())
    out["top1_share"] = float(shares.iloc[0]) if len(shares) else np.nan
    out["top3_share"] = float(shares.head(3).sum()) if len(shares) else np.nan
    out["hhi"] = float((shares ** 2).sum()) if len(shares) else np.nan
    return out


def _micro_window_stats(sub: pd.DataFrame) -> Dict[str, float]:
    out = {
        "swap_count": 0,
        "unique_traders": 0,
        "unique_buyers": 0,
        "unique_sellers": 0,
        "volume_usd": 0.0,
        "buy_volume_usd": 0.0,
        "sell_volume_usd": 0.0,
        "volume_imbalance": np.nan,
        "avg_trade_size_usd": np.nan,
        "median_trade_size_usd": np.nan,
        "max_trade_size_usd": np.nan,
        "return": np.nan,
        "range": np.nan,
        "local_drawdown": np.nan,
        "close_pos_in_range": np.nan,
        "top1_buy_volume_usd": np.nan,
        "top3_buy_volume_usd": np.nan,
        "top1_buyer_share": np.nan,
        "top3_buyer_share": np.nan,
        "buyer_hhi": np.nan,
        "top1_seller_share": np.nan,
        "top3_seller_share": np.nan,
        "seller_hhi": np.nan,
    }
    if sub.empty:
        return out

    buy_sub = sub[sub["is_buy"] == 1].copy()
    sell_sub = sub[sub["is_buy"] == 0].copy()
    out["swap_count"] = int(len(sub))
    out["unique_traders"] = int(sub["trader_address"].replace("", np.nan).nunique(dropna=True))
    out["unique_buyers"] = int(buy_sub["trader_address"].replace("", np.nan).nunique(dropna=True))
    out["unique_sellers"] = int(sell_sub["trader_address"].replace("", np.nan).nunique(dropna=True))
    out["volume_usd"] = float(pd.to_numeric(sub["usd_amount"], errors="coerce").fillna(0.0).sum())
    out["buy_volume_usd"] = float(pd.to_numeric(buy_sub["usd_amount"], errors="coerce").fillna(0.0).sum())
    out["sell_volume_usd"] = float(pd.to_numeric(sell_sub["usd_amount"], errors="coerce").fillna(0.0).sum())

    total = out["buy_volume_usd"] + out["sell_volume_usd"]
    if total > 0:
        out["volume_imbalance"] = float((out["buy_volume_usd"] - out["sell_volume_usd"]) / total)

    trade_sizes = pd.to_numeric(sub["usd_amount"], errors="coerce").dropna()
    if len(trade_sizes):
        out["avg_trade_size_usd"] = float(trade_sizes.mean())
        out["median_trade_size_usd"] = float(trade_sizes.median())
        out["max_trade_size_usd"] = float(trade_sizes.max())

    prices = pd.to_numeric(sub["effective_price_usd"], errors="coerce").dropna()
    out["return"] = _micro_price_return(prices)
    out["range"] = _micro_price_range(prices)
    out["local_drawdown"] = _micro_local_drawdown(prices)
    out["close_pos_in_range"] = _micro_close_pos_in_range(prices)

    buy_top = _micro_top_volume_stats(buy_sub)
    sell_top = _micro_top_volume_stats(sell_sub)
    out["top1_buy_volume_usd"] = buy_top["top1_volume_usd"]
    out["top3_buy_volume_usd"] = buy_top["top3_volume_usd"]
    out["top1_buyer_share"] = buy_top["top1_share"]
    out["top3_buyer_share"] = buy_top["top3_share"]
    out["buyer_hhi"] = buy_top["hhi"]
    out["top1_seller_share"] = sell_top["top1_share"]
    out["top3_seller_share"] = sell_top["top3_share"]
    out["seller_hhi"] = sell_top["hhi"]
    return out


def compute_super_winner_event_micro_overlay_features(
    swaps: List[SwapRecord],
    event_time_sec: int,
    sol_price_usd: float,
) -> Dict[str, float]:
    out = {
        "raw_swap_row_covered": False,
        "micro_swap_count_60s": np.nan,
        "micro_buy_volume_30s_vs_prev30": np.nan,
        "micro_unique_buyers_30s_vs_prev30": np.nan,
        "micro_imbalance_30s_minus_prev30": np.nan,
        "micro_top1_buy_volume_usd_60s": np.nan,
        "micro_top3_buy_volume_usd_60s": np.nan,
        "micro_top3_buyer_support_vs_sell_60s": np.nan,
        "micro_overlay_non_null_count": 0.0,
        "micro_overlay_available_flag": 0.0,
    }
    if not swaps:
        return out

    px_sol_usd = sol_price_usd if np.isfinite(sol_price_usd) and sol_price_usd > 0 else 80.0
    rows = [
        {
            "block_time": int(s.timestamp),
            "trader_address": (s.trader_address or "").strip(),
            "is_buy": int(bool(s.is_buy)),
            "usd_amount": float(s.volume_sol * px_sol_usd),
            "effective_price_usd": float(s.price_sol * px_sol_usd),
        }
        for s in swaps
        if s is not None and s.timestamp < int(event_time_sec)
    ]
    if not rows:
        return out

    df = pd.DataFrame(rows).sort_values("block_time").reset_index(drop=True)
    if df.empty:
        return out
    out["raw_swap_row_covered"] = True

    last30 = df[df["block_time"] >= int(event_time_sec) - 30].copy()
    last60 = df[df["block_time"] >= int(event_time_sec) - 60].copy()
    prev30 = df[(df["block_time"] >= int(event_time_sec) - 60) & (df["block_time"] < int(event_time_sec) - 30)].copy()

    s30 = _micro_window_stats(last30)
    s60 = _micro_window_stats(last60)
    p30 = _micro_window_stats(prev30)
    out.update(
        {
            "micro_swap_count_60s": float(s60["swap_count"]),
            "micro_top1_buy_volume_usd_60s": s60["top1_buy_volume_usd"],
            "micro_top3_buy_volume_usd_60s": s60["top3_buy_volume_usd"],
            "micro_top3_buyer_support_vs_sell_60s": _safe_ratio_value(
                s60["top3_buy_volume_usd"],
                s60["sell_volume_usd"],
            ),
            "micro_buy_volume_30s_vs_prev30": _safe_ratio_value(
                s30["buy_volume_usd"],
                p30["buy_volume_usd"],
            ),
            "micro_unique_buyers_30s_vs_prev30": _safe_ratio_value(
                float(s30["unique_buyers"]),
                float(p30["unique_buyers"]),
            ),
            "micro_imbalance_30s_minus_prev30": (
                float(s30["volume_imbalance"] - p30["volume_imbalance"])
                if np.isfinite(s30["volume_imbalance"]) and np.isfinite(p30["volume_imbalance"])
                else np.nan
            ),
        }
    )

    overlay_cols = [
        "micro_buy_volume_30s_vs_prev30",
        "micro_unique_buyers_30s_vs_prev30",
        "micro_imbalance_30s_minus_prev30",
        "micro_top1_buy_volume_usd_60s",
        "micro_top3_buy_volume_usd_60s",
        "micro_top3_buyer_support_vs_sell_60s",
    ]
    non_null = int(sum(np.isfinite(_safe_float_value(out[col])) for col in overlay_cols))
    out["micro_overlay_non_null_count"] = float(non_null)
    out["micro_overlay_available_flag"] = float(int(non_null == len(overlay_cols)))
    return out


def _family_pass_threshold(n_features: int) -> int:
    return max(3, int(math.ceil(n_features * 0.5)))


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

    def __init__(self, model_path: str):
        path = Path(model_path)
        model_data = joblib.load(path)
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

    def __init__(self, model_path: str):
        path = Path(model_path)
        model_data = joblib.load(path)
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
                       graduation_time: float) -> Optional[Dict[str, float]]:
    """Detect V-shape pattern from on-chain 1m bars at T+10m.

    Args:
        kline_bars: List of bar dicts with keys: open, high, low, close, volume, time
        graduation_time: Unix timestamp of graduation

    Returns:
        Dict of vf_* features, or None if insufficient data.
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

    # Pre-entry bars (first 10 minutes)
    pre_entry = [b for b in post if b["offset"] <= 600]
    if len(pre_entry) < 3:
        return None

    # Entry price at T+10m
    entry_bars = [b for b in post if b["offset"] >= 600]  # T+10m, matches research 14i
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


def compute_micro_10m_live(swaps: List[Dict], graduation_time: float) -> Dict[str, float]:
    """Compute microstructure features from raw swaps in first 10 minutes.

    Args:
        swaps: List of swap dicts with keys: block_time, is_buy, sol_amount, trader_address
        graduation_time: Unix timestamp

    Returns:
        Dict of m10_* features.
    """
    if not swaps:
        return {}

    # Filter to first 10 minutes
    early = [s for s in swaps
             if 0 <= (s.get("block_time", 0) - graduation_time) < 600]
    if len(early) < 5:
        return {}

    buys = [s for s in early if s.get("is_buy")]
    sells = [s for s in early if not s.get("is_buy")]
    n_buys = len(buys)
    n_sells = len(sells)
    buy_vol = sum(s.get("sol_amount", 0) for s in buys)
    sell_vol = sum(s.get("sol_amount", 0) for s in sells)
    total_vol = buy_vol + sell_vol

    buyer_addrs = set(s.get("trader_address", "") for s in buys) - {""}
    seller_addrs = set(s.get("trader_address", "") for s in sells) - {""}
    unique_buyers = len(buyer_addrs)
    unique_sellers = len(seller_addrs)

    # Buyer growth velocity (per 2-minute window)
    seen: set = set()
    growth_windows = []
    for lo in range(0, 600, 120):
        w_buys = [s for s in buys if lo <= (s.get("block_time", 0) - graduation_time) < lo + 120]
        new_addrs = set(s.get("trader_address", "") for s in w_buys) - seen - {""}
        seen.update(new_addrs)
        growth_windows.append(len(new_addrs))

    late_growth = sum(growth_windows[3:]) if len(growth_windows) > 3 else 0
    early_growth = sum(growth_windows[:2]) if len(growth_windows) > 1 else 1
    growth_accel = late_growth / early_growth if early_growth > 0 else 0

    # Concentration
    if n_buys > 0:
        buyer_vol_map: Dict[str, float] = {}
        for s in buys:
            addr = s.get("trader_address", "")
            if addr:
                buyer_vol_map[addr] = buyer_vol_map.get(addr, 0) + s.get("sol_amount", 0)
        if buyer_vol_map:
            bv_total = sum(buyer_vol_map.values())
            shares_sq = sum((v / bv_total) ** 2 for v in buyer_vol_map.values())
            hhi = shares_sq
            sorted_vols = sorted(buyer_vol_map.values(), reverse=True)
            top1 = sorted_vols[0] / bv_total
            top3 = sum(sorted_vols[:3]) / bv_total
        else:
            hhi = top1 = top3 = 1.0
    else:
        hhi = top1 = top3 = 1.0

    imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0

    # Late imbalance
    l2_buys = [s for s in buys if (s.get("block_time", 0) - graduation_time) >= 480]
    l2_sells = [s for s in sells if (s.get("block_time", 0) - graduation_time) >= 480]
    l2_buy_vol = sum(s.get("sol_amount", 0) for s in l2_buys)
    l2_sell_vol = sum(s.get("sol_amount", 0) for s in l2_sells)
    late_imb = (l2_buy_vol - l2_sell_vol) / (l2_buy_vol + l2_sell_vol) if (l2_buy_vol + l2_sell_vol) > 0 else 0

    # Wash ratio
    union = buyer_addrs | seller_addrs
    wash = len(buyer_addrs & seller_addrs) / len(union) if union else 0

    unique_per_trade = unique_buyers / n_buys if n_buys > 0 else 0
    seller_buyer_ratio = unique_sellers / unique_buyers if unique_buyers > 0 else 0

    return {
        "m10_unique_buyers": unique_buyers,
        "m10_unique_sellers": unique_sellers,
        "m10_buyer_hhi": hhi,
        "m10_top1_buyer_share": top1,
        "m10_top3_buyer_share": top3,
        "m10_imbalance": imbalance,
        "m10_late_imbalance": late_imb,
        "m10_growth_accel": growth_accel,
        "m10_wash_ratio": wash,
        "m10_unique_per_trade": unique_per_trade,
        "m10_seller_buyer_ratio": seller_buyer_ratio,
        "m10_buy_vol_total": buy_vol,
        "m10_sell_vol_frac": sell_vol / total_vol if total_vol > 0 else 0,
    }


# ──────────────────────────────────────────────
# M3: Gateway Trader (direct REST API)
# ──────────────────────────────────────────────

class GatewayTrader:
    """Direct Gateway REST API calls for token registration + Jupiter swap."""

    def __init__(self, gateway_url: str, wallet_address: str,
                 connector: str = "jupiter/router",
                 chain_network: str = "solana-mainnet-beta",
                 slippage_pct: float = 2.0,
                 jupiter_api_key: str = ""):
        self.gateway_url = gateway_url.rstrip("/")
        self.wallet_address = wallet_address
        self.connector = connector
        self.chain_network = chain_network
        self.slippage_pct = slippage_pct
        self.jupiter_api_key = jupiter_api_key or os.environ.get("JUPITER_API_KEY", "")
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def _router_connector(self) -> str:
        return self.connector.split("/", 1)[0]

    @property
    def _router_network(self) -> str:
        parts = self.chain_network.split("-", 1)
        return parts[1] if len(parts) == 2 else self.chain_network

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(30.0))
        return self._client

    async def _post(self, path: str, body: dict) -> dict:
        client = await self._get_client()
        url = f"{self.gateway_url}{path}"
        resp = await client.post(url, json=body)
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(f"Gateway POST {path} failed ({resp.status_code}): {data}")
        return data

    async def _get(self, path: str, params: dict) -> dict:
        client = await self._get_client()
        url = f"{self.gateway_url}{path}"
        resp = await client.get(url, params=params)
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(f"Gateway GET {path} failed ({resp.status_code}): {data}")
        return data

    async def register_token(self, mint_address: str, symbol: str, name: str = "",
                             decimals: int = 6) -> dict:
        """POST /tokens/ — register new token in Gateway token list."""
        return await self._post("/tokens/", {
            "chain": "solana",
            "network": "mainnet-beta",
            "token": {
                "name": name or symbol,
                "symbol": symbol,
                "address": mint_address,
                "decimals": decimals,
            },
        })

    async def get_sol_balance(self) -> Optional[float]:
        """POST /chains/solana/balances — fetch native SOL balance from Gateway."""
        try:
            resp = await self._post("/chains/solana/balances", {
                "chain": "solana",
                "network": "mainnet-beta",
                "address": self.wallet_address,
                "tokenSymbols": ["SOL"],
            })
            balances = resp.get("balances", {})
            sol_str = balances.get("SOL", None)
            if sol_str is not None:
                return float(sol_str)
            return None
        except Exception as e:
            logger.warning(f"Failed to fetch SOL balance: {e}")
            return None

    async def get_sol_price_usd(self) -> float:
        """Get SOL/USDT price from Binance public API (free, no auth, no Jupiter quota)."""
        client = await self._get_client()
        resp = await client.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": "SOLUSDT"},
        )
        if resp.status_code == 200:
            return float(resp.json().get("price", 0))
        raise RuntimeError(f"Binance SOL price failed ({resp.status_code}): {resp.text}")

    async def swap_buy(self, mint_address: str, sol_amount: float) -> dict:
        """POST /trading/swap/execute — Jupiter: SOL → TOKEN (ExactIn).

        Uses base=SOL quote=TOKEN side=SELL so Jupiter uses ExactIn routing
        (spend exact SOL, receive variable TOKEN amount).
        """
        return await self._post("/trading/swap/execute", {
            "walletAddress": self.wallet_address,
            "chainNetwork": self.chain_network,
            "connector": self.connector,
            "baseToken": "SOL",
            "quoteToken": mint_address,
            "amount": str(sol_amount),
            "side": "SELL",
            "slippagePct": self.slippage_pct,
        })

    async def swap_sell(self, mint_address: str, token_amount: float,
                        slippage_override: Optional[float] = None) -> dict:
        """POST /trading/swap/execute — Jupiter: TOKEN → SOL (ExactIn).

        Uses base=TOKEN quote=SOL side=SELL so Jupiter uses ExactIn routing
        (spend exact TOKEN, receive variable SOL amount).
        """
        return await self._post("/trading/swap/execute", {
            "walletAddress": self.wallet_address,
            "chainNetwork": self.chain_network,
            "connector": self.connector,
            "baseToken": mint_address,
            "quoteToken": "SOL",
            "amount": str(token_amount),
            "side": "SELL",
            "slippagePct": slippage_override if slippage_override is not None else self.slippage_pct,
        })

    async def quote_buy(self, mint_address: str, sol_amount: float) -> dict:
        """GET raw router quote for SOL -> TOKEN.

        This returns the quote metadata from Gateway/Jupiter, including
        `quoteId`, `priceImpactPct`, and `routePlan`.
        """
        return await self._get(
            f"/connectors/{self._router_connector}/router/quote-swap",
            {
                "network": self._router_network,
                "baseToken": "SOL",
                "quoteToken": mint_address,
                "amount": str(sol_amount),
                "side": "SELL",
                "slippagePct": self.slippage_pct,
            },
        )

    async def execute_quote(self, quote_id: str) -> dict:
        """Execute a previously validated router quote."""
        return await self._post(
            f"/connectors/{self._router_connector}/router/execute-quote",
            {
                "network": self._router_network,
                "address": self.wallet_address,
                "quoteId": quote_id,
            },
        )

    @staticmethod
    def extract_quote_price_impact_pct(quote: dict) -> Optional[float]:
        """Return quote price impact as a fraction (0.05 = 5%) when available."""
        candidates = [
            quote.get("priceImpactPct"),
            quote.get("price_impact_pct"),
            quote.get("quoteResponse", {}).get("priceImpactPct"),
        ]
        for value in candidates:
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def extract_route_labels(quote: dict) -> List[str]:
        route_plan = quote.get("routePlan") or quote.get("quoteResponse", {}).get("routePlan") or []
        labels: List[str] = []
        for leg in route_plan:
            label = leg.get("swapInfo", {}).get("label")
            if label:
                labels.append(str(label))
        return labels

    @staticmethod
    def extract_route_amm_keys(quote: dict) -> List[str]:
        route_plan = quote.get("routePlan") or quote.get("quoteResponse", {}).get("routePlan") or []
        amm_keys: List[str] = []
        for leg in route_plan:
            amm_key = leg.get("swapInfo", {}).get("ammKey")
            if amm_key:
                amm_keys.append(str(amm_key))
        return amm_keys

    async def get_quote_price(self, mint_address: str, amount: float) -> Optional[float]:
        """GET /trading/swap/quote — get current TOKEN/SOL price via Gateway Jupiter quote.

        Uses base=TOKEN quote=SOL side=SELL to price the token in SOL terms.
        Returns amountOut/amountIn = SOL per TOKEN.
        NOTE: This still goes through Jupiter Quote API — prefer get_batch_prices() for monitoring.
        """
        try:
            resp = await self._get("/trading/swap/quote", {
                "chainNetwork": self.chain_network,
                "connector": self.connector,
                "baseToken": mint_address,
                "quoteToken": "SOL",
                "amount": str(amount),
                "side": "SELL",
            })
            return float(resp.get("price", 0))
        except Exception as e:
            logger.error(f"Quote failed for {mint_address}: {e}")
            return None

    async def get_batch_prices(self, mint_addresses: List[str],
                               sol_price_usd: float) -> Dict[str, Optional[float]]:
        """Batch-fetch TOKEN/SOL prices WITHOUT using Jupiter Quote API.

        Primary: Jupiter Price API v2 (separate service, not rate-limited with Quote API).
        Fallback: GeckoTerminal (free, no API key).
        Returns {mint_address: price_in_sol} for each token.
        """
        result: Dict[str, Optional[float]] = {m: None for m in mint_addresses}
        if not mint_addresses or sol_price_usd <= 0:
            return result

        # --- Primary: Jupiter Price API v3 (batch, 1 request for all tokens) ---
        # v2 was deprecated 2026-04 (returns 404). v3 is the current endpoint.
        # With API key: api.jup.ag, 60 req/min free tier.
        # Without key: lite-api.jup.ag (free, lower limit).
        # v3 response format: {mint: {"usdPrice": ..., "decimals": ..., ...}}
        try:
            client = await self._get_client()
            ids_str = ",".join(mint_addresses)
            if self.jupiter_api_key:
                jup_url = "https://api.jup.ag/price/v3"
                jup_headers = {"x-api-key": self.jupiter_api_key}
            else:
                jup_url = "https://lite-api.jup.ag/price/v3"
                jup_headers = {}
            resp = await client.get(
                jup_url,
                params={"ids": ids_str},
                headers=jup_headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()  # v3 returns flat dict, no "data" wrapper
                filled = 0
                for mint in mint_addresses:
                    token_data = data.get(mint)
                    if token_data and token_data.get("usdPrice"):
                        usd_price = float(token_data["usdPrice"])
                        if usd_price > 0:
                            result[mint] = usd_price / sol_price_usd
                            filled += 1
                if filled == len(mint_addresses):
                    return result
                # Some tokens missing — fall through to GeckoTerminal for the gaps
            else:
                logger.warning(f"Jupiter Price API returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Jupiter Price API failed: {e}")

        # --- Fallback: GeckoTerminal (free, no auth, max 30 addresses) ---
        missing = [m for m in mint_addresses if result[m] is None]
        if missing:
            try:
                client = await self._get_client()
                addresses_str = ",".join(missing)
                resp = await client.get(
                    f"https://api.geckoterminal.com/api/v2/simple/networks/solana/token_price/{addresses_str}",
                    headers={"Accept": "application/json;version=20230302"},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    attrs = resp.json().get("data", {}).get("attributes", {})
                    token_prices = attrs.get("token_prices", {})
                    for mint in missing:
                        price_str = token_prices.get(mint)
                        if price_str:
                            usd_price = float(price_str)
                            if usd_price > 0:
                                result[mint] = usd_price / sol_price_usd
                else:
                    logger.warning(f"GeckoTerminal returned {resp.status_code}")
            except Exception as e:
                logger.warning(f"GeckoTerminal fallback failed: {e}")

        return result

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ──────────────────────────────────────────────
# M5: Risk Manager
# ──────────────────────────────────────────────

class RiskManager:
    """Track daily P&L, consecutive losses, and total trade count.

    State is rebuilt from TradeDB on startup and auto-resets at UTC midnight.
    """

    def __init__(self, daily_loss_limit_usd: float = 30.0,
                 max_consecutive_losses: int = 10,
                 max_total_trades: int = 30,
                 max_positions: int = 3,
                 cooldown_sec: float = 60.0):
        self.daily_loss_limit_usd = daily_loss_limit_usd
        self.max_consecutive_losses = max_consecutive_losses
        self.max_total_trades = max_total_trades
        self.max_positions = max_positions
        self.cooldown_sec = cooldown_sec

        self.daily_pnl_usd: float = 0.0
        self.consecutive_losses: int = 0
        self.total_trades: int = 0
        self._last_trade_time: float = 0.0
        self._halted: bool = False
        self._halt_reason: str = ""
        self._current_date: str = ""  # YYYY-MM-DD UTC, for daily reset detection

    def rebuild_from_db(self, db: "TradeDB"):
        """Rebuild risk state from DB on startup. Called once in on_start()."""
        import datetime
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        self._current_date = today

        rows = db.get_trades_for_date(today)
        if not rows:
            logger.info(f"RiskManager: no trades today ({today}), state clean")
            return

        self.total_trades = len(rows)
        self.daily_pnl_usd = sum(r["pnl_usd"] for r in rows)

        # Consecutive losses: count backwards from most recent trade
        self.consecutive_losses = 0
        for r in reversed(rows):  # rows ordered by timestamp ASC
            if r["pnl_usd"] < 0:
                self.consecutive_losses += 1
            else:
                break

        # Last trade time
        if rows:
            self._last_trade_time = rows[-1]["timestamp"]

        # Re-check halt conditions
        if self.total_trades >= self.max_total_trades:
            self._halt("max_total_trades reached (rebuilt)")
        elif self.consecutive_losses >= self.max_consecutive_losses:
            self._halt(f"consecutive_losses={self.consecutive_losses} (rebuilt)")
        elif self.daily_pnl_usd <= -self.daily_loss_limit_usd:
            self._halt(f"daily_loss_limit: pnl={self.daily_pnl_usd:.2f} (rebuilt)")

        logger.info(f"RiskManager rebuilt from DB: date={today}, trades={self.total_trades}, "
                    f"daily_pnl=${self.daily_pnl_usd:.2f}, consec_losses={self.consecutive_losses}, "
                    f"halted={self._halted}")

    def _check_daily_reset(self):
        """Reset daily counters if UTC date has changed."""
        import datetime
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if self._current_date and today != self._current_date:
            logger.info(f"RiskManager: daily reset {self._current_date} → {today} "
                        f"(was: trades={self.total_trades}, pnl=${self.daily_pnl_usd:.2f})")
            self.daily_pnl_usd = 0.0
            self.consecutive_losses = 0
            self.total_trades = 0
            self._halted = False
            self._halt_reason = ""
            self._current_date = today

    def can_trade(self, active_positions: int) -> bool:
        self._check_daily_reset()
        if self._halted:
            return False
        if active_positions >= self.max_positions:
            return False
        if self.total_trades >= self.max_total_trades:
            self._halt("max_total_trades reached")
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._halt(f"consecutive_losses={self.consecutive_losses}")
            return False
        if self.daily_pnl_usd <= -self.daily_loss_limit_usd:
            self._halt(f"daily_loss_limit: pnl={self.daily_pnl_usd:.2f}")
            return False
        if time.time() - self._last_trade_time < self.cooldown_sec:
            return False
        return True

    def is_cooldown_only(self, active_positions: int) -> bool:
        """Return True if the ONLY reason can_trade() is False is the cooldown timer.
        Used to decide whether to queue a candidate (cooldown) vs discard it (hard block)."""
        self._check_daily_reset()
        if self._halted:
            return False
        if active_positions >= self.max_positions:
            return False
        if self.total_trades >= self.max_total_trades:
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        if self.daily_pnl_usd <= -self.daily_loss_limit_usd:
            return False
        # All hard checks pass — only cooldown could be blocking
        return time.time() - self._last_trade_time < self.cooldown_sec

    def record_trade(self, pnl_usd: float):
        self._check_daily_reset()
        self.total_trades += 1
        self.daily_pnl_usd += pnl_usd
        self._last_trade_time = time.time()
        if pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def _halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        logger.warning(f"RiskManager HALTED: {reason}")

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def status(self) -> dict:
        return {
            "daily_pnl_usd": round(self.daily_pnl_usd, 4),
            "consecutive_losses": self.consecutive_losses,
            "total_trades": self.total_trades,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "current_date": self._current_date,
        }


# ──────────────────────────────────────────────
# Postgres Telemetry Sink
# ──────────────────────────────────────────────

class PostgresTelemetrySink:
    """Route high-frequency telemetry to Postgres instead of SQLite."""

    DDL_LATENCY_EVENTS = """
    CREATE TABLE IF NOT EXISTS latency_events (
        id BIGSERIAL PRIMARY KEY,
        source_sqlite_id BIGINT UNIQUE,
        timestamp DOUBLE PRECISION NOT NULL,
        mint_address TEXT NOT NULL,
        symbol TEXT,
        graduation_time DOUBLE PRECISION,
        event_name TEXT NOT NULL,
        age_sec DOUBLE PRECISION,
        metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    DDL_SHADOW_POLICY_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_policy_evals (
        id BIGSERIAL PRIMARY KEY,
        source_sqlite_id BIGINT UNIQUE,
        timestamp DOUBLE PRECISION NOT NULL,
        obs_id BIGINT,
        mint_address TEXT NOT NULL,
        symbol TEXT NOT NULL,
        graduation_time DOUBLE PRECISION,
        feature_delay_sec INTEGER,
        stage1_name TEXT,
        stage1_score DOUBLE PRECISION,
        stage1_threshold DOUBLE PRECISION,
        stage1_passed INTEGER,
        stage2_rule_name TEXT,
        stage2_passed INTEGER,
        feature_source TEXT,
        feature_view TEXT,
        range_1to3m DOUBLE PRECISION,
        drawdown_1to3m DOUBLE PRECISION,
        total_volume_3m DOUBLE PRECISION,
        total_trades_3m DOUBLE PRECISION,
        metadata JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """

    DDL_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_latency_events_mint ON latency_events(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_latency_events_name ON latency_events(event_name)",
        "CREATE INDEX IF NOT EXISTS idx_latency_events_ts ON latency_events(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_policy_evals_mint ON shadow_policy_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_policy_evals_rule ON shadow_policy_evals(stage2_rule_name)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_policy_evals_ts ON shadow_policy_evals(timestamp)",
    ]

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._lock = threading.RLock()
        self._conn = None
        self._psycopg2 = None
        self._connect()
        self._init_schema()

    def _connect(self):
        import psycopg2
        from psycopg2.extras import Json

        conn = psycopg2.connect(self._dsn, connect_timeout=5)
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute("SET application_name = 'meme_sniper_telemetry'")
        conn.commit()
        self._psycopg2 = psycopg2
        self._Json = Json
        self._conn = conn

    def _ensure_conn(self):
        if self._conn is None or getattr(self._conn, "closed", 1):
            self._connect()

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            last_error = None
            for attempt in range(2):
                try:
                    self._ensure_conn()
                    with self._conn.cursor() as cur:
                        cur.execute(sql, params)
                    self._conn.commit()
                    return
                except Exception as e:
                    last_error = e
                    try:
                        if self._conn is not None:
                            self._conn.rollback()
                    except Exception:
                        pass
                    try:
                        if self._conn is not None:
                            self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt == 0:
                        continue
                    raise last_error

    def _init_schema(self):
        self._exec(self.DDL_LATENCY_EVENTS)
        self._exec(self.DDL_SHADOW_POLICY_EVALS)
        for stmt in self.DDL_INDEXES:
            self._exec(stmt)

    def record_latency_event(
        self,
        mint_address: str,
        symbol: str,
        graduation_time: Optional[float],
        event_name: str,
        age_sec: Optional[float] = None,
        metadata: Optional[Dict] = None,
        source_sqlite_id: Optional[int] = None,
    ):
        self._exec(
            """INSERT INTO latency_events
               (source_sqlite_id, timestamp, mint_address, symbol, graduation_time, event_name, age_sec, metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (source_sqlite_id) DO NOTHING""",
            (
                source_sqlite_id,
                time.time(),
                mint_address,
                symbol,
                graduation_time,
                event_name,
                age_sec,
                self._Json(metadata) if metadata is not None else None,
            ),
        )

    def record_shadow_policy_eval(
        self,
        obs_id: Optional[int],
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
        metadata: Optional[Dict] = None,
        source_sqlite_id: Optional[int] = None,
    ):
        self._exec(
            """INSERT INTO shadow_policy_evals
               (source_sqlite_id, timestamp, obs_id, mint_address, symbol, graduation_time, feature_delay_sec,
                stage1_name, stage1_score, stage1_threshold, stage1_passed,
                stage2_rule_name, stage2_passed, feature_source, feature_view,
                range_1to3m, drawdown_1to3m, total_volume_3m, total_trades_3m, metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (source_sqlite_id) DO NOTHING""",
            (
                source_sqlite_id,
                time.time(),
                obs_id,
                mint_address,
                symbol,
                graduation_time,
                feature_delay_sec,
                stage1_name,
                stage1_score,
                stage1_threshold,
                int(stage1_passed) if stage1_passed is not None else None,
                stage2_rule_name,
                int(stage2_passed) if stage2_passed is not None else None,
                feature_source,
                feature_view,
                range_1to3m,
                drawdown_1to3m,
                total_volume_3m,
                total_trades_3m,
                self._Json(metadata) if metadata is not None else None,
            ),
        )

    def close(self):
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


# ──────────────────────────────────────────────
# Trade Database (SQLite)
# ──────────────────────────────────────────────

class TradeDB:
    """Persist trades and discoveries to SQLite for post-session analysis."""

    DDL_TRADES = """
    CREATE TABLE IF NOT EXISTS trades (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       REAL    NOT NULL,
        token_symbol    TEXT    NOT NULL,
        mint_address    TEXT    NOT NULL,
        entry_price_sol REAL,
        exit_price_sol  REAL,
        token_amount    REAL,
        sol_invested    REAL,
        sol_received    REAL,
        pnl_sol         REAL,
        pnl_usd         REAL,
        hold_seconds    REAL,
        exit_reason     TEXT,
        entry_tx        TEXT,
        exit_tx         TEXT,
        p_alive         REAL,
        features        TEXT
    )
    """

    DDL_DISCOVERIES = """
    CREATE TABLE IF NOT EXISTS discoveries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       REAL    NOT NULL,
        mint_address    TEXT    NOT NULL,
        symbol          TEXT    NOT NULL,
        liquidity_usd   REAL,
        price_usd       REAL,
        p_alive         REAL,
        passed_filter   INTEGER NOT NULL DEFAULT 0,
        reject_reason   TEXT
    )
    """

    DDL_EVENTS = """
    CREATE TABLE IF NOT EXISTS events (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL    NOT NULL,
        level     TEXT    NOT NULL,
        module    TEXT,
        message   TEXT    NOT NULL
    )
    """

    DDL_LATENCY_EVENTS = """
    CREATE TABLE IF NOT EXISTS latency_events (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       REAL    NOT NULL,
        mint_address    TEXT    NOT NULL,
        symbol          TEXT,
        graduation_time REAL,
        event_name      TEXT    NOT NULL,
        age_sec         REAL,
        metadata        TEXT
    )
    """

    DDL_LATENCY_EVENTS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_latency_mint ON latency_events(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_latency_event ON latency_events(event_name)",
        "CREATE INDEX IF NOT EXISTS idx_latency_ts ON latency_events(timestamp)",
    ]

    DDL_SWAPS = """
    CREATE TABLE IF NOT EXISTS swaps (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        mint_address    TEXT    NOT NULL,
        pool_address    TEXT    NOT NULL,
        timestamp       INTEGER NOT NULL,
        price_sol       REAL    NOT NULL,
        volume_sol      REAL    NOT NULL,
        is_buy          INTEGER NOT NULL,
        base_amount     REAL    NOT NULL,
        collected_at    REAL    NOT NULL,
        trader_address  TEXT    DEFAULT ''
    )
    """

    DDL_SWAPS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_swaps_mint ON swaps(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_swaps_ts ON swaps(mint_address, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_swaps_trader ON swaps(trader_address)",
    ]

    DDL_TRADER_RECORDS = """
    CREATE TABLE IF NOT EXISTS trader_records (
        trader_address  TEXT    PRIMARY KEY,
        wins            INTEGER NOT NULL DEFAULT 0,
        losses          INTEGER NOT NULL DEFAULT 0,
        total_volume    REAL    NOT NULL DEFAULT 0,
        last_updated    REAL    NOT NULL DEFAULT 0
    )
    """

    DDL_OPEN_POSITIONS = """
    CREATE TABLE IF NOT EXISTS open_positions (
        mint_address        TEXT    PRIMARY KEY,
        symbol              TEXT    NOT NULL,
        name                TEXT    NOT NULL,
        decimals            INTEGER NOT NULL DEFAULT 6,
        graduation_time     REAL    NOT NULL,
        liquidity_usd       REAL,
        price_usd           REAL,
        entry_price_sol     REAL    NOT NULL,
        token_amount        REAL    NOT NULL,
        entry_time          REAL    NOT NULL,
        entry_tx            TEXT,
        sol_invested        REAL    NOT NULL,
        p_alive             REAL,
        features            TEXT,
        peak_pnl_pct        REAL    NOT NULL DEFAULT 0,
        trailing_activated  INTEGER NOT NULL DEFAULT 0
    )
    """

    MIGRATE_OPEN_POSITIONS_V2 = [
        "ALTER TABLE open_positions ADD COLUMN peak_pnl_pct REAL NOT NULL DEFAULT 0",
        "ALTER TABLE open_positions ADD COLUMN trailing_activated INTEGER NOT NULL DEFAULT 0",
    ]

    MIGRATE_OPEN_POSITIONS_V3 = [
        "ALTER TABLE open_positions ADD COLUMN pool_address TEXT",
        "ALTER TABLE open_positions ADD COLUMN source TEXT NOT NULL DEFAULT 'chainstack'",
    ]

    MIGRATE_SWAPS_V2 = [
        "ALTER TABLE swaps ADD COLUMN trader_address TEXT DEFAULT ''",
    ]

    MIGRATE_TRADES_V2 = [
        "ALTER TABLE trades ADD COLUMN entry_time REAL",
        "ALTER TABLE trades ADD COLUMN peak_pnl_pct REAL",
        "ALTER TABLE trades ADD COLUMN sol_price_usd REAL",
        "ALTER TABLE trades ADD COLUMN trigger_pnl_pct REAL",
    ]

    # V3: execution-quality fields. Needed to diagnose the entry-chase /
    # preflight-latency / pool-depth issues that drove the 15-trade -$17.38.
    MIGRATE_TRADES_V3 = [
        "ALTER TABLE trades ADD COLUMN m2_ref_price_sol REAL",
        "ALTER TABLE trades ADD COLUMN preflight_latency_ms INTEGER",
        "ALTER TABLE trades ADD COLUMN pool_liq_at_entry_usd REAL",
    ]

    MIGRATE_DISCOVERIES_V2 = [
        "ALTER TABLE discoveries ADD COLUMN features TEXT",
    ]

    MIGRATE_OBSERVATIONS_V2 = [
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_entry TEXT",
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_60m TEXT",
        "ALTER TABLE token_observations ADD COLUMN gmgn_security TEXT",
    ]

    MIGRATE_OBSERVATIONS_V3 = [
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_15m TEXT",
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_30m TEXT",
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_45m TEXT",
    ]

    # V4: early GMGN snapshots at ~30s after graduation ("birth certificate").
    # These capture dev history, holder structure, wallet composition, and
    # security flags before the token has had time to trade significantly.
    # For analysis of entry quality — whether the token was a scam from the
    # start — these t0 fields are more informative than t=180s snapshots.
    MIGRATE_OBSERVATIONS_V4 = [
        "ALTER TABLE token_observations ADD COLUMN gmgn_info_t0 TEXT",
        "ALTER TABLE token_observations ADD COLUMN gmgn_security_t0 TEXT",
    ]

    # V5: DexScreener snapshot at t0. Orthogonal to GMGN — captures
    # boosts/paid-promotion, social link presence, dex-side liquidity/volume.
    # Used later by rug_filter_v2 once enough samples accumulate.
    MIGRATE_OBSERVATIONS_V5 = [
        "ALTER TABLE token_observations ADD COLUMN dexscr_info_t0 TEXT",
    ]

    # V6: pre-computed hypothetical PnL at multiple SL thresholds, using the
    # 60min kline captured in kline_30m. Lets us instantly answer "if I had
    # traded this token with SL=X%, what would have happened?" without
    # re-running ad-hoc simulation scripts.
    # Computed once in complete_observation(), stored here for fast queries.
    MIGRATE_OBSERVATIONS_V6 = [
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl25_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl30_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl40_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_peak_pnl_pct REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_exit_reason_sl25 TEXT",
        "ALTER TABLE token_observations ADD COLUMN sim_computed_at REAL",
    ]

    # V7: swap-level PnL sim. Validated against 20 real trades: mean bias
    # -$0.19 (vs kline-sim -$1.33) and median abs gap $1.16 (vs kline $2.64).
    # Uses the raw `swaps` table (same on-chain PumpSwap feed that built
    # kline_30m) but tick-level, so captures sub-minute rugs/peaks that
    # 1-min OHLC aggregation smooths over. This is the ground-truth sim
    # for all shadow-model analysis going forward.
    MIGRATE_OBSERVATIONS_V7 = [
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl25_swap_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl30_swap_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl40_swap_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_peak_pnl_pct_swap REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_exit_reason_sl25_swap TEXT",
        "ALTER TABLE token_observations ADD COLUMN sim_swap_n_ticks INTEGER",
        "ALTER TABLE token_observations ADD COLUMN sim_computed_at_swap REAL",
    ]

    # V8: realistic-cost sim. Takes swap-sim output and applies measured
    # execution costs (calibrated from 20 real trades, 2026-04-20):
    #   - entry_chase: linear 10% @ grad+90s → 48% @ grad+600s
    #   - trailing exit slip: 5% median
    #   - SL cascade: +40% extra loss when SL triggers (rug completes)
    #   - Jupiter routing gap: 3% constant
    # Validated: M2 PASS subset realistic sim -$1.71/trade vs real -$1.44/trade
    # (gap $0.27 vs $1.44 raw swap-sim gap). This is the trust-worthy label
    # for training models and comparing shadow performance.
    MIGRATE_OBSERVATIONS_V8 = [
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl25_realistic_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl30_realistic_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_pnl_sl40_realistic_usd REAL",
        "ALTER TABLE token_observations ADD COLUMN sim_realistic_cost_version TEXT",
        "ALTER TABLE token_observations ADD COLUMN sim_computed_at_realistic REAL",
    ]

    # 2026-04-24: parallel A/B with bot-schema retrain (swap_exit_f2a_hc_live_v1.pkl).
    # See Phase_14y_Enhancement_FROZEN_2026-04-24.md §9.9.
    MIGRATE_SHADOW_EXIT_WARN_EVALS_V2 = [
        "ALTER TABLE shadow_exit_warn_evals ADD COLUMN p_rug_live_v1 REAL",
    ]

    DDL_TOKEN_OBSERVATIONS = """
    CREATE TABLE IF NOT EXISTS token_observations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        mint_address    TEXT    NOT NULL,
        symbol          TEXT    NOT NULL,
        graduation_time REAL    NOT NULL,
        liquidity_usd   REAL,
        price_usd       REAL,
        source          TEXT,
        pool_address    TEXT,
        model_score     REAL,
        model_passed    INTEGER,
        reject_reason   TEXT,
        features        TEXT,
        kline_6m        TEXT,
        kline_30m       TEXT,
        trade_pnl_usd   REAL,
        trade_exit_reason TEXT,
        observed_at     REAL    NOT NULL,
        completed_at    REAL
    )
    """

    DDL_TOKEN_OBSERVATIONS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_obs_mint ON token_observations(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_obs_grad ON token_observations(graduation_time)",
    ]

    # Hot rank cross-check (M1 coverage analysis)
    DDL_HOT_RANK_OBSERVATIONS = """
    CREATE TABLE IF NOT EXISTS hot_rank_observations (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        observed_at        REAL    NOT NULL,
        mint_address       TEXT    NOT NULL,
        symbol             TEXT,
        age_sec            INTEGER,
        liquidity_usd      REAL,
        volume_5m_usd      REAL,
        swaps_5m           INTEGER,
        smart_degen_count  INTEGER,
        rank_position      INTEGER,
        match_status       TEXT    NOT NULL,
        m1_first_seen      REAL,
        m1_passed          INTEGER
    )
    """
    DDL_HOT_RANK_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_hotrank_mint ON hot_rank_observations(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_hotrank_obs ON hot_rank_observations(observed_at)",
        "CREATE INDEX IF NOT EXISTS idx_hotrank_status ON hot_rank_observations(match_status)",
    ]

    DDL_SHADOW_POLICY_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_policy_evals (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           REAL    NOT NULL,
        obs_id              INTEGER,
        mint_address        TEXT    NOT NULL,
        symbol              TEXT    NOT NULL,
        graduation_time     REAL,
        feature_delay_sec   INTEGER,
        stage1_name         TEXT,
        stage1_score        REAL,
        stage1_threshold    REAL,
        stage1_passed       INTEGER,
        stage2_rule_name    TEXT,
        stage2_passed       INTEGER,
        feature_source      TEXT,
        feature_view        TEXT,
        range_1to3m         REAL,
        drawdown_1to3m      REAL,
        total_volume_3m     REAL,
        total_trades_3m     REAL,
        metadata            TEXT
    )
    """

    DDL_SHADOW_POLICY_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_eval_mint ON shadow_policy_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_eval_obs ON shadow_policy_evals(obs_id)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_eval_ts ON shadow_policy_evals(timestamp)",
    ]

    # Rug filter v1 shadow: one row per M2 promotion where rug filter was
    # scored. would_reject indicates if the token is in the predicted top-N%
    # most rug-likely (reject in a combined Flow+Rug deployment). Joins
    # back to token_observations for actual-outcome (rug label) analysis.
    DDL_SHADOW_RUG_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_rug_evals (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp          REAL    NOT NULL,
        mint_address       TEXT    NOT NULL,
        symbol             TEXT,
        graduation_time    REAL,
        snapshot_delay_sec REAL,
        score              REAL,
        cutoff_band        TEXT,
        cutoff_value       REAL,
        would_reject       INTEGER,
        model_version      TEXT,
        features           TEXT
    )
    """
    DDL_SHADOW_RUG_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_rug_mint ON shadow_rug_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_rug_ts ON shadow_rug_evals(timestamp)",
    ]

    # Rug filter v3 shadow: one row per graduation where v3 scored at T+60s
    # from bot's own swaps table. decision is REJECT | PASS | SKIP_NO_DATA |
    # SKIP_MODEL_ERROR. actual_pnl / actual_rug are backfilled by the
    # resolver once the trade outcome is known, enabling offline
    # precision/recall analysis without rerunning the feature extractor.
    DDL_SHADOW_RUG_V3_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_rug_filter_v3_evals (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp          REAL    NOT NULL,
        mint_address       TEXT    NOT NULL,
        symbol             TEXT,
        graduation_time    REAL,
        scored_at_delay_s  REAL,
        window_s           INTEGER,
        n_swaps            INTEGER,
        v3_score           REAL,
        cutoff             REAL,
        decision           TEXT,
        reason             TEXT,
        creator            TEXT,
        features           TEXT,
        model_version      TEXT,
        actual_pnl_usd     REAL,
        actual_exit_reason TEXT,
        actual_rug         INTEGER,
        resolver_ran_at    REAL
    )
    """
    DDL_SHADOW_RUG_V3_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_rugv3_mint ON shadow_rug_filter_v3_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_rugv3_ts ON shadow_rug_filter_v3_evals(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_rugv3_grad ON shadow_rug_filter_v3_evals(graduation_time)",
    ]

    # Phase B shadow: event-triggered first-trigger log. One row per (mint,
    # scan_t) where (pattern AND score >= cutoff). Offline analysis joins
    # against swaps table to reconstruct hypothetical PnL.
    DDL_SHADOW_EVENT_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_event_evals (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         REAL    NOT NULL,
        mint_address      TEXT    NOT NULL,
        symbol            TEXT,
        graduation_time   REAL,
        scan_t_sec        INTEGER NOT NULL,
        cutoff_band       TEXT,
        cutoff_value      REAL,
        score             REAL,
        pattern           TEXT,
        entry_price_sol   REAL,
        sol_price_usd     REAL,
        model_version     TEXT,
        features          TEXT
    )
    """
    DDL_SHADOW_EVENT_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_event_mint ON shadow_event_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_event_ts ON shadow_event_evals(timestamp)",
    ]

    # Per-check M3 preflight outcomes. One row per (token, check_name).
    # Records ALL outcomes — pass, block, skip — so we can later analyze
    # which checks actually prevent catastrophic losses, how often each
    # fires, and the distribution of measured values (price_impact,
    # entry_chase, divergence etc) at decision time.
    #
    # Currently `events` only logs coarse BUY BLOCKED text reasons. This
    # table gives structured rows for research.
    DDL_PREFLIGHT_CHECKS = """
    CREATE TABLE IF NOT EXISTS preflight_checks (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        REAL    NOT NULL,
        mint_address     TEXT    NOT NULL,
        symbol           TEXT,
        check_name       TEXT    NOT NULL,
        outcome          TEXT    NOT NULL,
        value            REAL,
        threshold        REAL,
        detail           TEXT,
        model_score      REAL,
        pool_liq_usd     REAL
    )
    """
    DDL_PREFLIGHT_CHECKS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_preflight_mint ON preflight_checks(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_preflight_ts ON preflight_checks(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_preflight_check ON preflight_checks(check_name, outcome)",
    ]

    # Rug events detected via gRPC stream (large SOL-out swaps on watched pools).
    # Each row = one dump transaction that met the threshold. `action_taken`
    # tells us what the controller did (panic_sell, no_position=race, etc.)
    # For post-hoc analysis: compare M4's poll-based SL trigger time vs
    # gRPC's block-level detection time to measure latency savings.
    DDL_RUG_EVENTS = """
    CREATE TABLE IF NOT EXISTS rug_events (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           REAL    NOT NULL,
        mint_address        TEXT    NOT NULL,
        pool_address        TEXT,
        sol_amount          REAL,
        price_sol           REAL,
        detected_at         REAL,
        action_taken        TEXT,
        signature           TEXT,
        triggered_pnl_pct   REAL
    )
    """
    DDL_RUG_EVENTS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_rug_events_mint ON rug_events(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_rug_events_ts ON rug_events(timestamp)",
    ]

    # Event Invariant v1 Model B shadow evals (Phase B research validation).
    # Narrow T+3-5m scan window, top5 cutoff — the only combo that produced
    # positive first-trigger EV in holdout (30 tokens, 73% win, +$12.68).
    # Schema identical to shadow_event_evals for easy cross-comparison.
    DDL_SHADOW_EVENT_INVARIANT_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_event_invariant_evals (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp         REAL    NOT NULL,
        mint_address      TEXT    NOT NULL,
        symbol            TEXT,
        graduation_time   REAL,
        scan_t_sec        INTEGER NOT NULL,
        cutoff_band       TEXT,
        cutoff_value      REAL,
        score             REAL,
        pattern           TEXT,
        entry_price_sol   REAL,
        sol_price_usd     REAL,
        model_version     TEXT,
        features          TEXT
    )
    """
    DDL_SHADOW_EVENT_INVARIANT_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_ei_mint ON shadow_event_invariant_evals(mint_address)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_ei_ts ON shadow_event_invariant_evals(timestamp)",
    ]

    # Forensic dual-price log (BK 2026-04-14 postmortem). Writes one row per
    # open position per monitor cycle — captures both gRPC pool median (the
    # ground truth) and Jupiter Price API value, so we can reconstruct
    # divergence events and exit-execution gaps after the fact.
    DDL_PRICE_PROBES = """
    CREATE TABLE IF NOT EXISTS price_probes (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           REAL    NOT NULL,
        mint_address        TEXT    NOT NULL,
        symbol              TEXT,
        hold_sec            REAL,
        entry_price_sol     REAL,
        pool_price_sol      REAL,
        jupiter_price_usd   REAL,
        sol_price_usd       REAL,
        divergence_pct      REAL,
        price_source        TEXT,
        pool_pnl_pct        REAL,
        jup_pnl_pct         REAL,
        peak_pnl_pct        REAL,
        trailing_activated  INTEGER
    )
    """

    DDL_PRICE_PROBES_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_probes_mint ON price_probes(mint_address, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_probes_ts ON price_probes(timestamp)",
    ]

    # 14y V9 shadow-exit-warn: per-position per-tick predictions from the two
    # frozen exit models (Tier B drop-risk, F2a+HC rug-risk). OBSERVE_ONLY;
    # rows are resolved 60s later with realized outcomes. See
    # reports/Phase8_14y_Shadow_Wiring_Spec.md.
    DDL_SHADOW_EXIT_WARN_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_exit_warn_evals (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp               REAL    NOT NULL,   -- wall-clock
        position_id             INTEGER,            -- FK trades.id (NULL if still open)
        mint_address            TEXT    NOT NULL,
        symbol                  TEXT,
        dt_from_entry           INTEGER NOT NULL,   -- seconds since entry
        dt_from_grad            INTEGER,            -- seconds since graduation (for grid comparison)
        p_drop_raw              REAL    NOT NULL,   -- Tier B output (trained directly on drop target)
        p_rug_raw               REAL    NOT NULL,   -- F2a+HC output (NOT calibrated, miscalibrated by scale_pos_weight)
        mid_price_sol           REAL,               -- current smoothed mid at decision time
        n_cache_swaps           INTEGER,            -- size of swap-cache slice used for features
        feature_window_count    INTEGER,            -- swaps in [t-180, t]
        resolved                INTEGER DEFAULT 0,  -- flipped to 1 once 60s has elapsed
        realized_drop_60        REAL,               -- populated at resolution
        realized_max_sell_60    REAL,               -- populated at resolution
        y_drop_actual           INTEGER,            -- populated at resolution (drop<-15%)
        y_rug_actual            INTEGER,            -- populated at resolution (sell>=10 SOL)
        features_json           TEXT                -- optional raw features snapshot for audit
    )
    """
    DDL_SHADOW_EXIT_WARN_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_ew_mint ON shadow_exit_warn_evals(mint_address, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_ew_unresolved ON shadow_exit_warn_evals(resolved, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_ew_position ON shadow_exit_warn_evals(position_id)",
    ]

    # 19c V10 shadow_sw_v2_evals: per-graduation F1 (sw_v2_f1_v2) entry score
    # at T+180s, OBSERVE_ONLY. Resolved 15 min later with realized forward
    # return. See reports/Phase8_19c_SW_v2_Shadow_Wiring_Spec.md.
    DDL_SHADOW_SW_V2_EVALS = """
    CREATE TABLE IF NOT EXISTS shadow_sw_v2_evals (
        id                         INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp                  REAL    NOT NULL,   -- wall-clock at scoring time
        mint_address               TEXT    NOT NULL,
        symbol                     TEXT,
        graduation_time            REAL    NOT NULL,
        dt_from_grad               INTEGER NOT NULL,   -- seconds since grad (≈ 180)

        -- F1 inputs and output
        entry_offset_sec           REAL,               -- actual bar_start_offset_sec used
        entry_price_sol            REAL,
        grad_price_sol             REAL,
        n_bars_pre_entry           INTEGER,
        n_swaps_used               INTEGER,
        feature_ready              INTEGER,            -- 1 if predict returned a score
        f1_score                   REAL,               -- probability output (nullable on failure)
        features_json              TEXT,               -- full 45-feature dict for audit

        -- Contextual
        v14_candidate              INTEGER DEFAULT 0,
        v14_entered                INTEGER DEFAULT 0,
        v14_score                  REAL,
        v14_pattern                TEXT,
        rug_filter_score           REAL,

        -- Outcome (backfilled by resolver at T+18m)
        resolved                   INTEGER DEFAULT 0,
        realized_raw_15m           REAL,               -- raw return from entry to T+18m
        y_net15_emp_trail_actual   INTEGER,            -- (raw_15m − 0.063) > 0
        peak_ret_15m               REAL,
        trough_ret_15m             REAL,
        eventual_trade_pnl_usd     REAL                -- if v14 traded this token later
    )
    """
    DDL_SHADOW_SW_V2_EVALS_IDX = [
        "CREATE INDEX IF NOT EXISTS idx_shadow_swv2_mint ON shadow_sw_v2_evals(mint_address, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_shadow_swv2_unresolved ON shadow_sw_v2_evals(resolved, timestamp)",
    ]

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(
                os.environ.get("PROJECT_DIR", "/home/hummingbot"),
                "data", "meme_sniper_trades.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path  # exposed for read-only consumers (e.g. rug_filter_v3)
        # sqlite3 with check_same_thread=False requires explicit user-side
        # serialization. Without a lock, concurrent controller callbacks can
        # interleave writes on the shared connection and destabilize the file.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._recent_sell_triggers: Dict[Tuple[str, str], float] = {}
        # WAL mode corrupts repeatedly under Docker overlayfs (3 incidents in 2 days).
        # Switch to DELETE journal mode: slower writes but no WAL/SHM files that
        # overlayfs can desync. Combined with synchronous=FULL for maximum safety.
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=DELETE")
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
            self._conn.execute(self.DDL_SHADOW_RUG_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_EXIT_WARN_EVALS)
            for idx_stmt in self.DDL_SHADOW_EXIT_WARN_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_SW_V2_EVALS)
            for idx_stmt in self.DDL_SHADOW_SW_V2_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._conn.execute(self.DDL_SHADOW_RUG_V3_EVALS)
            for idx_stmt in self.DDL_SHADOW_RUG_V3_EVALS_IDX:
                self._conn.execute(idx_stmt)
            self._migrate_open_positions()
            self._migrate_table("trades", self.MIGRATE_TRADES_V2)
            self._migrate_table("trades", self.MIGRATE_TRADES_V3)
            self._migrate_table("discoveries", self.MIGRATE_DISCOVERIES_V2)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V2)
            self._migrate_table("swaps", self.MIGRATE_SWAPS_V2)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V3)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V4)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V5)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V6)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V7)
            self._migrate_table("token_observations", self.MIGRATE_OBSERVATIONS_V8)
            self._migrate_table("shadow_exit_warn_evals", self.MIGRATE_SHADOW_EXIT_WARN_EVALS_V2)
            self._conn.commit()
        self._telemetry_sink: Optional[PostgresTelemetrySink] = None
        telemetry_dsn = os.environ.get("TELEMETRY_DATABASE_URL", "").strip()
        if telemetry_dsn:
            try:
                self._telemetry_sink = PostgresTelemetrySink(telemetry_dsn)
                logger.info("TradeDB telemetry sink enabled: PostgreSQL")
            except Exception as e:
                logger.warning(f"TradeDB telemetry sink unavailable: {e}")
        logger.info(f"TradeDB initialized: {db_path}")

    def _should_skip_sell_trigger(self, mint_address: str, metadata: Optional[Dict]) -> bool:
        if not metadata:
            return False
        reason = str(metadata.get("reason") or "").strip()
        if not reason:
            return False
        now = time.time()
        key = (mint_address, reason)
        last_ts = self._recent_sell_triggers.get(key)
        if last_ts is not None and (now - last_ts) < 60:
            return True
        self._recent_sell_triggers[key] = now
        if len(self._recent_sell_triggers) > 256:
            cutoff = now - 120
            self._recent_sell_triggers = {
                k: ts for k, ts in self._recent_sell_triggers.items() if ts >= cutoff
            }
        return False

    def _fetchall(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _fetchone(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _execute_commit(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _executemany_commit(self, sql: str, seq_of_params):
        with self._lock:
            self._conn.executemany(sql, seq_of_params)
            self._conn.commit()

    def _migrate_open_positions(self):
        """Add trailing stop columns (v2) and pool_address/source (v3) if missing."""
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V2)
        self._migrate_table("open_positions", self.MIGRATE_OPEN_POSITIONS_V3)

    def _migrate_table(self, table: str, stmts: list):
        """Generic migration: add columns if missing."""
        cols = {row[1] for row in self._fetchall(f"PRAGMA table_info({table})")}
        for stmt in stmts:
            col_name = stmt.split("ADD COLUMN ")[1].split()[0]
            if col_name not in cols:
                try:
                    with self._lock:
                        self._conn.execute(stmt)
                        logger.info(f"TradeDB migration: {table}.{col_name} added")
                except Exception:
                    pass

    def record_trade(self, record: "TradeRecord", features: Optional[Dict] = None):
        """Insert a completed trade."""
        self._execute_commit(
            """INSERT INTO trades
               (timestamp, token_symbol, mint_address, entry_price_sol, exit_price_sol,
                token_amount, sol_invested, sol_received, pnl_sol, pnl_usd,
                hold_seconds, exit_reason, entry_tx, exit_tx, p_alive, features,
                entry_time, peak_pnl_pct, sol_price_usd, trigger_pnl_pct,
                m2_ref_price_sol, preflight_latency_ms, pool_liq_at_entry_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (record.timestamp, record.token_symbol, record.mint_address,
             record.entry_price_sol, record.exit_price_sol,
             record.token_amount, record.sol_invested, record.sol_received,
             record.pnl_sol, record.pnl_usd, record.hold_seconds,
             record.exit_reason, record.entry_tx, record.exit_tx,
             record.model_score, json.dumps(features) if features else None,
             record.entry_time, record.peak_pnl_pct,
             record.sol_price_usd, record.trigger_pnl_pct,
             record.m2_ref_price_sol, record.preflight_latency_ms,
             record.pool_liq_at_entry_usd))

    def record_hot_rank_observation(self, mint: str, symbol: str, age_sec: int,
                                     liquidity_usd: float, volume_5m: float,
                                     swaps_5m: int, smart_degen_count: int,
                                     rank_position: int, match_status: str,
                                     m1_first_seen: Optional[float] = None,
                                     m1_passed: Optional[bool] = None):
        """Insert a hot-rank cross-check observation.

        match_status: 'matched' (M1 also saw it) / 'm1_missed' (M1 didn't see it)
        """
        self._execute_commit(
            """INSERT INTO hot_rank_observations
               (observed_at, mint_address, symbol, age_sec, liquidity_usd,
                volume_5m_usd, swaps_5m, smart_degen_count, rank_position,
                match_status, m1_first_seen, m1_passed)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint, symbol, int(age_sec), liquidity_usd,
             volume_5m, int(swaps_5m), int(smart_degen_count), int(rank_position),
             match_status, m1_first_seen,
             int(m1_passed) if m1_passed is not None else None))

    def lookup_discovery(self, mint: str) -> Optional[Tuple[float, int]]:
        """Return (first_seen_timestamp, passed_filter) for a mint, or None if not found."""
        row = self._fetchone(
            "SELECT MIN(timestamp), MAX(passed_filter) FROM discoveries WHERE mint_address = ?",
            (mint,))
        if row and row[0] is not None:
            return (float(row[0]), int(row[1] or 0))
        return None

    def record_discovery(self, token: "GraduatedToken", p_alive: Optional[float],
                         passed: bool, reject_reason: str = "",
                         features: Optional[Dict] = None):
        """Insert a token discovery/evaluation result."""
        self._execute_commit(
            """INSERT INTO discoveries
               (timestamp, mint_address, symbol, liquidity_usd, price_usd,
                p_alive, passed_filter, reject_reason, features)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (time.time(), token.mint_address, token.symbol,
             token.liquidity_usd, token.price_usd,
             p_alive, int(passed), reject_reason,
             json.dumps(features) if features else None))

    def save_swaps(self, mint_address: str, pool_address: str,
                   swaps: List["SwapRecord"]):
        """Bulk-insert swap records for backtesting analysis.

        Dedup against existing rows: after a bot restart the in-memory
        kline_builder has no `_seen_sigs` so it would re-fetch swaps that
        are already in DB. We protect the swaps table from duplicates by
        querying existing (timestamp, base_amount, is_buy) tuples for this
        (mint, pool) and only inserting rows whose tuple isn't already
        present. This is a substitute for a real (mint, pool, signature)
        UNIQUE index — adding that would require a schema migration.
        """
        if not swaps:
            return
        now = time.time()

        # Pull existing fingerprints for this (mint, pool) pair so we can dedup.
        # Only query when there's a non-trivial number of swaps to save — for
        # a single new swap the round-trip overhead is small enough to skip.
        existing: set = set()
        try:
            for ts, ba, ib in self._fetchall(
                "SELECT timestamp, base_amount, is_buy FROM swaps "
                "WHERE mint_address = ? AND pool_address = ?",
                (mint_address, pool_address),
            ):
                # Round base_amount to 6 sig figs to avoid float-equality issues
                existing.add((int(ts), round(float(ba), 6), int(ib)))
        except Exception:
            existing = set()

        rows_to_insert = []
        skipped = 0
        for s in swaps:
            key = (int(s.timestamp), round(float(s.base_amount), 6), int(s.is_buy))
            if key in existing:
                skipped += 1
                continue
            existing.add(key)  # also dedup within the new batch
            rows_to_insert.append(
                (mint_address, pool_address, s.timestamp, s.price_sol,
                 s.volume_sol, int(s.is_buy), s.base_amount, now,
                 getattr(s, "trader_address", "") or "")
            )

        if rows_to_insert:
            self._executemany_commit(
                """INSERT INTO swaps
                   (mint_address, pool_address, timestamp, price_sol, volume_sol,
                    is_buy, base_amount, collected_at, trader_address)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                rows_to_insert,
            )
        if skipped:
            logger.debug(
                f"save_swaps: dedup skipped {skipped} existing rows for "
                f"{mint_address[:12]}... (inserted {len(rows_to_insert)})"
            )

    def record_observation(self, token: "GraduatedToken",
                           model_score: Optional[float], model_passed: Optional[bool],
                           reject_reason: str = "",
                           features: Optional[Dict] = None,
                           kline_6m: Optional[List[Dict]] = None,
                           gmgn_info_entry: Optional[Dict] = None) -> int:
        """Record a token observation for research. Returns the row id."""
        cur = self._execute_commit(
            """INSERT INTO token_observations
               (mint_address, symbol, graduation_time, liquidity_usd, price_usd,
                source, pool_address, model_score, model_passed, reject_reason,
                features, kline_6m, gmgn_info_entry, observed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (token.mint_address, token.symbol, token.graduation_time,
             token.liquidity_usd, token.price_usd,
             getattr(token, "source", "chainstack"),
             getattr(token, "pool_address", None),
             model_score,
             int(model_passed) if model_passed is not None else None,
             reject_reason,
             json.dumps(features) if features else None,
             json.dumps(kline_6m) if kline_6m else None,
             json.dumps(gmgn_info_entry) if gmgn_info_entry else None,
             time.time()))
        return cur.lastrowid

    @staticmethod
    def _simulate_trade_from_kline(bars: List[Dict], entry_price_usd: float,
                                   sl_pct: float, act_pct: float = 0.20,
                                   drop_pct: float = 0.05, cost_pct: float = 0.05,
                                   size_usd: float = 10.0) -> Tuple[float, float, str]:
        """Pure-function kline replay for hypothetical PnL.

        Returns (peak_pnl_pct, exit_pnl_pct, exit_reason).
        Mirrors the live policy: trailing activation at `act_pct`, drop by
        `drop_pct` from peak triggers exit, hard stop at -`sl_pct`, else
        time-limit exit at last bar. 5% cost subtracted at entry and exit.
        """
        if not bars or entry_price_usd <= 0:
            return 0.0, 0.0, "no_data"
        entry = entry_price_usd * (1 + cost_pct)
        peak = 0.0
        exit_pnl = None
        exit_reason = "time_limit"
        for bar in bars:
            try:
                hi = float(bar.get("high", 0) or 0)
                lo = float(bar.get("low", 0) or 0)
                cl = float(bar.get("close", 0) or 0)
            except (TypeError, ValueError):
                continue
            if hi <= 0 or cl <= 0:
                continue
            if lo <= 0:
                lo = cl
            ph = (hi - entry) / entry
            pl = (lo - entry) / entry
            pc = (cl - entry) / entry
            if ph > peak:
                peak = ph
            if pl <= -sl_pct:
                exit_pnl = -sl_pct - cost_pct
                exit_reason = "stop_loss"
                break
            if peak >= act_pct and (pc <= peak - drop_pct or pl <= peak - drop_pct):
                exit_pnl = max(pl, peak - drop_pct) - cost_pct
                exit_reason = "trailing_stop"
                break
        if exit_pnl is None:
            last_cl = float(bars[-1].get("close", 0) or 0) if bars else 0
            exit_pnl = ((last_cl - entry) / entry - cost_pct) if last_cl > 0 else -cost_pct
        return peak, exit_pnl, exit_reason

    @staticmethod
    def _simulate_trade_from_swaps(
        swaps_sorted: List[Dict],
        entry_time_unix: float,
        sl_pct: float,
        act_pct: float = 0.20,
        drop_pct: float = 0.05,
        cost_pct: float = 0.05,
        poll_interval_sec: int = 5,
        poll_interval_trail_sec: int = 2,
        time_limit_sec: int = 1800,
        early_crash_pct: float = 0.10,
        early_crash_window_sec: int = 120,
    ) -> Tuple[Optional[float], Optional[float], str, int]:
        """Tick-level swap replay. Much more accurate than kline sim.

        Validated against 20 real trades: mean bias -$0.19 vs kline -$1.33.

        Args:
          swaps_sorted: list of {"timestamp": int, "price_sol": float} pre-sorted
          entry_time_unix: seconds since epoch to enter at
          others: policy params matching live config

        Returns:
          (peak_pnl_pct, exit_pnl_pct, exit_reason, n_ticks_used)
          or (None, None, reason, 0) if insufficient data.
        """
        if not swaps_sorted:
            return None, None, "no_swaps", 0
        # Entry: first swap at or after entry_time_unix
        entry_idx = None
        for i, s in enumerate(swaps_sorted):
            if s["timestamp"] >= entry_time_unix:
                entry_idx = i
                break
        if entry_idx is None:
            return None, None, "no_entry_swap", 0
        entry_price = float(swaps_sorted[entry_idx]["price_sol"])
        entry_ts = int(swaps_sorted[entry_idx]["timestamp"])
        if entry_price <= 0:
            return None, None, "bad_entry_price", 0

        # Apply 5% execution cost to entry
        entry = entry_price * (1 + cost_pct)
        peak = 0.0
        last_poll_ts = entry_ts
        trailing_on = False
        n_ticks = 0
        for s in swaps_sorted[entry_idx + 1:]:
            ts = int(s["timestamp"])
            price = float(s["price_sol"])
            if price <= 0:
                continue
            n_ticks += 1
            hold_sec = ts - entry_ts
            if hold_sec > time_limit_sec:
                pnl = (price - entry) / entry - cost_pct
                return peak, pnl, "time_limit", n_ticks
            interval = poll_interval_trail_sec if trailing_on else poll_interval_sec
            if ts - last_poll_ts < interval:
                continue
            last_poll_ts = ts
            pnl_pct = (price - entry) / entry
            if pnl_pct > peak:
                peak = pnl_pct
            if hold_sec < early_crash_window_sec and pnl_pct <= -early_crash_pct:
                return peak, pnl_pct - cost_pct, "early_crash", n_ticks
            if pnl_pct <= -sl_pct:
                return peak, pnl_pct - cost_pct, "stop_loss", n_ticks
            if peak >= act_pct:
                trailing_on = True
                if pnl_pct <= peak - drop_pct:
                    return peak, pnl_pct - cost_pct, "trailing", n_ticks
        # Ran out of swap data
        last = swaps_sorted[-1]
        pnl = (float(last["price_sol"]) - entry) / entry - cost_pct
        return peak, pnl, "ran_out_of_data", n_ticks

    @staticmethod
    def _apply_realistic_cost(
        raw_pnl_usd: Optional[float],
        exit_reason: Optional[str],
        entry_t_sec: int = 90,
        size_usd: float = 10.0,
    ) -> Optional[float]:
        """Transform raw swap-sim PnL into expected realized PnL using
        cost model calibrated from 20 real trades (2026-04-20).

        Cost layers (cumulative):
          - entry chase: linear 10% @T+90s → 48% @T+600s
          - trailing slip: 5% (when exit_reason=trailing)
          - SL cascade: +40% when exit_reason=stop_loss (rug completes)
          - Jupiter routing gap: 3% constant

        Args:
          raw_pnl_usd: swap-sim PnL at $10 position
          exit_reason: "trailing" / "stop_loss" / "early_crash" / "time_limit" / "ran_out_of_data"
          entry_t_sec: entry delay from graduation
        """
        if raw_pnl_usd is None: return None
        # Entry chase
        if entry_t_sec <= 90: chase = 0.10
        elif entry_t_sec >= 600: chase = 0.48
        else: chase = 0.10 + (entry_t_sec - 90) / (600 - 90) * (0.48 - 0.10)
        adj = raw_pnl_usd - size_usd * chase
        # Exit layer
        if exit_reason == "stop_loss":
            adj -= size_usd * 0.40  # SL cascade: -25% sim → -65% realized
        elif exit_reason in ("trailing", "trailing_stop"):
            adj -= size_usd * 0.05  # trailing exit slip
        # Jupiter routing gap (constant)
        adj -= size_usd * 0.03
        return adj

    def complete_observation(self, obs_id: int,
                             kline_30m: Optional[List[Dict]] = None,
                             trade_pnl_usd: Optional[float] = None,
                             trade_exit_reason: Optional[str] = None,
                             gmgn_info_60m: Optional[Dict] = None,
                             gmgn_security: Optional[Dict] = None):
        """Update observation with 30-min kline, GMGN snapshots, and trade result.

        Also pre-computes hypothetical PnL at SL=25%/30%/40% (V6 sim fields)
        so downstream analysis can query "what would SL=X have done" without
        re-running ad-hoc Python scripts.
        """
        # V6: kline-based shadow pre-computation. Fetch observation entry price, run sim.
        sim_pnl_25 = sim_pnl_30 = sim_pnl_40 = None
        sim_peak = None
        sim_exit_25 = None
        # V7: swap-level (tick-level) sim — more accurate
        sim_swap_25 = sim_swap_30 = sim_swap_40 = None
        sim_peak_swap = None
        sim_exit_25_swap = None
        sim_swap_n_ticks = None

        if kline_30m:
            try:
                row = self._fetchall(
                    "SELECT price_usd, observed_at, mint_address FROM token_observations WHERE id=?",
                    (obs_id,))
                entry_px = float(row[0][0]) if row and row[0][0] else 0.0
                observed_at = float(row[0][1]) if row and row[0][1] else 0.0
                mint = str(row[0][2]) if row and row[0][2] else ""
            except Exception:
                entry_px = 0.0; observed_at = 0.0; mint = ""
            if entry_px > 0:
                peak_25, ep_25, er_25 = self._simulate_trade_from_kline(kline_30m, entry_px, 0.25)
                _, ep_30, _ = self._simulate_trade_from_kline(kline_30m, entry_px, 0.30)
                _, ep_40, _ = self._simulate_trade_from_kline(kline_30m, entry_px, 0.40)
                sim_pnl_25 = 10.0 * ep_25   # $10 position
                sim_pnl_30 = 10.0 * ep_30
                sim_pnl_40 = 10.0 * ep_40
                sim_peak = peak_25          # peak is SL-independent
                sim_exit_25 = er_25

            # V7 swap-level sim: pull swaps for this mint after observed_at
            if mint and observed_at > 0:
                try:
                    swap_rows = self._fetchall(
                        "SELECT timestamp, price_sol FROM swaps "
                        "WHERE mint_address=? ORDER BY timestamp",
                        (mint,))
                    if swap_rows:
                        swaps = [{"timestamp": r[0], "price_sol": r[1]} for r in swap_rows]
                        pk_s, ep_s25, er_s25, n_s = self._simulate_trade_from_swaps(
                            swaps, observed_at, 0.25)
                        _, ep_s30, _, _ = self._simulate_trade_from_swaps(
                            swaps, observed_at, 0.30)
                        _, ep_s40, _, _ = self._simulate_trade_from_swaps(
                            swaps, observed_at, 0.40)
                        if ep_s25 is not None:
                            sim_swap_25 = 10.0 * ep_s25
                            sim_swap_30 = 10.0 * ep_s30 if ep_s30 is not None else None
                            sim_swap_40 = 10.0 * ep_s40 if ep_s40 is not None else None
                            sim_peak_swap = pk_s
                            sim_exit_25_swap = er_s25
                            sim_swap_n_ticks = n_s
                except Exception as e:
                    logger.debug(f"V7 swap-sim failed for obs {obs_id}: {e}")

        # V8: realistic-cost sim — apply measured execution costs to swap-sim
        sim_real_25 = self._apply_realistic_cost(sim_swap_25, sim_exit_25_swap, 90)
        sim_real_30 = self._apply_realistic_cost(sim_swap_30, sim_exit_25_swap, 90)
        sim_real_40 = self._apply_realistic_cost(sim_swap_40, sim_exit_25_swap, 90)
        realistic_ver = "v1_20trades_2026-04-20" if sim_real_25 is not None else None

        self._execute_commit(
            """UPDATE token_observations
               SET kline_30m = ?, trade_pnl_usd = ?, trade_exit_reason = ?,
                   gmgn_info_60m = ?, gmgn_security = ?, completed_at = ?,
                   sim_pnl_sl25_usd = ?, sim_pnl_sl30_usd = ?,
                   sim_pnl_sl40_usd = ?, sim_peak_pnl_pct = ?,
                   sim_exit_reason_sl25 = ?, sim_computed_at = ?,
                   sim_pnl_sl25_swap_usd = ?, sim_pnl_sl30_swap_usd = ?,
                   sim_pnl_sl40_swap_usd = ?, sim_peak_pnl_pct_swap = ?,
                   sim_exit_reason_sl25_swap = ?, sim_swap_n_ticks = ?,
                   sim_computed_at_swap = ?,
                   sim_pnl_sl25_realistic_usd = ?, sim_pnl_sl30_realistic_usd = ?,
                   sim_pnl_sl40_realistic_usd = ?, sim_realistic_cost_version = ?,
                   sim_computed_at_realistic = ?
               WHERE id = ?""",
            (json.dumps(kline_30m) if kline_30m else None,
             trade_pnl_usd, trade_exit_reason,
             json.dumps(gmgn_info_60m) if gmgn_info_60m else None,
             json.dumps(gmgn_security) if gmgn_security else None,
             time.time(),
             sim_pnl_25, sim_pnl_30, sim_pnl_40, sim_peak,
             sim_exit_25, time.time() if sim_pnl_25 is not None else None,
             sim_swap_25, sim_swap_30, sim_swap_40, sim_peak_swap,
             sim_exit_25_swap, sim_swap_n_ticks,
             time.time() if sim_swap_25 is not None else None,
             sim_real_25, sim_real_30, sim_real_40, realistic_ver,
             time.time() if sim_real_25 is not None else None,
             obs_id))

    def update_observation_snapshot(self, obs_id: int, column: str, data: Dict):
        """Update a single GMGN / DexScreener snapshot column."""
        allowed = {
            "gmgn_info_t0", "gmgn_security_t0",
            "gmgn_info_15m", "gmgn_info_30m", "gmgn_info_45m",
            "gmgn_info_60m",  # was missing
            "dexscr_info_t0",  # V5 addition
        }
        if column not in allowed:
            return
        self._execute_commit(
            f"UPDATE token_observations SET {column} = ? WHERE id = ?",
            (json.dumps(data), obs_id))

    # ── Trader Record Management ──────────────────────────────────────

    def update_trader_records(self, mint_address: str, is_winner: bool):
        """Update win/loss records for all early buyers of a token.

        Called when an observation completes and we know the outcome.
        Only updates traders who bought in the first 60 seconds.

        Args:
            mint_address: the token that just completed observation
            is_winner: True if the token had positive net return at T+3m
        """
        # Get early buyers (first 60s) with trader_address
        rows = self._fetchall(
            """SELECT DISTINCT trader_address, SUM(volume_sol) as total_vol
               FROM swaps
               WHERE mint_address = ?
                 AND trader_address != ''
                 AND is_buy = 1
               GROUP BY trader_address""",
            (mint_address,),
        )

        if not rows:
            return 0

        now = time.time()
        updated = 0
        for row in rows:
            trader = row[0]
            vol = row[1] or 0
            if not trader or not trader.strip():
                continue

            # Upsert: increment wins or losses
            if is_winner:
                self._execute_commit(
                    """INSERT INTO trader_records (trader_address, wins, losses, total_volume, last_updated)
                       VALUES (?, 1, 0, ?, ?)
                       ON CONFLICT(trader_address) DO UPDATE SET
                           wins = wins + 1,
                           total_volume = total_volume + ?,
                           last_updated = ?""",
                    (trader, vol, now, vol, now),
                )
            else:
                self._execute_commit(
                    """INSERT INTO trader_records (trader_address, wins, losses, total_volume, last_updated)
                       VALUES (?, 0, 1, ?, ?)
                       ON CONFLICT(trader_address) DO UPDATE SET
                           losses = losses + 1,
                           total_volume = total_volume + ?,
                           last_updated = ?""",
                    (trader, vol, now, vol, now),
                )
            updated += 1

        return updated

    def get_trader_record(self, trader_address: str) -> Optional[Dict]:
        """Get a single trader's win/loss record."""
        row = self._fetchone(
            "SELECT wins, losses, total_volume, last_updated FROM trader_records WHERE trader_address = ?",
            (trader_address,),
        )
        if not row:
            return None
        return {"wins": row[0], "losses": row[1], "total_volume": row[2],
                "last_updated": row[3]}

    def get_trader_records_batch(self, addresses: List[str]) -> Dict[str, Dict]:
        """Get win/loss records for multiple traders at once."""
        if not addresses:
            return {}
        placeholders = ",".join("?" * len(addresses))
        rows = self._fetchall(
            f"""SELECT trader_address, wins, losses, total_volume
                FROM trader_records
                WHERE trader_address IN ({placeholders})""",
            addresses,
        )
        return {
            r[0]: {"wins": r[1], "losses": r[2], "total_volume": r[3]}
            for r in rows
        }

    def get_trader_records_count(self) -> int:
        """Return total number of rated traders."""
        row = self._fetchone("SELECT COUNT(*) FROM trader_records")
        return row[0] if row else 0

    def compute_smart_money_features(self, mint_address: str,
                                      min_history: int = 3,
                                      smart_threshold: float = 0.5) -> Dict[str, float]:
        """Compute smart money features for a token from its early buyers.

        Matches the 14a research feature definitions exactly:
        - Only uses traders with min_history prior trades (rated traders)
        - Smart = win_rate >= smart_threshold
        - All features observable at T+60s (uses first 60s of swaps)
        - CRITICAL: restrict to first 60s of swaps only (matches research 14a)

        Returns dict of sm_* features, or empty dict if insufficient data.
        """
        # Get graduation_time for the time window
        gt_row = self._fetchone(
            "SELECT graduation_time FROM token_observations WHERE mint_address = ? LIMIT 1",
            (mint_address,),
        )
        grad_time = gt_row[0] if gt_row else None

        # Get early buyers (first 60s only) with their volumes
        # This matches research 14a: early_buys = swaps[(offset >= 0) & (offset < 60) & (is_buy == 1)]
        if grad_time and grad_time > 0:
            rows = self._fetchall(
                """SELECT trader_address, SUM(volume_sol) as total_vol
                   FROM swaps
                   WHERE mint_address = ? AND trader_address != '' AND is_buy = 1
                   AND timestamp >= ? AND timestamp < ?
                   GROUP BY trader_address
                   ORDER BY total_vol DESC""",
                (mint_address, int(grad_time), int(grad_time) + 60),
            )
        else:
            # Fallback: use all swaps if graduation_time unavailable
            rows = self._fetchall(
                """SELECT trader_address, SUM(volume_sol) as total_vol
                   FROM swaps
                   WHERE mint_address = ? AND trader_address != '' AND is_buy = 1
                   GROUP BY trader_address
                   ORDER BY total_vol DESC""",
                (mint_address,),
            )
        if not rows:
            return {}

        buyers = [(r[0], r[1]) for r in rows if r[0] and r[0].strip()]
        if not buyers:
            return {}

        total_buyers = len(buyers)
        total_volume = sum(vol for _, vol in buyers)

        # Get trader records for all buyers
        addresses = [addr for addr, _ in buyers]
        records = self.get_trader_records_batch(addresses)

        n_rated = 0
        n_smart = 0
        n_dumb = 0
        smart_volume = 0.0
        dumb_volume = 0.0
        top_buyer_wr = float("nan")
        top_buyer_n = 0
        best_buyer_wr = 0.0
        best_buyer_n = 0
        weighted_wr_sum = 0.0
        weighted_wr_denom = 0.0

        for buyer_addr, buyer_vol in buyers:
            rec = records.get(buyer_addr)
            if not rec:
                continue
            total_trades = rec["wins"] + rec["losses"]
            if total_trades < min_history:
                continue

            wr = rec["wins"] / total_trades
            n_rated += 1
            weighted_wr_sum += wr * buyer_vol
            weighted_wr_denom += buyer_vol

            if wr >= smart_threshold:
                n_smart += 1
                smart_volume += buyer_vol
            else:
                n_dumb += 1
                dumb_volume += buyer_vol

            # Top buyer (by volume)
            if buyer_addr == buyers[0][0]:
                top_buyer_wr = wr
                top_buyer_n = total_trades
            # Best buyer (by win rate, min 5 trades)
            if total_trades >= 5 and wr > best_buyer_wr:
                best_buyer_wr = wr
                best_buyer_n = total_trades

        return {
            "sm_rated_buyer_frac": n_rated / total_buyers if total_buyers > 0 else 0,
            "sm_smart_buyer_frac": n_smart / total_buyers if total_buyers > 0 else 0,
            "sm_smart_vol_frac": smart_volume / total_volume if total_volume > 0 else 0,
            "sm_dumb_buyer_frac": n_dumb / total_buyers if total_buyers > 0 else 0,
            "sm_dumb_vol_frac": dumb_volume / total_volume if total_volume > 0 else 0,
            "sm_smart_vs_dumb_vol": (smart_volume - dumb_volume) / total_volume if total_volume > 0 else 0,
            "sm_top_buyer_wr": top_buyer_wr,
            "sm_top_buyer_n_trades": top_buyer_n,
            "sm_best_buyer_wr": best_buyer_wr,
            "sm_best_buyer_n_trades": best_buyer_n,
            "sm_vol_weighted_wr": weighted_wr_sum / weighted_wr_denom if weighted_wr_denom > 0 else float("nan"),
        }

    def get_observation_gmgn_t0(self, mint_address: str) -> Optional[Dict]:
        """Get gmgn_info_t0 JSON from token_observations for a given mint."""
        row = self._fetchone(
            "SELECT gmgn_info_t0 FROM token_observations WHERE mint_address = ? AND gmgn_info_t0 IS NOT NULL LIMIT 1",
            (mint_address,),
        )
        if row and row[0]:
            try:
                import json
                return json.loads(row[0])
            except Exception:
                pass
        return None

    def get_swaps_for_token(self, mint_address: str) -> List[Dict]:
        """Return all swaps for a token as list of dicts for micro feature computation."""
        rows = self._fetchall(
            """SELECT timestamp as block_time, is_buy, volume_sol as sol_amount,
                      trader_address
               FROM swaps
               WHERE mint_address = ?
               ORDER BY timestamp""",
            (mint_address,),
        )
        return [{"block_time": r[0], "is_buy": bool(r[1]),
                 "sol_amount": r[2], "trader_address": r[3] or ""}
                for r in rows]

    def load_orphan_observations(self, buffer_sec: float = 62 * 60) -> List[Dict]:
        """Return uncompleted observations whose GMGN kline window has elapsed."""
        cutoff = time.time() - buffer_sec
        rows = self._fetchall(
            """SELECT id, mint_address, symbol, graduation_time, pool_address, source
               FROM token_observations
               WHERE completed_at IS NULL AND graduation_time < ?
               ORDER BY graduation_time ASC""",
            (cutoff,))
        return [{"id": r[0], "mint": r[1], "symbol": r[2],
                 "graduation_time": r[3], "pool": r[4], "source": r[5]}
                for r in rows]

    def load_pending_observations(self) -> List[Dict]:
        """Return ALL uncompleted observations (for restart recovery)."""
        rows = self._fetchall(
            """SELECT id, mint_address, symbol, graduation_time, pool_address, source,
                      liquidity_usd, price_usd,
                      gmgn_info_15m IS NOT NULL as has_15m,
                      gmgn_info_30m IS NOT NULL as has_30m,
                      gmgn_info_45m IS NOT NULL as has_45m
               FROM token_observations
               WHERE completed_at IS NULL
               ORDER BY graduation_time ASC""")
        return [{"id": r[0], "mint": r[1], "symbol": r[2],
                 "graduation_time": r[3], "pool": r[4], "source": r[5],
                 "liquidity_usd": r[6], "price_usd": r[7],
                 "has_15m": bool(r[8]), "has_30m": bool(r[9]), "has_45m": bool(r[10])}
                for r in rows]

    def record_event(self, level: str, module: str, message: str):
        """Insert a structured log event."""
        self._execute_commit(
            "INSERT INTO events (timestamp, level, module, message) VALUES (?,?,?,?)",
            (time.time(), level, module, message))

    def record_latency_event(self, mint_address: str, symbol: str,
                             graduation_time: Optional[float], event_name: str,
                             age_sec: Optional[float] = None,
                             metadata: Optional[Dict] = None):
        """Insert a latency event for live engineering analysis."""
        if event_name == "sell_trigger" and self._should_skip_sell_trigger(mint_address, metadata):
            return
        if self._telemetry_sink is not None:
            self._telemetry_sink.record_latency_event(
                mint_address=mint_address,
                symbol=symbol,
                graduation_time=graduation_time,
                event_name=event_name,
                age_sec=age_sec,
                metadata=metadata,
            )
            return
        # Sell retries can hit the same time-limit branch many times while the
        # position is still open. Keep only one recent sell_trigger per
        # (mint, reason) window so telemetry does not become a write-amplified
        # spam source for the DB.
        if event_name == "sell_trigger" and metadata and metadata.get("reason"):
            recent = self._fetchone(
                """SELECT timestamp, metadata
                   FROM latency_events
                   WHERE mint_address = ? AND event_name = ?
                   ORDER BY id DESC LIMIT 1""",
                (mint_address, event_name),
            )
            if recent:
                recent_ts, recent_meta = recent
                try:
                    recent_reason = json.loads(recent_meta).get("reason") if recent_meta else None
                except Exception:
                    recent_reason = None
                if recent_reason == metadata.get("reason") and (time.time() - float(recent_ts)) < 60:
                    return
        self._execute_commit(
            """INSERT INTO latency_events
               (timestamp, mint_address, symbol, graduation_time, event_name, age_sec, metadata)
               VALUES (?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, graduation_time, event_name,
             age_sec, json.dumps(metadata) if metadata else None))

    def get_latest_shadow_rug_eval(self, mint_address: str) -> Optional[Dict]:
        """Return latest rug filter score for a mint, or None if not scored.

        Used by the rug_filter gate at M3 preflight to reuse the t0 score
        computed at graduation time (the trained feature time-point) rather
        than re-scoring at T+600s with stale snapshot.
        """
        row = self._fetchall(
            """SELECT score, cutoff_band, cutoff_value, would_reject, model_version
               FROM shadow_rug_evals
               WHERE mint_address = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (mint_address,))
        if not row:
            return None
        score, band, cutoff, would_reject, version = row[0]
        return {
            "score": float(score) if score is not None else 0.0,
            "cutoff_band": band,
            "cutoff_value": float(cutoff) if cutoff is not None else 1.0,
            "would_reject": bool(would_reject),
            "model_version": version,
        }

    def record_shadow_rug_eval(self, *, mint_address: str, symbol: Optional[str],
                                graduation_time: Optional[float],
                                snapshot_delay_sec: Optional[float],
                                score: float, cutoff_band: str,
                                cutoff_value: float, would_reject: bool,
                                model_version: Optional[str],
                                features: Optional[Dict]):
        """One-shot record of rug filter score at M2 promotion time."""
        self._execute_commit(
            """INSERT INTO shadow_rug_evals
               (timestamp, mint_address, symbol, graduation_time,
                snapshot_delay_sec, score, cutoff_band, cutoff_value,
                would_reject, model_version, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, graduation_time,
             snapshot_delay_sec, score, cutoff_band, cutoff_value,
             1 if would_reject else 0, model_version,
             json.dumps(features) if features else None))

    def has_shadow_rug_v3_eval(self, mint_address: str) -> bool:
        """Check if v3 has already scored this mint (idempotency guard)."""
        row = self._fetchall(
            "SELECT 1 FROM shadow_rug_filter_v3_evals WHERE mint_address = ? LIMIT 1",
            (mint_address,))
        return bool(row)

    def record_shadow_rug_v3_eval(self, *, mint_address: str, symbol: Optional[str],
                                   graduation_time: Optional[float],
                                   scored_at_delay_s: Optional[float],
                                   window_s: int, n_swaps: int,
                                   score: Optional[float], cutoff: float,
                                   decision: str, reason: str,
                                   creator: Optional[str],
                                   features: Optional[Dict],
                                   model_version: Optional[str]):
        """Log v3 T+60s shadow score (shadow-only, never gates trades)."""
        self._execute_commit(
            """INSERT INTO shadow_rug_filter_v3_evals
               (timestamp, mint_address, symbol, graduation_time,
                scored_at_delay_s, window_s, n_swaps, v3_score, cutoff,
                decision, reason, creator, features, model_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, graduation_time,
             scored_at_delay_s, int(window_s), int(n_swaps),
             float(score) if score is not None and score == score else None,
             float(cutoff), decision, reason, creator,
             json.dumps(features) if features else None,
             model_version))

    def record_shadow_event_eval(self, *, mint_address: str, symbol: Optional[str],
                                  graduation_time: Optional[float],
                                  scan_t_sec: int,
                                  cutoff_band: str, cutoff_value: float,
                                  score: float, pattern: str,
                                  entry_price_sol: Optional[float],
                                  sol_price_usd: Optional[float],
                                  model_version: Optional[str],
                                  features: Optional[Dict]):
        """First-trigger record for Phase B shadow event scanner."""
        self._execute_commit(
            """INSERT INTO shadow_event_evals
               (timestamp, mint_address, symbol, graduation_time, scan_t_sec,
                cutoff_band, cutoff_value, score, pattern, entry_price_sol,
                sol_price_usd, model_version, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, graduation_time, scan_t_sec,
             cutoff_band, cutoff_value, score, pattern, entry_price_sol,
             sol_price_usd, model_version,
             json.dumps(features) if features else None))

    def record_preflight_check(self, *, mint_address: str,
                               symbol: Optional[str], check_name: str,
                               outcome: str, value: Optional[float] = None,
                               threshold: Optional[float] = None,
                               detail: Optional[str] = None,
                               model_score: Optional[float] = None,
                               pool_liq_usd: Optional[float] = None):
        """Log one M3 preflight check outcome (pass/block/skip)."""
        self._execute_commit(
            """INSERT INTO preflight_checks
               (timestamp, mint_address, symbol, check_name, outcome, value,
                threshold, detail, model_score, pool_liq_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, check_name, outcome, value,
             threshold, detail, model_score, pool_liq_usd))

    def record_rug_event(self, *, mint_address: str,
                         pool_address: Optional[str],
                         sol_amount: float, price_sol: float,
                         detected_at: float, action_taken: str,
                         signature: Optional[str] = None,
                         triggered_pnl_pct: Optional[float] = None):
        """Log one rug event from gRPC stream. Used to measure latency
        savings vs M4 poll-based detection and to validate thresholds."""
        self._execute_commit(
            """INSERT INTO rug_events
               (timestamp, mint_address, pool_address, sol_amount, price_sol,
                detected_at, action_taken, signature, triggered_pnl_pct)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, pool_address, sol_amount, price_sol,
             detected_at, action_taken, signature, triggered_pnl_pct))

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
        """First-trigger record for Phase B event_invariant Model B shadow."""
        self._execute_commit(
            """INSERT INTO shadow_event_invariant_evals
               (timestamp, mint_address, symbol, graduation_time, scan_t_sec,
                cutoff_band, cutoff_value, score, pattern, entry_price_sol,
                sol_price_usd, model_version, features)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, graduation_time, scan_t_sec,
             cutoff_band, cutoff_value, score, pattern, entry_price_sol,
             sol_price_usd, model_version,
             json.dumps(features) if features else None))

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
        """One row per open-position monitor cycle. Used to reconstruct
        Jupiter-vs-pool divergence events and exit-execution gaps after
        the fact (BK 2026-04-14 postmortem motivation).
        """
        self._execute_commit(
            """INSERT INTO price_probes
               (timestamp, mint_address, symbol, hold_sec, entry_price_sol,
                pool_price_sol, jupiter_price_usd, sol_price_usd,
                divergence_pct, price_source, pool_pnl_pct, jup_pnl_pct,
                peak_pnl_pct, trailing_activated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mint_address, symbol, hold_sec, entry_price_sol,
             pool_price_sol, jupiter_price_usd, sol_price_usd,
             divergence_pct, price_source, pool_pnl_pct, jup_pnl_pct,
             peak_pnl_pct, 1 if trailing_activated else 0))

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
                                     p_rug_live_v1: Optional[float] = None):
        """One row per (position, decision_tick) from the 14y dual-model
        inference hook. Resolved asynchronously 60s later.

        `p_rug_live_v1` is the bot-schema retrain score (2026-04-24 A/B).
        See Phase_14y_Enhancement_FROZEN_2026-04-24.md §9.9.
        """
        self._execute_commit(
            """INSERT INTO shadow_exit_warn_evals
               (timestamp, position_id, mint_address, symbol,
                dt_from_entry, dt_from_grad,
                p_drop_raw, p_rug_raw, mid_price_sol,
                n_cache_swaps, feature_window_count,
                resolved, features_json, p_rug_live_v1)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?,?)""",
            (time.time(), position_id, mint_address, symbol,
             dt_from_entry, dt_from_grad,
             p_drop_raw, p_rug_raw, mid_price_sol,
             n_cache_swaps, feature_window_count,
             features_json, p_rug_live_v1))

    def fetch_unresolved_shadow_exit_warn_evals(self, older_than_sec: float = 60.0,
                                                limit: int = 1000) -> list:
        """Return unresolved rows older than `older_than_sec` — the resolution
        job calls this, computes realized outcomes from live swap data, then
        invokes `resolve_shadow_exit_warn_eval()` per row.
        """
        cutoff = time.time() - older_than_sec
        return self._fetchall(
            """SELECT id, mint_address, timestamp
               FROM shadow_exit_warn_evals
               WHERE resolved = 0 AND timestamp <= ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (cutoff, limit))

    def resolve_shadow_exit_warn_eval(self, *, row_id: int,
                                      realized_drop_60: Optional[float],
                                      realized_max_sell_60: Optional[float],
                                      y_drop_actual: Optional[int],
                                      y_rug_actual: Optional[int]):
        """Mark a shadow-exit-warn row as resolved with its 60s-later outcomes."""
        self._execute_commit(
            """UPDATE shadow_exit_warn_evals
               SET resolved = 1,
                   realized_drop_60 = ?,
                   realized_max_sell_60 = ?,
                   y_drop_actual = ?,
                   y_rug_actual = ?
               WHERE id = ?""",
            (realized_drop_60, realized_max_sell_60,
             y_drop_actual, y_rug_actual, row_id))

    def count_shadow_exit_warn_evals(self, resolved: Optional[bool] = None) -> int:
        """Utility for monitoring: total / resolved-only / unresolved-only counts."""
        if resolved is None:
            rows = self._fetchall("SELECT COUNT(*) FROM shadow_exit_warn_evals", ())
        else:
            rows = self._fetchall(
                "SELECT COUNT(*) FROM shadow_exit_warn_evals WHERE resolved = ?",
                (1 if resolved else 0,))
        return int(rows[0][0]) if rows else 0

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
        """One row per graduation at T+180s from the sw_v2 shadow hook.
        Resolved asynchronously at T+18m (entry + 15 min).
        """
        self._execute_commit(
            """INSERT INTO shadow_sw_v2_evals
               (timestamp, mint_address, symbol, graduation_time, dt_from_grad,
                entry_offset_sec, entry_price_sol, grad_price_sol,
                n_bars_pre_entry, n_swaps_used, feature_ready, f1_score,
                features_json, v14_candidate, v14_entered, v14_score,
                v14_pattern, rug_filter_score, resolved)
               VALUES (?,?,?,?,?, ?,?,?, ?,?,?,?, ?, ?,?,?, ?,?, 0)""",
            (time.time(), mint_address, symbol, graduation_time, dt_from_grad,
             entry_offset_sec, entry_price_sol, grad_price_sol,
             n_bars_pre_entry, n_swaps_used,
             1 if feature_ready else 0, f1_score,
             features_json,
             1 if v14_candidate else 0, 1 if v14_entered else 0,
             v14_score, v14_pattern, rug_filter_score))

    def fetch_unresolved_shadow_sw_v2_evals(self, older_than_sec: float = 900.0,
                                            limit: int = 500) -> list:
        """Return rows where timestamp + older_than_sec < now AND resolved=0.
        Default 900s = 15 min, matching forward_raw_15m horizon."""
        cutoff = time.time() - older_than_sec
        return self._fetchall(
            """SELECT id, mint_address, graduation_time, entry_offset_sec,
                      entry_price_sol, timestamp
               FROM shadow_sw_v2_evals
               WHERE resolved = 0 AND timestamp <= ?
               ORDER BY timestamp ASC
               LIMIT ?""",
            (cutoff, limit))

    def resolve_shadow_sw_v2_eval(self, *, row_id: int,
                                  realized_raw_15m: Optional[float],
                                  y_net15_emp_trail_actual: Optional[int],
                                  peak_ret_15m: Optional[float],
                                  trough_ret_15m: Optional[float],
                                  eventual_trade_pnl_usd: Optional[float]):
        """Mark a sw_v2 shadow row as resolved with its T+18m outcomes."""
        self._execute_commit(
            """UPDATE shadow_sw_v2_evals
               SET resolved = 1,
                   realized_raw_15m = ?,
                   y_net15_emp_trail_actual = ?,
                   peak_ret_15m = ?,
                   trough_ret_15m = ?,
                   eventual_trade_pnl_usd = ?
               WHERE id = ?""",
            (realized_raw_15m, y_net15_emp_trail_actual,
             peak_ret_15m, trough_ret_15m, eventual_trade_pnl_usd, row_id))

    def count_shadow_sw_v2_evals(self, resolved: Optional[bool] = None) -> int:
        if resolved is None:
            rows = self._fetchall("SELECT COUNT(*) FROM shadow_sw_v2_evals", ())
        else:
            rows = self._fetchall(
                "SELECT COUNT(*) FROM shadow_sw_v2_evals WHERE resolved = ?",
                (1 if resolved else 0,))
        return int(rows[0][0]) if rows else 0

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
        """Insert a live shadow evaluation for the rule-first policy."""
        if self._telemetry_sink is not None:
            self._telemetry_sink.record_shadow_policy_eval(
                obs_id=obs_id,
                mint_address=mint_address,
                symbol=symbol,
                graduation_time=graduation_time,
                feature_delay_sec=feature_delay_sec,
                stage1_name=stage1_name,
                stage1_score=stage1_score,
                stage1_threshold=stage1_threshold,
                stage1_passed=stage1_passed,
                stage2_rule_name=stage2_rule_name,
                stage2_passed=stage2_passed,
                feature_source=feature_source,
                feature_view=feature_view,
                range_1to3m=range_1to3m,
                drawdown_1to3m=drawdown_1to3m,
                total_volume_3m=total_volume_3m,
                total_trades_3m=total_trades_3m,
                metadata=metadata,
            )
            return
        self._execute_commit(
            """INSERT INTO shadow_policy_evals
               (timestamp, obs_id, mint_address, symbol, graduation_time, feature_delay_sec,
                stage1_name, stage1_score, stage1_threshold, stage1_passed,
                stage2_rule_name, stage2_passed, feature_source, feature_view,
                range_1to3m, drawdown_1to3m, total_volume_3m, total_trades_3m, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), obs_id, mint_address, symbol, graduation_time, feature_delay_sec,
             stage1_name, stage1_score, stage1_threshold,
             int(stage1_passed) if stage1_passed is not None else None,
             stage2_rule_name,
             int(stage2_passed) if stage2_passed is not None else None,
             feature_source, feature_view,
             range_1to3m, drawdown_1to3m, total_volume_3m, total_trades_3m,
             json.dumps(metadata) if metadata else None))

    def save_position(self, pos: "Position"):
        """Upsert an open position for restart recovery."""
        self._execute_commit(
            """INSERT OR REPLACE INTO open_positions
               (mint_address, symbol, name, decimals, graduation_time,
                liquidity_usd, price_usd, entry_price_sol, token_amount,
                entry_time, entry_tx, sol_invested, p_alive, features,
                peak_pnl_pct, trailing_activated, pool_address, source)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pos.token.mint_address, pos.token.symbol, pos.token.name,
             pos.token.decimals, pos.token.graduation_time,
             pos.token.liquidity_usd, pos.token.price_usd,
             pos.entry_price_sol, pos.token_amount,
             pos.entry_time, pos.entry_tx, pos.sol_invested,
             pos.model_score, json.dumps(pos.features) if pos.features else None,
             pos.peak_pnl_pct, int(pos.trailing_activated),
             pos.token.pool_address, getattr(pos.token, "source", "chainstack")))

    def remove_position(self, mint_address: str):
        """Remove a closed position from the open_positions table."""
        self._execute_commit(
            "DELETE FROM open_positions WHERE mint_address = ?", (mint_address,))

    def load_open_positions(self) -> List["Position"]:
        """Load all open positions from DB for restart recovery."""
        rows = self._fetchall(
            """SELECT mint_address, symbol, name, decimals, graduation_time,
                      liquidity_usd, price_usd, entry_price_sol, token_amount,
                      entry_time, entry_tx, sol_invested, p_alive, features,
                      peak_pnl_pct, trailing_activated, pool_address, source
               FROM open_positions""")
        positions = []
        for row in rows:
            token = GraduatedToken(
                mint_address=row[0], symbol=row[1], name=row[2],
                decimals=row[3], graduation_time=row[4],
                liquidity_usd=row[5] or 0, price_usd=row[6] or 0,
                pool_address=row[16],
                source=row[17] or "chainstack",
            )
            features = json.loads(row[13]) if row[13] else None
            pos = Position(
                token=token,
                entry_price_sol=row[7],
                token_amount=row[8],
                entry_time=row[9],
                entry_tx=row[10] or "",
                sol_invested=row[11],
                model_score=row[12] or 0,
                features=features,
                peak_pnl_pct=row[14] or 0.0,
                trailing_activated=bool(row[15]),
            )
            positions.append(pos)
        return positions

    def get_recent_stoploss_mints(self, cooldown_sec: float) -> Dict[str, float]:
        """Return {mint_address: blacklist_expiry_ts} for recent stop_loss trades.
        Used to rebuild _stoploss_blacklist on startup."""
        cutoff = time.time() - cooldown_sec
        rows = self._fetchall(
            "SELECT mint_address, MAX(timestamp) as last_sl FROM trades "
            "WHERE exit_reason = 'stop_loss' AND timestamp >= ? "
            "GROUP BY mint_address",
            (cutoff,))
        result = {}
        for mint, ts in rows:
            expiry = ts + cooldown_sec
            if expiry > time.time():
                result[mint] = expiry
        return result

    def get_trades_for_date(self, date_str: str) -> List[Dict]:
        """Return all trades for a given UTC date (YYYY-MM-DD), ordered by timestamp ASC.
        Used by RiskManager.rebuild_from_db() to reconstruct daily state."""
        import calendar
        import datetime
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        start_ts = calendar.timegm(dt.timetuple())  # UTC, not local TZ
        end_ts = start_ts + 86400
        rows = self._fetchall(
            "SELECT timestamp, pnl_usd FROM trades WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
            (start_ts, end_ts))
        return [{"timestamp": r[0], "pnl_usd": r[1]} for r in rows]

    def load_all_trades(self) -> List[TradeRecord]:
        """Load all trades from DB into TradeRecord objects for _trade_log rebuild."""
        rows = self._fetchall(
            "SELECT timestamp, token_symbol, mint_address, entry_price_sol, exit_price_sol, "
            "token_amount, sol_invested, sol_received, pnl_sol, pnl_usd, hold_seconds, "
            "exit_reason, entry_tx, exit_tx, p_alive, entry_time, peak_pnl_pct, "
            "sol_price_usd, trigger_pnl_pct "
            "FROM trades ORDER BY timestamp ASC"
        )
        records = []
        for r in rows:
            records.append(TradeRecord(
                token_symbol=r[1] or "", mint_address=r[2] or "",
                entry_price_sol=r[3] or 0, exit_price_sol=r[4] or 0,
                token_amount=r[5] or 0, sol_invested=r[6] or 0,
                sol_received=r[7] or 0, pnl_sol=r[8] or 0,
                pnl_usd=r[9] or 0, hold_seconds=r[10] or 0,
                exit_reason=r[11] or "", entry_tx=r[12] or "",
                exit_tx=r[13] or "", model_score=r[14] or 0,
                timestamp=r[0] or 0, entry_time=r[15] or 0,
                peak_pnl_pct=r[16] or 0, sol_price_usd=r[17] or 0,
                trigger_pnl_pct=r[18] or 0,
            ))
        return records

    def close(self):
        with self._lock:
            self._conn.close()
        if self._telemetry_sink is not None:
            self._telemetry_sink.close()
