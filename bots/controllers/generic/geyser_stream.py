"""
Yellowstone gRPC Geyser stream for PumpSwap AMM transactions.

Single stream replaces:
  M1: WS logsSubscribe + getTransaction → graduation detection
  M2: getSignaturesForAddress + batch getTransaction → swap collection

Subscribes to all PumpSwap AMM transactions and routes each to:
  - create_pool → ChainGraduation → graduation queue
  - swap on registered pool → SwapRecord → KlineBuilder push
  - other → discard
"""
import asyncio
import logging
import time
from collections import deque
from statistics import median
from typing import TYPE_CHECKING, Deque, Dict, Optional, Set, Tuple

import grpc

if TYPE_CHECKING:
    from controllers.generic.meme_sniper_utils import (
        ChainGraduation, OnChainKlineBuilder, SwapRecord,
    )

logger = logging.getLogger(__name__)

# ── Solana / PumpSwap constants (duplicated to avoid circular import) ──
PUMPSWAP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
QUOTE_TOKEN_BLACKLIST = frozenset({WSOL_MINT, USDC_MINT, USDT_MINT})

PUMPSWAP_CREATE_POOL_DISC = bytes([233, 146, 209, 142, 207, 104, 64, 188])
ACCOUNT_IDX_POOL = 0
ACCOUNT_IDX_CREATOR = 2
ACCOUNT_IDX_BASE_MINT = 3
ACCOUNT_IDX_QUOTE_MINT = 4

# Max entries in slot→block_time cache (~500 slots = ~3-4 min at 400ms/slot)
SLOT_CACHE_MAX = 500

# Solana validator slot time (avg ~0.4s, with rare skipped slots).
# Used by _get_block_time for cache-miss estimation (Phase 26.X 2026-05-11).
SLOT_TIME_SEC = 0.4

# Dust threshold for swap volume (SOL)
DUST_THRESHOLD_SOL = 0.01

# Inline base58 encoder so the gRPC path does not depend on the optional
# third-party `base58` package being present in the runtime image.
_B58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = bytearray()
    while n:
        n, rem = divmod(n, 58)
        out.append(_B58_ALPHABET[rem])
    pad = len(data) - len(data.lstrip(b"\x00"))
    encoded = (b"1" * pad) + bytes(reversed(out or b""))
    return encoded.decode("ascii")


class GeyserPumpSwapStream:
    """Single Yellowstone gRPC stream that routes PumpSwap AMM transactions
    to graduation detection (M1) and swap collection (M2).

    Args:
        grpc_url: Yellowstone gRPC endpoint (e.g. solana-yellowstone-grpc.chainstack.com:443)
        grpc_token: Authentication token (x-token header)
        graduation_queue: asyncio.Queue for ChainGraduation objects (shared with TokenDiscovery)
        kline_builder: OnChainKlineBuilder for push-based swap injection
        seen_mints: Set of already-seen mints (shared with TokenDiscovery)
    """

    def __init__(
        self,
        grpc_url: str,
        grpc_token: str,
        graduation_queue: asyncio.Queue,
        kline_builder: "OnChainKlineBuilder",
        seen_mints: Set[str],
        rug_event_queue: Optional[asyncio.Queue] = None,
        rug_sell_threshold_sol: float = 15.0,
        price_event_queue: Optional[asyncio.Queue] = None,
        price_event_min_delta: float = 0.02,
    ):
        self._grpc_url = grpc_url
        self._grpc_token = grpc_token
        self._graduation_queue = graduation_queue
        self._kline_builder = kline_builder
        self._seen_mints = seen_mints

        # Rug event detection: when a large SOL-out swap is observed on a
        # pool where we hold an open position, emit an event so the
        # controller can trigger panic sell before M4's poll loop catches
        # up. Empirically our SL triggers at -86% mean PnL (rug completed);
        # gRPC swaps are pushed sub-second so this path can catch the dump
        # one block after it starts.
        self._rug_event_queue = rug_event_queue
        self._rug_sell_threshold_sol = float(rug_sell_threshold_sol)
        # pool_address → mint_address for positions we watch for rugs
        self._watched_pools: Dict[str, str] = {}
        # de-dup: mint → last rug event timestamp (don't emit twice in a row)
        self._rug_event_last_ts: Dict[str, float] = {}

        # Phase 22.S.P1 (2026-04-30) — price event queue for SL/EC triggers.
        # Emits ANY swap on a watched pool to controller's _price_event_watcher
        # so SL/EC fire sub-second instead of waiting for 5s/2s polling cycle.
        # min_delta filter: only emit when price moved ≥ min_delta from
        # last emitted price (per mint). Default 2% — avoids spamming queue
        # with sandwich-tick noise.
        self._price_event_queue = price_event_queue
        self._price_event_min_delta = float(price_event_min_delta)
        # Per-mint last EMITTED price (to enforce min_delta filter)
        self._price_event_last_price: Dict[str, float] = {}

        self._connected = False
        self._task: Optional[asyncio.Task] = None
        self._slot_time_cache: Dict[int, int] = {}  # slot → unix timestamp
        self._seen_sigs: set = set()  # dedup graduation signatures

        # Stats — basic counters
        self._graduations_detected = 0
        self._swaps_injected = 0
        self._reconnect_count = 0

        # Phase 25 Fix #4 + #5 (2026-05-02): detailed drop-point stats.
        # Each silent return None / early return path increments a specific
        # counter so we can identify which drop dominates the 24% gap with
        # Birdeye pump_amm. Periodic log every 5min publishes deltas.
        self._stats: Dict[str, int] = {
            # Stream-level
            "tx_received": 0,                # All txs from gRPC stream
            "tx_processed": 0,               # Successfully reached _process_transaction end
            "tx_processing_errors": 0,       # Exception in _process_transaction
            # Graduation detection
            "graduation_handler_called": 0,
            "graduation_dedup_skipped": 0,   # signature in _seen_sigs
            "graduation_seen_mints_skipped": 0,  # base_mint in _seen_mints (already known)
            "graduation_invalid_accounts": 0,    # ix_accounts < 5
            "graduation_quote_blacklist": 0,     # base_mint in QUOTE_TOKEN_BLACKLIST
            "graduation_quote_not_wsol": 0,      # quote_mint != WSOL_MINT
            "graduation_no_base_mint": 0,        # base_mint empty after resolve
            # Swap routing
            "swap_no_pool_match": 0,             # _find_registered_pools returned empty
            "swap_pool_matched": 0,              # tx routes to _handle_swap
            # Parser failures (silent return None)
            "parse_no_base_deltas": 0,           # base_deltas dict empty
            "parse_no_base_received": 0,         # base_received <= 0
            "parse_no_sol_amount": 0,            # sol_amount <= 0
            # inject_swap drops
            "inject_mint_not_registered": 0,     # KlineBuilder._swaps doesn't have mint
            # Backpressure
            "queue_full_drops": 0,               # When async queue overflows (Fix #1)
            # Reconnect / data loss
            "reconnect_events": 0,
            "reconnect_backfill_swaps": 0,       # Recovered via RPC backfill (Fix #2)
        }
        # Last-stats-log timestamp for periodic deltas
        self._last_stats_log_ts = time.time()
        self._stats_log_interval_sec = 300  # 5 min

        # Phase 25 Fix #1 (2026-05-02): async tx queue to eliminate backpressure.
        # Producer (async for stream) enqueues txs; single consumer drains and
        # runs sync `_process_transaction`. This prevents the gRPC client buffer
        # from filling when processing is slower than stream rate (high-traffic
        # moments). Bounded queue size — drops measured via stats.
        self._tx_queue: Optional[asyncio.Queue] = None  # initialized in _stream_loop
        self._tx_queue_max = 5000  # ~50s buffer at 100 tx/sec
        self._consumer_task: Optional[asyncio.Task] = None

        # Phase 25 Fix #5.1 (2026-05-02): periodic stats snapshot runs as its
        # own bg task so we keep emitting logs even if the gRPC stream
        # produces no updates for >5 min (e.g. quiet period or stalled
        # connection). Previously the snapshot was only triggered on update
        # arrival, hiding silent stalls.
        self._stats_logger_task: Optional[asyncio.Task] = None

        # Phase 25 Fix #2 (2026-05-02): track last-seen swap timestamp per pool
        # for reconnect backfill. Map pool_address → last block_time observed.
        self._pool_last_swap_ts: Dict[str, int] = {}

        # Rolling per-mint swap history (price_sol, timestamp). Position
        # monitor reads median-of-recent as ground-truth pool price, which
        # is robust to single-tick sandwich spikes that Jupiter Price API
        # often picks up on low-liquidity meme tokens.
        self._recent_swap_prices: Dict[str, Deque[Tuple[float, float]]] = {}

        # Rolling per-mint FULL swap records for the 14y shadow-exit-warn
        # hook (stores SwapRecord so feature-compute has is_buy / volume_sol /
        # trader_address / base_amount). Separate from _recent_swap_prices so
        # the existing median-price reader is unaffected. maxlen 400 covers
        # ~10 min at dense trading.
        self._recent_swap_records: Dict[str, Deque] = {}

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self):
        """Launch the gRPC stream as a background task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_event_loop().create_task(self._stream_loop())

    async def stop(self):
        """Cancel the background stream task."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Stop the periodic stats logger as well (Fix #5.1)
        if self._stats_logger_task and not self._stats_logger_task.done():
            self._stats_logger_task.cancel()
            try:
                await self._stats_logger_task
            except (asyncio.CancelledError, Exception):
                pass
            self._stats_logger_task = None
        self._connected = False
        logger.info("gRPC: stream stopped "
                     f"(graduations={self._graduations_detected}, "
                     f"swaps={self._swaps_injected}, "
                     f"reconnects={self._reconnect_count})")

    async def _stream_loop(self):
        """Main loop: connect, subscribe, process, reconnect on error."""
        backoff = 1

        while True:
            try:
                # Import generated stubs (deferred to avoid import errors if not generated)
                from controllers.generic.generated import (
                    geyser_pb2, geyser_pb2_grpc,
                )

                # Build authenticated channel
                channel = self._create_channel()
                stub = geyser_pb2_grpc.GeyserStub(channel)

                # Build subscription request
                request = geyser_pb2.SubscribeRequest(
                    transactions={
                        "pumpswap": geyser_pb2.SubscribeRequestFilterTransactions(
                            vote=False,
                            failed=False,
                            account_include=[PUMPSWAP_AMM_PROGRAM],
                        ),
                    },
                    blocks_meta={
                        "meta": geyser_pb2.SubscribeRequestFilterBlocksMeta(),
                    },
                    commitment=geyser_pb2.CommitmentLevel.CONFIRMED,
                )

                logger.info(f"gRPC: connecting to {self._grpc_url}...")

                # Subscribe — the stub returns an async iterator
                stream = stub.Subscribe(iter([request]))

                self._connected = True
                was_reconnect = self._reconnect_count > 0 and self._swaps_injected > 0
                backoff = 1
                logger.info("gRPC: connected and subscribed to PumpSwap AMM")

                # Fix #1: initialize bounded queue + start consumer task
                self._tx_queue = asyncio.Queue(maxsize=self._tx_queue_max)
                self._consumer_task = asyncio.create_task(self._tx_consumer())

                # Fix #5.1: start the periodic stats logger as its own bg
                # task so it keeps publishing snapshots during quiet periods.
                if self._stats_logger_task is None or self._stats_logger_task.done():
                    self._stats_logger_task = asyncio.create_task(
                        self._periodic_stats_logger())

                # Fix #2: post-reconnect, RPC-poll all registered pools to
                # backfill any swaps that arrived during the disconnect gap.
                # KlineBuilder dedups via _seen_sigs so re-polling is safe.
                if was_reconnect:
                    asyncio.create_task(self._reconnect_backfill())

                async for update in stream:
                    try:
                        # Cache block_time from blocks_meta (fast, inline)
                        if update.HasField("block_meta"):
                            bm = update.block_meta
                            if bm.block_time and bm.block_time.timestamp:
                                self._cache_slot_time(bm.slot, bm.block_time.timestamp)

                        # Enqueue transactions for async processing
                        if update.HasField("transaction"):
                            try:
                                self._tx_queue.put_nowait(update.transaction)
                            except asyncio.QueueFull:
                                # Drop the tx — queue is overwhelmed, but at least
                                # gRPC client can keep reading from server. Better
                                # than blocking the entire stream.
                                self._stats["queue_full_drops"] += 1

                    except Exception as e:
                        logger.debug(f"gRPC: update processing error: {e}")

                # Stream ended (server closed)
                self._connected = False
                logger.warning("gRPC: stream ended by server, reconnecting...")
                await self._cleanup_consumer()

            except asyncio.CancelledError:
                self._connected = False
                await self._cleanup_consumer()
                logger.info("gRPC: stream cancelled")
                return

            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                self._stats["reconnect_events"] += 1
                logger.warning(f"gRPC: disconnected ({type(e).__name__}: {e}), "
                               f"reconnecting in {backoff}s "
                               f"(total reconnects: {self._reconnect_count})")
                await self._cleanup_consumer()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _cleanup_consumer(self):
        """Phase 25 Fix #1.1 (2026-05-02): drain queue, THEN cancel consumer.

        Previously this just cancelled the task, which meant any txs already
        sitting in `_tx_queue` (up to 5000) were silently discarded on every
        reconnect. Now we wait (bounded) for the consumer to drain the queue
        first via `queue.join()`, so already-received txs are not lost.

        Bound the drain at 5s so a stuck consumer can't hang reconnect; any
        residual is then dropped on cancel and counted via `queue_full_drops`
        (best-effort observability).
        """
        if self._tx_queue is not None and self._consumer_task and not self._consumer_task.done():
            try:
                drain_target = self._tx_queue.qsize()
                if drain_target > 0:
                    logger.info(
                        f"gRPC: cleanup draining {drain_target} queued tx "
                        f"before consumer cancel")
                    await asyncio.wait_for(self._tx_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                # Stuck consumer — count remaining as drops and proceed.
                remaining = self._tx_queue.qsize() if self._tx_queue is not None else 0
                if remaining > 0:
                    self._stats["queue_full_drops"] += remaining
                logger.warning(
                    f"gRPC: cleanup drain timed out, dropping {remaining} tx")
            except Exception:
                pass

        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except (asyncio.CancelledError, Exception):
                pass
            self._consumer_task = None

    async def _periodic_stats_logger(self):
        """Phase 25 Fix #5.1 (2026-05-02): emit stats snapshot on a fixed timer.

        Decoupled from update arrivals so quiet streams (or stalled
        connections) still produce a snapshot every 5 min. Self-recovers from
        log exceptions to avoid taking the whole bot down on a formatter bug.
        """
        try:
            while True:
                await asyncio.sleep(self._stats_log_interval_sec)
                try:
                    self._log_stats_snapshot()
                    self._last_stats_log_ts = time.time()
                except Exception:
                    logger.debug("gRPC: stats snapshot error", exc_info=True)
        except asyncio.CancelledError:
            return

    async def _tx_consumer(self):
        """Phase 25 Fix #1: drain tx_queue, run sync _process_transaction.

        Single consumer (not parallel) to avoid races on _kline_builder._swaps.
        Yields control between txs so the event loop stays responsive for the
        producer (async for stream).
        """
        if self._tx_queue is None:
            return
        while True:
            try:
                tx = await self._tx_queue.get()
                try:
                    self._process_transaction(tx)
                except Exception:
                    self._stats["tx_processing_errors"] += 1
                    logger.debug("gRPC: _process_transaction error", exc_info=True)
                finally:
                    self._tx_queue.task_done()
            except asyncio.CancelledError:
                return

    async def _reconnect_backfill(self):
        """Phase 25 Fix #2.1 (2026-05-02): after a reconnect, recover swaps
        that arrived during the disconnect gap.

        Originally this used `kline_builder.poll_swaps(mint)`, which only
        fetches one page of `getSignaturesForAddress` (limit ≤ 1000). On a
        busy pool that covers ~10s — useless for typical reconnect gaps of
        30-180s. The audit found `_pool_last_swap_ts` was tracked but never
        consumed.

        This rewrite:
          1. Walks `_kline_builder._pools` to map mint → pools.
          2. For each mint, looks up the youngest `_pool_last_swap_ts` across
             its pools. The gap = now - that timestamp is the disconnect
             window we need to backfill (with a 1.5× safety margin).
          3. Calls `backfill_swaps(mint, target_span_sec=gap*1.5)` which
             pages backward through signatures (max_pages=10, limit=1000)
             until coverage exceeds the target span.
          4. Mints with no recorded swap timestamp default to a 300s span
             (matches `backfill_swaps` default), since we can't size the
             window for them.

        Idempotent: `seen_sigs` in OnChainKlineBuilder dedups already-seen
        signatures, so re-polling overlap is a no-op.
        """
        if self._kline_builder is None:
            return
        try:
            mints = list(self._kline_builder._pools.keys())
            if not mints:
                return

            now_ts = int(time.time())
            mint_to_span: Dict[str, int] = {}
            for mint, pools in self._kline_builder._pools.items():
                # Most-recent swap across this mint's pools.
                last_ts = 0
                for pool in pools:
                    ts = self._pool_last_swap_ts.get(pool, 0)
                    if ts > last_ts:
                        last_ts = ts
                if last_ts == 0:
                    # No prior swap observed (mint registered but no flow yet);
                    # use 300s default which covers typical reconnect gaps.
                    span = 300
                else:
                    gap = max(0, now_ts - last_ts)
                    # 1.5× safety margin; floor 60s, cap 1800s (30 min) so a
                    # very long disconnect doesn't request thousands of pages.
                    span = max(60, min(1800, int(gap * 1.5)))
                mint_to_span[mint] = span

            avg_span = sum(mint_to_span.values()) // max(len(mint_to_span), 1)
            logger.info(
                f"gRPC backfill: recovering {len(mints)} mints via paged "
                f"backfill_swaps (avg target_span={avg_span}s)")

            total_recovered = 0
            errors = 0
            for mint in mints:
                try:
                    n = await self._kline_builder.backfill_swaps(
                        mint,
                        target_span_sec=mint_to_span[mint],
                        max_pages=10,
                    )
                    total_recovered += n
                except Exception:
                    errors += 1
            self._stats["reconnect_backfill_swaps"] += total_recovered
            logger.info(
                f"gRPC backfill: recovered {total_recovered} swaps "
                f"({errors} errors) across {len(mints)} mints")
        except Exception as e:
            logger.warning(f"gRPC backfill failed: {e}")

    def _log_stats_snapshot(self):
        """Phase 25 Fix #4+#5: dump per-drop-point counters periodically.

        Called every 5min from _stream_loop. Helps diagnose which silent
        drop dominates the Birdeye/Chainstack pump_amm gap.
        """
        s = self._stats
        # Compute key derived metrics
        tx = max(s["tx_received"], 1)
        graduations = max(s["graduation_handler_called"], 1)
        # Of all txs, what fraction touched a registered pool (eligible for swap)?
        match_pct = s["swap_pool_matched"] / tx * 100
        # Of pool-matched, what fraction did parser fail on?
        parse_fail = (s["parse_no_base_deltas"]
                      + s["parse_no_base_received"]
                      + s["parse_no_sol_amount"])
        parse_fail_pct = (parse_fail / max(s["swap_pool_matched"], 1)) * 100
        # Of parse-success, what fraction got dropped at inject?
        inject_drop_pct = (s["inject_mint_not_registered"]
                           / max(s["swap_pool_matched"], 1)) * 100

        logger.info(
            f"gRPC stats snapshot — "
            f"tx={s['tx_received']:,} match={s['swap_pool_matched']:,} "
            f"({match_pct:.2f}%) injected={self._swaps_injected:,} "
            f"parse_fail={parse_fail:,} ({parse_fail_pct:.1f}%) "
            f"inject_drop={s['inject_mint_not_registered']:,} ({inject_drop_pct:.2f}%) "
            f"reconnects={s['reconnect_events']} "
            f"queue_full={s['queue_full_drops']}"
        )
        logger.info(
            f"gRPC stats detail — "
            f"grad_handler={s['graduation_handler_called']} "
            f"(dedup={s['graduation_dedup_skipped']}, "
            f"seen_mint={s['graduation_seen_mints_skipped']}, "
            f"invalid_acct={s['graduation_invalid_accounts']}, "
            f"quote_blacklist={s['graduation_quote_blacklist']}, "
            f"not_wsol={s['graduation_quote_not_wsol']}, "
            f"no_base_mint={s['graduation_no_base_mint']}) | "
            f"parse: no_base_deltas={s['parse_no_base_deltas']}, "
            f"no_base_received={s['parse_no_base_received']}, "
            f"no_sol_amount={s['parse_no_sol_amount']} | "
            f"backfill_recovered={s['reconnect_backfill_swaps']}"
        )

    def _create_channel(self) -> grpc.aio.Channel:
        """Create an authenticated gRPC channel."""
        # Parse URL: strip protocol prefix if present
        url = self._grpc_url
        for prefix in ("https://", "http://", "grpc://"):
            if url.startswith(prefix):
                url = url[len(prefix):]
                break

        # Build credentials
        auth_metadata = grpc.metadata_call_credentials(
            lambda _, callback: callback(
                (("x-token", self._grpc_token),), None
            )
        )
        channel_creds = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(), auth_metadata
        )

        # Phase 26.R1 (2026-05-06) — gRPC stability tuning.
        # Audit (audit_swap_stream_lag.py + 6h log analysis) found Chainstack
        # Yellowstone shared endpoint resets stream every 8-15 min via
        # RST_STREAM (HTTP/2 error 1/2). Tuned config below pushes reconnect
        # interval longer + handles GOAWAY gracefully.
        options = [
            # Keepalive — longer to reduce server-side ping pressure.
            # Chainstack throttles clients that ping too aggressively.
            ("grpc.keepalive_time_ms", 60_000),       # 30→60s
            ("grpc.keepalive_timeout_ms", 20_000),    # 10→20s
            ("grpc.keepalive_permit_without_calls", True),
            # HTTP/2 — allow unlimited pings while there's data flow.
            ("grpc.http2.max_pings_without_data", 0),
            ("grpc.http2.min_time_between_pings_ms", 60_000),       # 10→60s
            ("grpc.http2.min_ping_interval_without_data_ms", 60_000),
            # HTTP/2 frame tuning — large frames reduce per-frame overhead
            # and lower CPU when receiving high-throughput Yellowstone stream.
            ("grpc.http2.max_frame_size", 1 << 20),   # 1 MB (default 16 KB)
            ("grpc.http2.write_buffer_size", 1 << 20),
            # Built-in retry — gRPC handles transient errors before raising.
            ("grpc.enable_retries", 1),
            # Existing — keep.
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),  # 64 MB
        ]

        return grpc.aio.secure_channel(url, channel_creds, options=options)

    def _cache_slot_time(self, slot: int, timestamp: int):
        """Cache slot → block_time mapping with LRU eviction."""
        self._slot_time_cache[slot] = timestamp
        if len(self._slot_time_cache) > SLOT_CACHE_MAX:
            # Evict oldest entries
            oldest = sorted(self._slot_time_cache.keys())[:SLOT_CACHE_MAX // 2]
            for s in oldest:
                del self._slot_time_cache[s]

    def _get_block_time(self, slot: int) -> int:
        """Get block_time for a slot.

        Phase 26.X (2026-05-11): cache miss no longer falls back to wall
        clock (which caused +1-3s systematic bias on stored swaps.timestamp;
        empirical: 71/100 production rows off by exactly +1s, mean +3.47s
        vs RPC blockTime ground truth). Estimate from nearest cached
        (slot, time) anchor via Solana ~0.4s slot time. Wall-clock
        fallback retained ONLY for cold-start (cache empty) case.
        """
        cached = self._slot_time_cache.get(slot)
        if cached is not None:
            return cached

        if not self._slot_time_cache:
            # Cold start: no anchor — fall back to wall clock with warning.
            logger.warning(
                f"_get_block_time: cache empty, slot={slot} → wall clock fallback"
            )
            return int(time.time())

        # Find nearest cached (slot, time) anchor; estimate via slot delta.
        nearest_slot = min(
            self._slot_time_cache.keys(), key=lambda s: abs(s - slot)
        )
        nearest_time = self._slot_time_cache[nearest_slot]
        return nearest_time + int(round((slot - nearest_slot) * SLOT_TIME_SEC))

    # ────────────────────────────────────────────────────────────────────
    # Transaction routing
    # ────────────────────────────────────────────────────────────────────

    def _process_transaction(self, tx_update):
        """Route a PumpSwap transaction to graduation detection or swap injection."""
        self._stats["tx_received"] += 1
        slot = tx_update.slot
        block_time = self._get_block_time(slot)
        tx_info = tx_update.transaction

        # Decode signature
        sig_bytes = bytes(tx_info.signature)
        signature = _b58encode(sig_bytes)

        # Build account keys list
        tx = tx_info.transaction
        meta = tx_info.meta
        msg = tx.message

        account_keys = [
            _b58encode(bytes(k))
            for k in msg.account_keys
        ]
        # Append loaded addresses (versioned transactions)
        for addr in meta.loaded_writable_addresses:
            account_keys.append(_b58encode(bytes(addr)))
        for addr in meta.loaded_readonly_addresses:
            account_keys.append(_b58encode(bytes(addr)))

        # Track which registered pool this tx touches (for swap injection)
        pool_mint_map = self._find_registered_pools(account_keys)

        # Scan inner instructions for PumpSwap AMM
        is_graduation = False
        for inner_group in meta.inner_instructions:
            for ix in inner_group.instructions:
                if ix.program_id_index >= len(account_keys):
                    continue
                program_id = account_keys[ix.program_id_index]
                if program_id != PUMPSWAP_AMM_PROGRAM:
                    continue

                ix_data = bytes(ix.data)
                if len(ix_data) >= 8 and ix_data[:8] == PUMPSWAP_CREATE_POOL_DISC:
                    # Graduation (create_pool)
                    self._handle_graduation(
                        account_keys, ix, signature, slot, block_time
                    )
                    is_graduation = True

        # If not a graduation and touches a registered pool, inject as swap
        if not is_graduation:
            if pool_mint_map:
                self._stats["swap_pool_matched"] += 1
                self._handle_swap(
                    account_keys, meta, pool_mint_map, signature, block_time
                )
            else:
                self._stats["swap_no_pool_match"] += 1
        self._stats["tx_processed"] += 1

    def _find_registered_pools(self, account_keys: list) -> Dict[str, str]:
        """Check if any account in the tx is a registered KlineBuilder pool.

        Returns: {pool_address: mint} for each match.
        """
        if not self._kline_builder:
            return {}

        result = {}
        # Build reverse lookup: pool_address → mint
        for mint, pools in self._kline_builder._pools.items():
            for pool in pools:
                if pool in account_keys:
                    result[pool] = mint
        return result

    # ────────────────────────────────────────────────────────────────────
    # Graduation detection (M1 replacement)
    # ────────────────────────────────────────────────────────────────────

    def _handle_graduation(self, account_keys, inner_ix, signature, slot, block_time):
        """Parse a create_pool inner instruction and queue a ChainGraduation."""
        from controllers.generic.meme_sniper_utils import ChainGraduation

        self._stats["graduation_handler_called"] += 1

        # Dedup
        if signature in self._seen_sigs:
            self._stats["graduation_dedup_skipped"] += 1
            return
        self._seen_sigs.add(signature)
        if len(self._seen_sigs) > 5000:
            self._seen_sigs = set(list(self._seen_sigs)[-2500:])

        # Extract accounts from inner instruction
        ix_accounts = list(inner_ix.accounts)  # bytes → list of uint8 indices
        if len(ix_accounts) < 5:
            self._stats["graduation_invalid_accounts"] += 1
            return

        def resolve(idx):
            return account_keys[idx] if idx < len(account_keys) else ""

        base_mint = resolve(ix_accounts[ACCOUNT_IDX_BASE_MINT])
        quote_mint = resolve(ix_accounts[ACCOUNT_IDX_QUOTE_MINT])
        pool = resolve(ix_accounts[ACCOUNT_IDX_POOL])
        creator = resolve(ix_accounts[ACCOUNT_IDX_CREATOR])

        if not base_mint:
            self._stats["graduation_no_base_mint"] += 1
            return

        # Guard: base_mint cannot be a known quote token
        if base_mint in QUOTE_TOKEN_BLACKLIST:
            self._stats["graduation_quote_blacklist"] += 1
            logger.debug(f"gRPC: rejected graduation base_mint={base_mint[:12]}... "
                         f"in QUOTE_TOKEN_BLACKLIST")
            return

        # Guard: quote must be WSOL
        if quote_mint != WSOL_MINT:
            self._stats["graduation_quote_not_wsol"] += 1
            logger.debug(f"gRPC: rejected graduation quote_mint={quote_mint[:12]}... "
                         f"(expected WSOL)")
            return

        # Skip if already seen this mint
        if base_mint in self._seen_mints:
            self._stats["graduation_seen_mints_skipped"] += 1
            return

        grad = ChainGraduation(
            mint=base_mint, pool=pool, creator=creator,
            quote_mint=quote_mint, signature=signature,
            block_time=block_time, slot=slot,
        )

        # Phase 22.D Route X RC2 (2026-05-01): pre-register pool with
        # KlineBuilder IMMEDIATELY at graduation detection. Previously the
        # controller's M1 main loop (graduation_poll_interval=10s) was the
        # only register path, creating a 0-10s window where gRPC saw swap
        # txs for this pool but `_find_registered_pools` returned empty
        # (pool not yet in `_kline_builder._pools`) and silently dropped
        # them. Measured first-swap-lag p50 was 18s vs Birdeye's 1s.
        # With register here the gap closes to a single block (~1s).
        # `register()` is idempotent — controller will call it again
        # later and it'll be a no-op.
        if pool and self._kline_builder is not None:
            try:
                self._kline_builder.register(base_mint, pool)
            except Exception as e:
                logger.debug(
                    f"gRPC: kline_builder.register pre-graduation failed for "
                    f"{base_mint[:12]}... ({e}) — controller will retry on M1 poll")

        logger.info(f"gRPC: graduation detected — {base_mint[:16]}... "
                     f"pool={pool[:16]}...")

        self._graduation_queue.put_nowait(grad)
        self._graduations_detected += 1

    # ────────────────────────────────────────────────────────────────────
    # Swap injection (M2 replacement)
    # ────────────────────────────────────────────────────────────────────

    def get_swap_prices_batch(self, mints: list) -> Dict[str, float]:
        """Return median of last-7 swap prices (≤15s old) per mint.

        Median-of-recent is robust to single-tick sandwich spikes. Mints
        without any fresh swaps are simply absent from the result — callers
        fall back to Jupiter Price API for those.
        """
        cutoff = time.time() - 15.0
        result: Dict[str, float] = {}
        for mint in mints:
            dq = self._recent_swap_prices.get(mint)
            if not dq:
                continue
            recent = [px for px, ts in dq if ts >= cutoff]
            if recent:
                result[mint] = float(median(recent[-7:]))
        return result

    def get_recent_swap_records(self, mint: str, max_age_sec: float = 600.0) -> list:
        """14y shadow-exit-warn hook accessor — returns full SwapRecord list
        for `mint`, filtered to records within `max_age_sec` of now.

        Each returned record has: timestamp (block_time), price_sol,
        volume_sol, is_buy, base_amount, trader_address. Ordered oldest
        → newest. Returns [] if mint not seen or buffer empty.
        """
        dq = self._recent_swap_records.get(mint)
        if not dq:
            return []
        cutoff = time.time() - max_age_sec
        # SwapRecord.timestamp is block_time (seconds). Filter and return.
        return [r for r in dq if r.timestamp >= cutoff]

    def is_dip_confirmed(self, mint: str, entry_price: float,
                          threshold_pnl_pct: float,
                          window_sec: float = 5.0,
                          min_confirmations: int = 3):
        """B6 fix (2026-05-09): require N confirming swaps below threshold
        before firing SL/EC. Mirror image of get_peak_price_since.

        Returns True if there are ≥ min_confirmations swaps in the last
        window_sec where pnl <= threshold_pnl_pct AND price within
        sandwich gate (≤ 120% of candidate price). This filters out single
        transient dip ticks that bounce back within 5s — 87% of SL trades
        in audit recovered > 30% within 10min after exit, indicating SL
        triggered on momentary dips.

        Args:
            mint: token mint
            entry_price: position entry price (for pnl calc)
            threshold_pnl_pct: SL/EC threshold (negative, e.g. -0.30)
            window_sec: window to look for confirming swaps
            min_confirmations: number of swaps required at or below threshold

        Returns:
            True if dip is confirmed (real price drop), False if transient
        """
        if entry_price <= 0:
            return True  # safety: can't validate, allow exit
        dq = self._recent_swap_records.get(mint)
        if not dq:
            return False  # no data → don't fire SL on stale state
        cutoff_ts = time.time() - window_sec
        recent = [r for r in dq if r.timestamp >= cutoff_ts and r.price_sol > 0]
        if len(recent) < min_confirmations:
            # Not enough recent ticks at all → don't fire on partial data
            return False
        below_threshold = 0
        for r in recent:
            pnl = (r.price_sol - entry_price) / entry_price
            if pnl <= threshold_pnl_pct:
                below_threshold += 1
        return below_threshold >= min_confirmations

    def get_peak_price_since(self, mint: str, since_ts: float,
                              sandwich_window_sec: float = 5.0,
                              min_confirmations: int = 2):
        # Returns Optional[float] — sandwich-filtered max price since since_ts.
        """Sandwich-filtered max price since `since_ts` (B5 fix 2026-05-09).

        Median-based `get_swap_prices_batch` (15s + median-of-7) misses
        intra-tick spikes (5s pumps that complete between polls), causing
        peak/trail logic to miss legitimate +20% peaks. v2 sim audit showed
        this caused 20× divergence in trailing exit rate (sim 81% / live 4%).

        This method returns max price WITH sandwich protection: a price tick
        only counts if there are ≥ `min_confirmations` other swaps within
        `sandwich_window_sec` at >= 80% of that tick's price. Single-tick
        Jupiter sandwich spikes (BK postmortem) are rejected by this gate.

        Args:
            mint: token mint
            since_ts: only consider swaps with block_time >= since_ts (e.g.
                position entry timestamp)
            sandwich_window_sec: window around each candidate tick to look
                for confirmation swaps
            min_confirmations: number of other swaps within window required
                at >= 80% of candidate price (default 2 → at least 3 swaps
                within 5s confirming the high)

        Returns:
            float (max confirmed price) or None if no confirmed peak found
        """
        dq = self._recent_swap_records.get(mint)
        if not dq:
            return None
        valid = [r for r in dq if r.timestamp >= since_ts and r.price_sol > 0]
        if len(valid) < min_confirmations + 1:
            return None
        confirmed_max = None
        for r in valid:
            nearby = [v for v in valid
                      if abs(v.timestamp - r.timestamp) <= sandwich_window_sec
                      and v.price_sol >= 0.80 * r.price_sol]
            # nearby includes r itself; need >= min_confirmations + 1 total
            if len(nearby) >= min_confirmations + 1:
                if confirmed_max is None or r.price_sol > confirmed_max:
                    confirmed_max = r.price_sol
        return float(confirmed_max) if confirmed_max is not None else None

    # ─── Rug event watched-pool management ──────────────────────────────
    def watch_pool(self, pool_address: str, mint_address: str) -> None:
        """Start watching a pool for large SOL-out swaps (rug signal).

        Called on position open. Detection fires only once per mint (event
        consumers should stop watching after the panic-sell is initiated).
        """
        if not pool_address or not mint_address:
            return
        self._watched_pools[pool_address] = mint_address
        logger.debug(
            f"gRPC: watching pool {pool_address[:12]}... for rug (mint={mint_address[:12]}...)")

    def unwatch_pool(self, pool_address: str) -> None:
        """Stop watching a pool. Called on position close / exit."""
        removed = self._watched_pools.pop(pool_address, None)
        if removed:
            self._rug_event_last_ts.pop(removed, None)
            self._price_event_last_price.pop(removed, None)

    def _handle_swap(self, account_keys, meta, pool_mint_map, signature, block_time):
        """Parse a swap transaction and inject SwapRecord into KlineBuilder.

        Also: if this swap touches a watched pool AND it's a large SOL-out
        (a rug-size dump), emit a rug event to the queue for the controller
        to act on ASAP (panic sell).
        """
        from controllers.generic.meme_sniper_utils import SwapRecord

        # Determine which base_mint this swap is for
        # (a tx might touch multiple registered pools, but typically just one)
        for pool_addr, base_mint in pool_mint_map.items():
            swap = self._parse_swap_from_grpc(
                account_keys, meta, base_mint, block_time
            )
            if swap:
                # inject_swap is silent if mint not in _swaps; check explicitly
                if base_mint in self._kline_builder._swaps:
                    self._kline_builder.inject_swap(base_mint, swap, signature)
                    self._swaps_injected += 1
                    # Fix #2: track per-pool last-seen timestamp for backfill
                    if block_time:
                        prev = self._pool_last_swap_ts.get(pool_addr, 0)
                        if block_time > prev:
                            self._pool_last_swap_ts[pool_addr] = block_time
                else:
                    self._stats["inject_mint_not_registered"] += 1
                dq = self._recent_swap_prices.get(base_mint)
                if dq is None:
                    dq = deque(maxlen=40)
                    self._recent_swap_prices[base_mint] = dq
                dq.append((swap.price_sol, time.time()))

                # 14y shadow-exit-warn full-record buffer
                dq_full = self._recent_swap_records.get(base_mint)
                if dq_full is None:
                    dq_full = deque(maxlen=400)
                    self._recent_swap_records[base_mint] = dq_full
                dq_full.append(swap)

                # Rug detection: sell (not buy) on a watched pool, large SOL amount.
                if (self._rug_event_queue is not None
                        and pool_addr in self._watched_pools
                        and not swap.is_buy
                        and swap.volume_sol >= self._rug_sell_threshold_sol):
                    # De-dup: one event per mint per 5 seconds
                    now = time.time()
                    last = self._rug_event_last_ts.get(base_mint, 0.0)
                    if now - last >= 5.0:
                        self._rug_event_last_ts[base_mint] = now
                        event = {
                            "mint_address": base_mint,
                            "pool_address": pool_addr,
                            "sol_amount": float(swap.volume_sol),
                            "price_sol": float(swap.price_sol),
                            "signature": signature,
                            "block_time": block_time,
                            "detected_at": now,
                        }
                        try:
                            self._rug_event_queue.put_nowait(event)
                            logger.warning(
                                f"gRPC: RUG EVENT — mint={base_mint[:12]}... "
                                f"pool={pool_addr[:12]}... "
                                f"sol_out={swap.volume_sol:.2f} "
                                f"(threshold={self._rug_sell_threshold_sol:.1f})")
                        except asyncio.QueueFull:
                            logger.warning(
                                f"gRPC: rug event queue full — dropped {base_mint[:12]}")

                # Phase 22.S.P1: Price event for SL/EC fast-trigger.
                # Emit ANY swap on watched pool when |price_delta| >= min_delta
                # since last emitted. Controller's _price_event_watcher will
                # check SL/EC thresholds and fire exit if crossed.
                if (self._price_event_queue is not None
                        and pool_addr in self._watched_pools
                        and swap.price_sol > 0):
                    last_p = self._price_event_last_price.get(base_mint, 0.0)
                    delta = (abs(swap.price_sol - last_p) / last_p
                              if last_p > 0 else 1.0)
                    if last_p == 0 or delta >= self._price_event_min_delta:
                        self._price_event_last_price[base_mint] = swap.price_sol
                        evt = {
                            "mint_address": base_mint,
                            "pool_address": pool_addr,
                            "price_sol": float(swap.price_sol),
                            "is_buy": bool(swap.is_buy),
                            "sol_amount": float(swap.volume_sol),
                            "block_time": block_time,
                            "detected_at": time.time(),
                        }
                        try:
                            self._price_event_queue.put_nowait(evt)
                        except asyncio.QueueFull:
                            # silent drop — controller will catch on next poll
                            pass

    def _parse_swap_from_grpc(self, account_keys, meta, base_mint, block_time):
        """Parse swap price/volume/direction from gRPC transaction meta.

        Mirrors OnChainKlineBuilder._parse_swap_from_tx() but reads from protobuf.
        """
        from controllers.generic.meme_sniper_utils import SwapRecord

        # Step 1: Build token balance deltas
        pre_map = {}  # account_index → {mint, owner, amount}
        for tb in meta.pre_token_balances:
            amount = tb.ui_token_amount.ui_amount
            if amount == 0 and tb.ui_token_amount.amount:
                raw = int(tb.ui_token_amount.amount)
                decimals = tb.ui_token_amount.decimals
                amount = raw / (10 ** decimals) if decimals > 0 else float(raw)
            pre_map[tb.account_index] = {
                "mint": tb.mint, "owner": tb.owner, "amount": float(amount),
            }

        post_map = {}
        for tb in meta.post_token_balances:
            amount = tb.ui_token_amount.ui_amount
            if amount == 0 and tb.ui_token_amount.amount:
                raw = int(tb.ui_token_amount.amount)
                decimals = tb.ui_token_amount.decimals
                amount = raw / (10 ** decimals) if decimals > 0 else float(raw)
            post_map[tb.account_index] = {
                "mint": tb.mint, "owner": tb.owner, "amount": float(amount),
            }

        # Compute deltas grouped by owner
        all_indices = set(list(pre_map.keys()) + list(post_map.keys()))
        base_deltas = {}  # owner → delta
        wsol_deltas = {}

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
            elif mint_addr == WSOL_MINT:
                wsol_deltas[owner] = wsol_deltas.get(owner, 0) + delta

        if not base_deltas:
            self._stats["parse_no_base_deltas"] += 1
            return None

        # Step 2: base_amount
        base_received = sum(d for d in base_deltas.values() if d > 0)
        if base_received <= 0:
            self._stats["parse_no_base_received"] += 1
            return None
        base_amount = base_received

        # Step 3: SOL amount (WSOL deltas + native lamport changes)
        wsol_positive = sum(d for d in wsol_deltas.values() if d > 0)
        wsol_negative = sum(abs(d) for d in wsol_deltas.values() if d < 0)
        wsol_amount = max(wsol_positive, wsol_negative)

        native_sol_amount = 0.0
        pre_bal = list(meta.pre_balances)
        post_bal = list(meta.post_balances)
        if pre_bal and post_bal:
            sol_received = 0.0
            sol_spent = 0.0
            for i in range(min(len(pre_bal), len(post_bal))):
                delta_lamports = post_bal[i] - pre_bal[i]
                if delta_lamports > 100_000:
                    sol_received += delta_lamports
                elif delta_lamports < -100_000:
                    sol_spent += abs(delta_lamports)
            native_sol_amount = max(sol_received, sol_spent) / 1e9

        sol_amount = max(wsol_amount, native_sol_amount)
        if sol_amount <= 0:
            self._stats["parse_no_sol_amount"] += 1
            return None

        # Phase 22.D Route X RC1 (2026-05-01): dust filter REMOVED at parse time.
        # Birdeye keeps sub-0.01 SOL trades and v3.3 was trained on that
        # distribution. Filtering here at parse time was causing 22% of
        # Birdeye-equivalent volume to be silently dropped from Chainstack
        # storage, contributing to ~20pp of the train-vs-live data drift.
        # Bar-build still applies DUST_THRESHOLD_SOL (Layer 1 filter in
        # OnChainKlineBuilder.build_kline) for OHLCV computation; raw swaps
        # remain in DB for full-fidelity downstream features.
        # Original check (kept for reference):
        #     if sol_amount < DUST_THRESHOLD_SOL:
        #         return None

        # Step 4: Direction (buy vs sell)
        # First signer is at account_keys[0]
        signer = account_keys[0] if account_keys else None
        if signer and signer in base_deltas:
            is_buy = base_deltas[signer] > 0
        else:
            largest = max(base_deltas.items(), key=lambda x: x[1])
            is_buy = largest[1] > 0

        price_sol = sol_amount / base_amount

        # Phase 22.E Route Y (2026-05-07): in-line outlier guard.
        # If recent prices exist for this mint and new price is >100x or
        # <0.01x median, drop the swap. Prevents downstream poisoning of
        # OHLCV bars + V5.x feature panels. Bootstrap-safe: skip when <5
        # recent prices.
        recent_dq = self._recent_swap_prices.get(base_mint)
        if recent_dq and len(recent_dq) >= 5:
            recent_prices = sorted(p for p, _ in recent_dq if p > 0)
            if recent_prices:
                median_p = recent_prices[len(recent_prices) // 2]
                if median_p > 0:
                    ratio = price_sol / median_p
                    if ratio > 100.0 or ratio < 0.01:
                        self._stats["parse_price_outlier"] = (
                            self._stats.get("parse_price_outlier", 0) + 1
                        )
                        # Sample warning for first 20 rejections
                        if self._stats["parse_price_outlier"] <= 20:
                            logger.warning(
                                f"[parse_outlier] reject mint={base_mint[:12]} "
                                f"price={price_sol:.4e} median={median_p:.4e} "
                                f"ratio={ratio:.2e}"
                            )
                        return None

        # Phase 22.E (2026-05-08 v2): align trader_address with training pipeline.
        # Training (09d_normalize_backfill_swaps.py:312) extracts Birdeye `owner`
        # field — the wallet whose post-token-balance changed (= actual user).
        # Live previously used account_keys[0] (signer/fee payer) which differs
        # from owner in JITO bundles, MEV sandwiches, and aggregator routes.
        # Caused 2-3.6× drift on v5.3 hc_n_net_long_t / hc_top3 / hc_hhi.
        #
        # v2 fix: prefer signer when signer is also in base_deltas (= direct
        # trade where signer owns the user's token account, common case ≈80%).
        # Only fall back to max-abs-delta when signer isn't in base_deltas
        # (bundled tx / aggregator routing where bundler signs for user).
        # Avoids accidentally picking pool authority (same |delta| as user)
        # in the common case.
        if signer and signer in base_deltas:
            trader = signer
        elif base_deltas:
            # Bundled / aggregated: signer didn't move tokens. Pick the
            # owner whose token delta is largest in absolute value as
            # best-effort approximation of the user.
            trader = max(base_deltas.items(), key=lambda x: abs(x[1]))[0]
        else:
            trader = signer or ""

        return SwapRecord(
            timestamp=block_time,
            price_sol=price_sol,
            volume_sol=sol_amount,
            is_buy=is_buy,
            base_amount=base_amount,
            trader_address=trader or "",
        )
