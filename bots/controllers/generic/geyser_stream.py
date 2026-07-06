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
import os
import random
import time
from collections import OrderedDict, deque
from statistics import median
from typing import TYPE_CHECKING, Deque, Dict, Optional, Set, Tuple

import grpc

if TYPE_CHECKING:
    from controllers.generic.cleaning_service_bridge import CleaningServiceBridge
    from controllers.generic.meme_sniper_utils import (
        ChainGraduation, OnChainKlineBuilder, SwapRecord,
    )

logger = logging.getLogger(__name__)

# ── Solana / PumpSwap protocol primitives (canonical: ms.solana_protocol) ────
# Production loads this module as ``controllers.generic.geyser_stream`` (sys.path
# root = /home/hummingbot), so the canonical package is ``controllers.generic.ms``.
# The in-container test channel does ``sys.path.insert(0, controllers/generic)``
# and imports it flat as ``geyser_stream``, where the package is bare ``ms``.
# Try the production-qualified path first, fall back to the flat one.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.solana_protocol import (
        PUMPSWAP_AMM_PROGRAM,
        PUMPSWAP_CREATE_POOL_DISC,
        WSOL_MINT,
        USDC_MINT,
        USDT_MINT,
        QUOTE_TOKEN_BLACKLIST,
        ACCOUNT_IDX_POOL,
        ACCOUNT_IDX_CREATOR,
        ACCOUNT_IDX_BASE_MINT,
        ACCOUNT_IDX_QUOTE_MINT,
        b58encode as _b58encode,
        b58encode_cached as _b58encode_cached,
    )
except ImportError:  # test channel / flat import: controllers/generic on sys.path
    from ms.solana_protocol import (
        PUMPSWAP_AMM_PROGRAM,
        PUMPSWAP_CREATE_POOL_DISC,
        WSOL_MINT,
        USDC_MINT,
        USDT_MINT,
        QUOTE_TOKEN_BLACKLIST,
        ACCOUNT_IDX_POOL,
        ACCOUNT_IDX_CREATOR,
        ACCOUNT_IDX_BASE_MINT,
        ACCOUNT_IDX_QUOTE_MINT,
        b58encode as _b58encode,
        b58encode_cached as _b58encode_cached,
    )

# Phase 3a (2026-05-31): SLOT_CACHE_MAX / SLOT_TIME_SEC moved to ms.data.slot_time_cache.
# Dual-context import mirrors the production/test-channel pattern used throughout.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.data.slot_time_cache import SlotTimeCache
except ImportError:  # in-container test channel (flat ms package)
    from ms.data.slot_time_cache import SlotTimeCache  # type: ignore[no-redef]

# Phase 3b (2026-05-31): swap deques + 4 query methods extracted to ms.data.stream_swap_store.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.data.stream_swap_store import StreamSwapStore
except ImportError:  # in-container test channel (flat ms package)
    from ms.data.stream_swap_store import StreamSwapStore  # type: ignore[no-redef]

# Phase 3c (2026-05-31): _parse_swap_from_grpc extracted to ms.data.parse_swap.
try:  # production: controllers package root on sys.path
    from controllers.generic.ms.data.parse_swap import parse_swap_from_grpc
except ImportError:  # in-container test channel (flat ms package)
    from ms.data.parse_swap import parse_swap_from_grpc  # type: ignore[no-redef]

# Dust threshold for swap volume (SOL)
DUST_THRESHOLD_SOL = 0.01


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
        cleaning_service_bridge: Optional["CleaningServiceBridge"] = None,
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

        # Phase B.1 (2026-05-18) — cleaning_service bridge shadow tap.
        # When set, _handle_swap mirrors every parsed swap into the bridge
        # (which forwards to CleaningService.ingest_swap). None = disabled,
        # zero overhead on hot path (single null check per swap).
        self._cleaning_service_bridge = cleaning_service_bridge

        self._connected = False
        self._task: Optional[asyncio.Task] = None
        self._slot_cache = SlotTimeCache()  # Phase 3a: extracted to ms.data.slot_time_cache
        self._slot_time_cache = self._slot_cache.cache  # alias → same dict object (line 746/767/842 unchanged)
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
            # Phase B1 (2026-05-23): PROCESSED commitment phantom monitor
            "b1_phantom_sampled": 0,             # Total sigs sampled (10%)
            "b1_phantom_confirmed": 0,           # Sig found via RPC getSignatureStatuses
            "b1_phantom_orphaned": 0,            # Sig NOT found 60s after inject → rolled back
            "b1_phantom_check_errors": 0,        # RPC call failed
            # Phase B4 (2026-05-23): slot_time_cache pre-warm
            "b4_prewarm_success": 0,             # # slots successfully cached at startup
            "b4_prewarm_attempted": 0,           # # slots attempted
            "b4_prewarm_elapsed_ms": 0,          # Total pre-warm wall-clock time (ms)
            "b4_prewarm_done": 0,                # 1 = pre-warm completed, 0 = pending/disabled
        }
        # Last-stats-log timestamp for periodic deltas
        self._last_stats_log_ts = time.time()
        self._stats_log_interval_sec = 300  # 5 min

        # Phase R5 (2026-06-10): silent-stall watchdog. The gRPC server can
        # hold the connection open while delivering ZERO messages (no
        # exception) — `async for update in stream` then blocks forever and
        # `connected` keeps lying True. This wedged the feed for ~30h on
        # 2026-06-09 (tx counter frozen, mints=0 → peak_pnl_pct stuck at 0 →
        # profit-taking + trailing-activation disabled, positions rode to the
        # 900s time_limit). blocks_meta streams every block (~400ms), so N
        # seconds with no update of ANY kind = an unambiguous stall → force a
        # fresh subscribe. The pre-existing reconnect path only fires on a hard
        # disconnect/exception, never on a silent stall.
        self._stall_timeout_sec = float(
            os.getenv("GEYSER_STALL_TIMEOUT_SEC", "90"))
        # tx_received value at the previous stats snapshot — used to alert when
        # a stall survives reconnect (an upstream feed outage the watchdog's
        # reconnect can't cure; see _log_stats_snapshot).
        self._last_snapshot_tx_received = 0

        # Phase B1 (2026-05-23): PROCESSED commitment phantom monitor.
        # Track 10% sample of injected sigs → 60s later, query RPC
        # getSignatureStatuses to confirm tx finalized vs rolled-back.
        # If orphan rate > 5%, _periodic_stats_logger raises CRITICAL alert.
        self._b1_pending_sigs: "OrderedDict[str, Tuple[float, int]]" = OrderedDict()
        self._b1_sample_rate = 0.10            # 10% of injected sigs (per F11)
        self._b1_check_delay_sec = 60          # wait 60s after inject before query
        self._b1_orphan_threshold = 0.05       # 5% → CRITICAL alert
        self._b1_pending_cap = 10000            # LRU cap防内存
        self._b1_orphan_task: Optional[asyncio.Task] = None
        # Try CHAINSTACK_SOLANA_RPC first (current env var), fallback to
        # SOLANA_RPC_URL for forward compat. Empty string disables B1 monitor.
        self._b1_rpc_endpoint = (
            os.getenv("CHAINSTACK_SOLANA_RPC", "")
            or os.getenv("SOLANA_RPC_URL", "")
        )
        self._b1_rpc_session = None             # httpx.AsyncClient, lazy init

        # Phase B4 (2026-05-23): slot_time_cache startup pre-warm. Eliminates
        # +1-3s estimation bias on cold start (per §13 F2). Idempotent flag
        # ensures pre-warm only runs once per process, not per stream
        # reconnect.
        self._b4_prewarm_n_slots = 2000          # spec §2.2
        self._b4_prewarm_concurrency = 20        # parallel RPC limit (Chainstack 250 RPS)
        self._b4_prewarmed = False               # idempotent flag

        # Phase 14.2 (2026-05-23): lat_breakdown_e2e instrumentation. 1% sample
        # of injected swaps writes structured log line `LAT_BREAKDOWN ...` with
        # per-stage timing (block→grpc / grpc→consumer / consumer→inject). Daily
        # cron parses logs for p50/p90 trend. NO API change (uses logger only).
        self._lat_sample_rate = 0.01             # 1% per spec §14.2
        # Per-tx trace state (set by _tx_consumer, read by _handle_swap).
        # Safe because _process_transaction is sync inside async consumer task.
        self._cur_trace_grpc_t = 0.0
        self._cur_trace_consumer_t = 0.0

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

        # Phase 3b (2026-05-31): swap deques + query methods live in StreamSwapStore.
        # Alias the public dicts back onto self so all existing parser write-paths
        # (_recent_swap_prices lines 1331/1334/1527 and _recent_swap_records
        # lines 1338/1341) remain byte-identical — they write through the same
        # dict object that StreamSwapStore holds.
        # NOTE: StreamSwapStore.__init__ receives self._kline_builder, so this
        # block must come AFTER self._kline_builder = kline_builder above.
        self._swap_store = StreamSwapStore(self._kline_builder)
        # Rolling per-mint swap history (price_sol, timestamp). Position
        # monitor reads median-of-recent as ground-truth pool price, which
        # is robust to single-tick sandwich spikes that Jupiter Price API
        # often picks up on low-liquidity meme tokens.
        self._recent_swap_prices = self._swap_store.recent_swap_prices
        # Rolling per-mint FULL swap records for the 14y shadow-exit-warn
        # hook (stores SwapRecord so feature-compute has is_buy / volume_sol /
        # trader_address / base_amount). Separate from _recent_swap_prices so
        # the existing median-price reader is unaffected. maxlen 400 covers
        # ~10 min at dense trading.
        self._recent_swap_records = self._swap_store.recent_swap_records

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
        # Audit P0 #3 fix (2026-05-23): cancel B1 phantom orphan check task
        # and close httpx session to prevent TCP socket leak on restart.
        if self._b1_orphan_task and not self._b1_orphan_task.done():
            self._b1_orphan_task.cancel()
            try:
                await self._b1_orphan_task
            except (asyncio.CancelledError, Exception):
                pass
            self._b1_orphan_task = None
        if self._b1_rpc_session is not None:
            try:
                await self._b1_rpc_session.aclose()
            except Exception as e:
                logger.debug(f"B1: rpc session close err: {e}")
            self._b1_rpc_session = None
        self._connected = False
        logger.info("gRPC: stream stopped "
                     f"(graduations={self._graduations_detected}, "
                     f"swaps={self._swaps_injected}, "
                     f"reconnects={self._reconnect_count})")

    async def _stream_loop(self):
        """Main loop: connect, subscribe, process, reconnect on error."""
        backoff = 1

        while True:
            channel = None
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
                    # Phase B1 (2026-05-23) — was CONFIRMED, switched to PROCESSED.
                    # Saves ~2-5s of Solana cluster voting wait. ~1-3% txs may
                    # later rollback; tracked via _b1_phantom_orphan_check.
                    # If orphan_rate > 5%, see logger.critical in _log_stats_snapshot
                    # to revert to CONFIRMED.
                    commitment=geyser_pb2.CommitmentLevel.PROCESSED,
                )

                logger.info(f"gRPC: connecting to {self._grpc_url}...")

                # Subscribe — the stub returns an async iterator
                stream = stub.Subscribe(iter([request]))

                self._connected = True
                was_reconnect = self._reconnect_count > 0 and self._swaps_injected > 0
                logger.info(
                    "gRPC: connected and subscribed to PumpSwap AMM "
                    "(B1 ACTIVE: commitment=PROCESSED, sample=10%, orphan_threshold=5%)"
                )

                # Fix #1: initialize bounded queue + start consumer task
                self._tx_queue = asyncio.Queue(maxsize=self._tx_queue_max)
                self._consumer_task = asyncio.create_task(self._tx_consumer())

                # Phase B1: launch phantom orphan check task (idempotent)
                if (self._b1_orphan_task is None or self._b1_orphan_task.done()) \
                        and self._b1_rpc_endpoint:
                    self._b1_orphan_task = asyncio.create_task(
                        self._b1_phantom_orphan_check()
                    )

                # Phase B4: kick off slot_time_cache pre-warm (once per process)
                # Non-blocking, fire-and-forget. _b4_prewarmed flag prevents
                # repeat on stream reconnect.
                if not self._b4_prewarmed:
                    asyncio.create_task(self._b4_prewarm_slot_cache())

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

                # Phase R5 (2026-06-10): read with a per-message timeout instead
                # of a bare `async for`. A silent stall (server keeps the
                # connection open but stops sending) would otherwise block the
                # `__anext__` forever with no exception — the 2026-06-09 zombie.
                stream_iter = stream.__aiter__()
                while True:
                    try:
                        update = await asyncio.wait_for(
                            stream_iter.__anext__(),
                            timeout=self._stall_timeout_sec,
                        )
                    except asyncio.TimeoutError:
                        # No update of ANY kind (tx OR blocks_meta) for
                        # stall_timeout_sec while blocks_meta should arrive every
                        # ~400ms = a wedged feed. Break out to force a fresh
                        # subscribe via the reconnect path below.
                        self._stats["stall_reconnects"] = (
                            self._stats.get("stall_reconnects", 0) + 1)
                        logger.warning(
                            f"gRPC: no update for {self._stall_timeout_sec:.0f}s "
                            f"(SILENT STALL, tx_received={self._stats['tx_received']:,}) "
                            f"— forcing reconnect "
                            f"(stall_reconnects={self._stats['stall_reconnects']})"
                        )
                        break
                    except StopAsyncIteration:
                        break  # stream ended by server (old `async for` exit)

                    try:
                        # First update received — connection is healthy; reset backoff.
                        backoff = 1
                        # Cache block_time from blocks_meta (fast, inline)
                        if update.HasField("block_meta"):
                            bm = update.block_meta
                            if bm.block_time and bm.block_time.timestamp:
                                self._cache_slot_time(bm.slot, bm.block_time.timestamp)

                        # Enqueue transactions for async processing.
                        # Phase 14.2: tag with grpc_received_t for instrumentation.
                        if update.HasField("transaction"):
                            grpc_received_t = time.time()
                            try:
                                self._tx_queue.put_nowait(
                                    (update.transaction, grpc_received_t)
                                )
                            except asyncio.QueueFull:
                                # Drop the tx — queue is overwhelmed, but at least
                                # gRPC client can keep reading from server. Better
                                # than blocking the entire stream.
                                self._stats["queue_full_drops"] += 1

                    except Exception as e:
                        logger.debug(f"gRPC: update processing error: {e}")

                # Stream ended (server closed) or stalled (watchdog broke out).
                # was_reconnect (line above) re-triggers _reconnect_backfill on
                # the next iteration to recover swaps missed during the gap.
                self._connected = False
                logger.warning("gRPC: stream ended/stalled, reconnecting...")
                await self._cleanup_consumer()
                await self._close_channel(channel)
                self._reconnect_count += 1
                self._stats["reconnect_events"] += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

            except asyncio.CancelledError:
                self._connected = False
                await self._cleanup_consumer()
                await self._close_channel(channel)
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
                await self._close_channel(channel)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _close_channel(self, channel):
        """Best-effort close of a gRPC channel on reconnect (Phase R5).

        The pre-existing loop abandoned the channel on every reconnect (rare:
        4 in the whole run). The stall watchdog can now reconnect every
        ~stall_timeout_sec during a sustained upstream outage, so explicitly
        close the old channel to avoid leaking sockets/threads. Guarded — a
        close failure must never break the reconnect loop.
        """
        if channel is None:
            return
        try:
            await channel.close()
        except Exception:
            pass

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
                item = await self._tx_queue.get()
                # Phase 14.2: queue items are (tx, grpc_received_t) tuples.
                # Backward compat: bare tx (legacy) treated as grpc_received_t=0.
                if isinstance(item, tuple) and len(item) == 2:
                    tx, grpc_received_t = item
                else:
                    tx, grpc_received_t = item, 0.0
                self._cur_trace_grpc_t = grpc_received_t
                self._cur_trace_consumer_t = time.time()
                try:
                    self._process_transaction(tx)
                except Exception:
                    self._stats["tx_processing_errors"] += 1
                    logger.debug("gRPC: _process_transaction error", exc_info=True)
                finally:
                    # Phase 14.2: clear trace state to avoid leaking across txs
                    self._cur_trace_grpc_t = 0.0
                    self._cur_trace_consumer_t = 0.0
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
            mints = self._kline_builder.registered_mints()
            if not mints:
                return

            now_ts = int(time.time())
            mint_to_span: Dict[str, int] = {}
            for mint, pools in self._kline_builder.pools_by_mint().items():
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

    # ─── Phase B1 (2026-05-23) — PROCESSED commitment phantom monitor ──
    async def _b1_phantom_orphan_check(self):
        """B1 safety net: every 30s, query RPC getSignatureStatuses for sigs
        that were injected ≥60s ago. If status=None → tx was orphaned (rolled
        back), increment counter. If orphan_rate > 5% sustained, _log_stats_snapshot
        raises CRITICAL to manually revert.

        10% sampling (per F11) → ~33K RPC call/day = 0.84% Chainstack Growth quota.
        Batch size 100 sigs/call.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("B1: httpx not available, phantom check disabled")
            return
        if self._b1_rpc_session is None:
            self._b1_rpc_session = httpx.AsyncClient(timeout=10.0)
        while True:
            try:
                await asyncio.sleep(30)
                now = time.time()
                # Find sigs ≥ 60s old
                ready = [
                    sig for sig, (ts, _slot) in self._b1_pending_sigs.items()
                    if now - ts >= self._b1_check_delay_sec
                ]
                if not ready:
                    continue
                # Batch up to 100 sigs per RPC call
                batch = ready[:100]
                try:
                    statuses = await self._b1_get_signature_statuses(batch)
                except Exception as e:
                    self._stats["b1_phantom_check_errors"] += 1
                    logger.debug(f"B1: RPC getSignatureStatuses err: {e}")
                    continue
                if statuses is None:
                    self._stats["b1_phantom_check_errors"] += 1
                    continue
                # Count + clean up
                for sig, status in zip(batch, statuses):
                    if status is None:
                        # tx not found 60s after PROCESSED inject → orphaned
                        self._stats["b1_phantom_orphaned"] += 1
                    else:
                        self._stats["b1_phantom_confirmed"] += 1
                    self._b1_pending_sigs.pop(sig, None)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.debug(f"B1 phantom loop err: {e}")

    async def _b1_get_signature_statuses(
        self, sigs: list
    ) -> Optional[list]:
        """RPC getSignatureStatuses helper. Returns list of status dicts (or None
        per sig if not found). One RPC call for up to 256 sigs (Solana spec).

        Uses httpx.AsyncClient (lazy-init on first call). Returns None on
        network/parse failure (caller increments b1_phantom_check_errors).
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignatureStatuses",
            "params": [sigs, {"searchTransactionHistory": True}],
        }
        try:
            resp = await self._b1_rpc_session.post(
                self._b1_rpc_endpoint, json=payload
            )
            data = resp.json()
            # Audit P1 #7 fix (2026-05-23): RPC may legitimately return {} or
            # missing 'result' key on error. Distinguish:
            #   - 'result' missing/None → RPC error → return None
            #   - 'result' present (even with empty 'value' list) → valid response
            rpc_result = data.get("result")
            if rpc_result is None:
                return None  # RPC-level error
            return rpc_result.get("value", [])  # valid response, may be []
        except Exception:
            return None

    # ─── Phase B4 (2026-05-23) — slot_time_cache pre-warm ──
    async def _b4_prewarm_slot_cache(self):
        """B4: pre-fill slot_time_cache via RPC getBlockTime to eliminate
        cold-start estimation bias (+1-3s per §13 F2).

        Idempotent: runs once per process. Reuses _b1_rpc_session (httpx).
        Parallel via semaphore (Chainstack 250 RPS, we use 20 concurrent).

        Cost: ~2000 RPC calls × 5 RU = 10K RU per startup ≈ 0.05% Chainstack
        Growth monthly quota.
        Wall-clock: ~5-10s with 20-way parallelism.
        """
        if self._b4_prewarmed:
            return  # already done
        # Audit P0 #1 fix (2026-05-23): do NOT set _b4_prewarmed = True here.
        # If RPC fails, flag would stay True and pre-warm wouldn't retry on
        # next reconnect. Set flag only AFTER successful pre-warm (end of try).

        if not self._b1_rpc_endpoint:
            logger.warning("B4: no RPC endpoint, slot pre-warm disabled")
            self._b4_prewarmed = True  # permanent disable, no retry without env var
            return
        try:
            import httpx
        except ImportError:
            logger.warning("B4: httpx not available, slot pre-warm disabled")
            self._b4_prewarmed = True  # permanent disable
            return
        if self._b1_rpc_session is None:
            self._b1_rpc_session = httpx.AsyncClient(timeout=10.0)
        session = self._b1_rpc_session

        t_start = time.time()
        try:
            # Step 1: get current slot
            r = await session.post(self._b1_rpc_endpoint, json={
                "jsonrpc": "2.0", "id": 1, "method": "getSlot",
                "params": [{"commitment": "confirmed"}],
            })
            current_slot = r.json().get("result")
            if not current_slot or not isinstance(current_slot, int):
                logger.warning("B4: failed to get current_slot, will retry on next reconnect")
                self._stats["b4_prewarm_done"] = -1  # mark as failed (per P2 #9)
                return  # _b4_prewarmed stays False → retry on reconnect

            # Step 2: parallel fetch block_time for last N slots
            n_slots = self._b4_prewarm_n_slots
            sem = asyncio.Semaphore(self._b4_prewarm_concurrency)

            async def fetch_one(slot: int) -> bool:
                async with sem:
                    try:
                        rr = await session.post(self._b1_rpc_endpoint, json={
                            "jsonrpc": "2.0", "id": 1, "method": "getBlockTime",
                            "params": [slot],
                        })
                        t = rr.json().get("result")
                        if t and isinstance(t, int):
                            self._slot_time_cache[slot] = t
                            return True
                    except Exception:
                        pass
                    return False

            slots = list(range(current_slot - n_slots, current_slot))
            results = await asyncio.gather(
                *[fetch_one(s) for s in slots], return_exceptions=True
            )
            success = sum(1 for r in results if r is True)

            elapsed_ms = int((time.time() - t_start) * 1000)
            self._stats["b4_prewarm_success"] = success
            self._stats["b4_prewarm_attempted"] = n_slots
            self._stats["b4_prewarm_elapsed_ms"] = elapsed_ms
            self._stats["b4_prewarm_done"] = 1

            logger.info(
                f"B4 active: slot_time_cache pre-warmed {success}/{n_slots} "
                f"slots in {elapsed_ms}ms ({success / n_slots:.0%} hit rate, "
                f"cache_size={len(self._slot_time_cache)})"
            )
            # Audit P0 #1 fix: only set flag after successful pre-warm
            self._b4_prewarmed = True
        except Exception as e:
            logger.error(f"B4: pre-warm failed: {e}", exc_info=True)
            self._stats["b4_prewarm_done"] = -1  # mark failure (per P2 #9)
            # _b4_prewarmed stays False → retry on next reconnect

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

        # Phase R5 (2026-06-10): silent-stall alert (defense-in-depth behind
        # the read-loop watchdog). If tx_received has not advanced since the
        # previous snapshot while we still believe we're connected, the feed is
        # wedged AND reconnect is not curing it (e.g. an upstream Chainstack/
        # geyser outage the watchdog cannot fix by resubscribing). logger.critical
        # so monitor_grpc_stability.sh / log alerting catches it within one
        # 5-min snapshot instead of ~30h later (the 2026-06-09 incident).
        if (self._connected
                and s["tx_received"] == self._last_snapshot_tx_received
                and s["tx_received"] > 0):
            logger.critical(
                f"gRPC STALL ALERT: tx_received frozen at {s['tx_received']:,} "
                f"across a full {self._stats_log_interval_sec}s snapshot while "
                f"connected=True (stall_reconnects={s.get('stall_reconnects', 0)}, "
                f"reconnects={s['reconnect_events']}). Feed wedged — peak_pnl_pct "
                f"+ rug fast-path are degraded; check the geyser/Chainstack "
                f"subscription (a restart re-establishes it)."
            )
        self._last_snapshot_tx_received = s["tx_received"]

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

        # Phase O7 (2026-05-23) — b58 encoding LRU cache hit rate
        try:
            info = _b58encode_cached.cache_info()
            total = info.hits + info.misses
            if total > 0:
                hit_rate = info.hits / total
                logger.info(
                    f"O7 stats — b58 cache hits={info.hits:,} misses={info.misses:,} "
                    f"hit_rate={hit_rate:.1%} currsize={info.currsize}/{info.maxsize}"
                )
        except Exception:
            pass

        # Phase B4 (2026-05-23) — slot_time_cache pre-warm (one-time at startup)
        if s["b4_prewarm_done"]:
            # Only log B4 if pre-warm completed (one-time, suppress after first log)
            if not hasattr(self, "_b4_logged_once"):
                self._b4_logged_once = True
                logger.info(
                    f"B4 stats — pre-warmed={s['b4_prewarm_success']}/{s['b4_prewarm_attempted']} "
                    f"slots in {s['b4_prewarm_elapsed_ms']}ms "
                    f"(cache_size={len(self._slot_time_cache)})"
                )

        # Phase B1 (2026-05-23) — PROCESSED commitment phantom monitor
        b1_total = s["b1_phantom_confirmed"] + s["b1_phantom_orphaned"]
        if b1_total > 0:
            orph_rate = s["b1_phantom_orphaned"] / b1_total
            logger.info(
                f"B1 stats — sampled={s['b1_phantom_sampled']:,} "
                f"confirmed={s['b1_phantom_confirmed']:,} "
                f"orphaned={s['b1_phantom_orphaned']:,} "
                f"orphan_rate={orph_rate:.2%} "
                f"check_errors={s['b1_phantom_check_errors']:,} "
                f"pending_buf={len(self._b1_pending_sigs)}"
            )
            # Auto-alert: > 5% sustained → revert to CONFIRCMED needed
            if b1_total >= 100 and orph_rate > self._b1_orphan_threshold:
                logger.critical(
                    f"B1 ALERT: phantom orphan_rate={orph_rate:.2%} exceeds "
                    f"threshold {self._b1_orphan_threshold:.0%} "
                    f"(orphaned={s['b1_phantom_orphaned']:,} / "
                    f"total={b1_total:,}). Consider manual revert: "
                    f"geyser_stream.py:288 PROCESSED → CONFIRMED + redeploy."
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
        """Cache slot → block_time mapping with LRU eviction. Phase 3a: delegates to SlotTimeCache."""
        self._slot_cache.cache_slot_time(slot, timestamp)

    def _get_block_time(self, slot: int) -> int:
        """Get block_time for a slot. Phase 3a: delegates to SlotTimeCache."""
        return self._slot_cache.get_block_time(slot)

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

        # Phase O7 (2026-05-23): use LRU-cached b58encode for account_keys.
        # System programs (Token / AssociatedToken / Compute Budget) repeat
        # across every tx → high cache hit rate. Sig stays uncached above
        # (each tx unique). bytes() conversion is hashable input for lru_cache.
        account_keys = [
            _b58encode_cached(bytes(k))
            for k in msg.account_keys
        ]
        # Append loaded addresses (versioned transactions)
        for addr in meta.loaded_writable_addresses:
            account_keys.append(_b58encode_cached(bytes(addr)))
        for addr in meta.loaded_readonly_addresses:
            account_keys.append(_b58encode_cached(bytes(addr)))

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

        Phase O3 (2026-05-23): O(k) loop via KlineBuilder.find_mint_by_pool
        reverse index, instead of original O(n×m×k) (n=mints, m=pools/mint,
        k=accounts). Speedup ~50× under normal load (n=50-200, m=1-3, k=30-100).
        """
        if not self._kline_builder:
            return {}
        result = {}
        # O3: O(k) loop. Each iteration is O(1) dict lookup via reverse index.
        find = self._kline_builder.find_mint_by_pool  # local binding for speed
        for acct in account_keys:
            mint = find(acct)
            if mint is not None:
                result[acct] = mint
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
        return self._swap_store.get_swap_prices_batch(mints)

    def get_recent_swap_records(self, mint: str, max_age_sec: float = 600.0) -> list:
        """14y shadow-exit-warn hook accessor — returns full SwapRecord list
        for `mint`, filtered to records within `max_age_sec` of now.

        Each returned record has: timestamp (block_time), price_sol,
        volume_sol, is_buy, base_amount, trader_address. Ordered oldest
        → newest. Returns [] if mint not seen or buffer empty.

        Buffer-freshness fix (2026-05-16): production VWMP audit found 89/345
        fallback ticks (26%) had ≥3 qualifying swaps in `swaps` archive but
        missing from `_recent_swap_records`. Root cause: backfill swaps
        (meme_sniper_utils.py:_backfill_single_pool) populate
        kline_builder._swaps but NOT this deque (see line 395 comment). Post-
        restart bot has no historical entries until live gRPC fills them.
        Fix: when caller wants recent records, merge live deque with kline
        builder's in-memory swap list (superset for registered mints,
        includes RPC backfill). Dedupe by signature, else by
        (ts, price, vol) tuple. List snapshot first per VWMP §3.3 GIL pattern.
        """
        return self._swap_store.get_recent_swap_records(mint, max_age_sec)

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
        return self._swap_store.is_dip_confirmed(
            mint, entry_price, threshold_pnl_pct, window_sec, min_confirmations
        )

    def is_peak_confirmed(self, mint: str, peak_price: float,
                           window_sec: float = None,
                           min_confirmations: int = 2,
                           tol_pct: float = 0.05,
                           vol_floor: float = 0.1):
        """P2 corroboration (2026-06-04; v2 2026-06-05, SHADOW-ONLY). Delegates to StreamSwapStore.
        Was the arming peak real (>=N distinct traders, each >=vol_floor SOL) or a phantom spike?"""
        return self._swap_store.is_peak_confirmed(
            mint, peak_price, window_sec=window_sec,
            min_confirmations=min_confirmations, tol_pct=tol_pct, vol_floor=vol_floor
        )

    def is_level_confirmed(self, mint: str, entry_price: float,
                            level_pnl_pct: float,
                            window_sec: float = 5.0,
                            min_confirmations: int = 3,
                            vol_floor: float = 0.1):
        """P2 corroboration (2026-06-04; v2 2026-06-05, SHADOW-ONLY). Delegates to StreamSwapStore.
        Is the current fire-level a persistent drop (>=N distinct traders, each >=vol_floor SOL)?"""
        return self._swap_store.is_level_confirmed(
            mint, entry_price, level_pnl_pct,
            window_sec=window_sec, min_confirmations=min_confirmations, vol_floor=vol_floor
        )

    def window_swap_count(self, mint: str, window_sec: float = 5.0):
        """P2 corroboration (2026-06-04). Delegates to StreamSwapStore — price-valid
        swaps in window, for shadow-eval tape-availability conditioning."""
        return self._swap_store.window_swap_count(mint, window_sec=window_sec)

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
        return self._swap_store.get_peak_price_since(
            mint, since_ts, sandwich_window_sec, min_confirmations
        )

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
                if self._kline_builder.has_swaps(base_mint):
                    self._kline_builder.inject_swap(base_mint, swap, signature)
                    self._swaps_injected += 1
                    # Fix #2: track per-pool last-seen timestamp for backfill
                    if block_time:
                        prev = self._pool_last_swap_ts.get(pool_addr, 0)
                        if block_time > prev:
                            self._pool_last_swap_ts[pool_addr] = block_time
                    # Phase B1 (2026-05-23): tag 10% of injected sigs for
                    # phantom orphan check (60s later via RPC).
                    # P0 fix (2026-05-23): tx_update.slot was undefined in
                    # _handle_swap scope; drop slot, use block_time as proxy
                    # (sufficient for phantom check — only signature matters).
                    if (self._b1_rpc_endpoint
                            and random.random() < self._b1_sample_rate):
                        self._b1_pending_sigs[signature] = (
                            time.time(), int(block_time or 0)
                        )
                        self._stats["b1_phantom_sampled"] += 1
                        if len(self._b1_pending_sigs) > self._b1_pending_cap:
                            self._b1_pending_sigs.popitem(last=False)

                    # Phase 14.2 (2026-05-23): 1% sample end-to-end latency
                    # breakdown. Structured log line (no DB write — daily cron
                    # parses log). Skip if no grpc_received_t (legacy queue) or
                    # block_time missing (would render delivery delta useless).
                    if (random.random() < self._lat_sample_rate
                            and self._cur_trace_grpc_t > 0
                            and block_time):
                        inject_done_t = time.time()
                        block_to_grpc_ms = (self._cur_trace_grpc_t - block_time) * 1000
                        grpc_to_consumer_ms = (
                            self._cur_trace_consumer_t - self._cur_trace_grpc_t) * 1000
                        consumer_to_inject_ms = (
                            inject_done_t - self._cur_trace_consumer_t) * 1000
                        logger.info(
                            f"LAT_BREAKDOWN "
                            f"block_to_grpc_ms={block_to_grpc_ms:.0f} "
                            f"grpc_to_consumer_ms={grpc_to_consumer_ms:.1f} "
                            f"consumer_to_inject_ms={consumer_to_inject_ms:.1f} "
                            f"block_time={int(block_time)} "
                            f"sig={signature[:16]}"
                        )
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

                # Phase B.1: mirror swap into cleaning_service via bridge.
                # Defensive — never let cleaning_service break the gRPC path.
                # Bridge.on_swap itself swallows ingest exceptions; we add
                # one more try/except as belt-and-braces for unexpected
                # attribute errors during partial rollouts.
                if self._cleaning_service_bridge is not None:
                    try:
                        self._cleaning_service_bridge.on_swap(base_mint, swap)
                    except Exception:
                        pass

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

        Delegates to the module-level parse_swap_from_grpc() (ms.data.parse_swap,
        Phase 3c). stats and recent_swap_prices are passed by reference so all
        counter increments and deque reads are reflected on the original objects.
        """
        return parse_swap_from_grpc(
            account_keys, meta, base_mint, block_time,
            self._stats, self._recent_swap_prices,
        )
