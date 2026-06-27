import signal
import time  # used for sleeping between cycles
from concurrent.futures import ThreadPoolExecutor  # lets us run multiple markets in parallel on separate threads
from typing import Dict, List, Tuple  # type hints
import threading  # used to create stop signals for each worker

import requests  # HTTP library — used to catch network errors

from ..factories import create_api           # builds the Kalshi API client from config
from ..logging_utils import build_logger     # creates a named logger for terminal output
from ..selection.scoring import safe_float, select_top_markets  # market ranking functions
from .cleanup import stop_worker_then_cancel, cancel_resting_orders_for_ticker  # gracefully shuts down a worker and cancels its orders
from .workers import run_market_worker        # the function each market thread runs


def run_dynamic_strategy(dynamic_config: Dict):
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    logger = build_logger("DynamicSelector", dynamic_config.get("log_level", "INFO"))  # create a logger named "DynamicSelector"

    selector_cfg = dynamic_config.get("market_selector", {})
    refresh_seconds = safe_float(selector_cfg.get("refresh_seconds", 20), 20.0)
    orphan_check_seconds = safe_float(selector_cfg.get("orphan_check_seconds", 120.0), 120.0)
    series_ticker = selector_cfg.get("series_ticker")
    mve_filter = selector_cfg.get("mve_filter", "exclude")
    page_limit = int(selector_cfg.get("page_limit", 250))
    max_pages = int(selector_cfg.get("max_pages", 5))
    max_markets = int(selector_cfg.get("max_markets", 1250))

    selector_api = create_api(dynamic_config.get("api", {}), logger, market_ticker="DYNAMIC")  # create one API client just for market scanning

    # Startup order/position cleanup disabled — preserve all existing orders and positions across restarts.
    try:
        stale_orders = selector_api.list_all_resting_orders()
        if stale_orders:
            logger.warning(f"Startup: found {len(stale_orders)} resting orders — preserving (no startup cancellation)")
        stale_positions = [p for p in selector_api.list_all_positions() if float(p.get("position", p.get("position_fp", 0))) != 0]
        if stale_positions:
            logger.warning(f"Startup: found {len(stale_positions)} existing positions — skipping liquidation (preserve_positions=true)")
    except Exception as e:
        logger.error(f"Startup cleanup failed: {e}")

    active_workers: Dict[str, Tuple[threading.Event, object]] = {}
    max_workers = int(selector_cfg.get("top_n", 8)) + 1
    last_selected_tickers: List[str] = []
    _selector_wake = threading.Event()
    shared_risk_state = {
        "active_markets": 1,
        "_lock": threading.Lock(),       # guards total_realized_pnl updates across worker threads
        "_selector_wake": _selector_wake, # lets a worker wake the selector early on global stop
    }
    selector_backoff_seconds = 5.0
    max_selector_backoff_seconds = 120.0
    last_orphan_check_time: float = 0.0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # create a pool of threads to run workers
        try:
            while True:  # run forever until Ctrl+C
                # Fast-path exit: a worker may have set global_stop while the selector was sleeping.
                # Check before spending API calls on a market scan we'd immediately discard.
                if shared_risk_state.get("global_stop", False):
                    logger.warning(
                        f"Global drawdown limit reached (portfolio_pnl=${shared_risk_state.get('total_realized_pnl', 0.0):+.4f}). "
                        f"Stopping all workers."
                    )
                    break

                markets: List[Dict] = []             # will hold all markets fetched from Kalshi
                selected_tickers = last_selected_tickers  # default to last known good selection

                try:
                    markets = selector_api.list_all_open_markets(  # fetch all open markets from Kalshi
                        series_ticker=series_ticker,               # filter by series if set
                        mve_filter=mve_filter,                     # exclude combo/MVE markets
                        page_limit=page_limit,                     # markets per page
                        max_pages=max_pages,                       # max pages to fetch
                        max_markets=max_markets,                   # hard cap
                    )
                    # Supplemental targeted fetches — some high-value series fall outside the broad cap
                    targeted_series = selector_cfg.get("targeted_series", [])
                    if targeted_series:
                        seen = {m.get("ticker") for m in markets}
                        for ts in targeted_series:
                            try:
                                extra = selector_api.list_all_open_markets(
                                    series_ticker=ts, mve_filter=mve_filter,
                                    page_limit=page_limit, max_pages=2, max_markets=200,
                                )
                                for m in extra:
                                    if m.get("ticker") not in seen:
                                        markets.append(m)
                                        seen.add(m.get("ticker"))
                            except Exception:
                                pass
                    ranked = select_top_markets(markets, selector_cfg)             # score and rank all markets
                    selected_tickers = [ticker for ticker, _, _, _ in ranked]      # extract just the ticker names from the ranking
                    last_selected_tickers = selected_tickers                       # save for fallback next cycle
                    selector_backoff_seconds = 5.0                                 # reset backoff after a successful call
                except requests.exceptions.HTTPError as http_error:
                    status_code = http_error.response.status_code if http_error.response is not None else None  # get HTTP error code
                    if status_code == 429:                                         # rate limited — Kalshi told us to slow down
                        logger.warning(
                            f"Selector rate-limited (429). Reusing previous selection and backing off for "
                            f"{selector_backoff_seconds:.1f}s"
                        )
                        time.sleep(selector_backoff_seconds)                       # wait before retrying
                        selector_backoff_seconds = min(
                            selector_backoff_seconds * 2,                          # double the wait time each time
                            max_selector_backoff_seconds,                          # but never wait more than 120s
                        )
                    else:
                        logger.error(f"Selector HTTP error ({status_code}): {http_error}")  # some other HTTP error
                        time.sleep(selector_backoff_seconds)                                  # wait before retrying
                except requests.exceptions.RequestException as request_exception:
                    logger.error(f"Selector request error: {request_exception}")   # network error (timeout, DNS failure, etc.)
                    time.sleep(selector_backoff_seconds)                            # wait before retrying

                selected_set = set(selected_tickers)                               # convert to set for fast lookup
                shared_risk_state["active_markets"] = max(1, len(selected_tickers))  # update how many markets are active
                logger.info(f"Selector found {len(markets)} open markets; selected: {selected_tickers}")  # log the selection

                # Global drawdown limit: any worker can set this flag to stop all trading
                if shared_risk_state.get("global_stop", False):
                    logger.warning(
                        f"Global drawdown limit reached (portfolio_pnl=${shared_risk_state.get('total_realized_pnl', 0.0):+.4f}). "
                        f"Stopping all workers."
                    )
                    break

                for ticker in list(active_workers.keys()):     # check every currently running worker
                    stop_event, future = active_workers[ticker] # get its stop signal and thread handle

                    if future.done() and ticker in selected_set:  # worker finished (normally or crashed) but market still selected
                        exc = future.exception()
                        if exc is not None:
                            logger.error(f"Worker for {ticker} crashed: {exc}; cancelling stale orders before restart")
                            cancel_resting_orders_for_ticker(ticker, dynamic_config, logger)
                        else:
                            logger.info(f"Worker for {ticker} completed its time window — restarting")
                        del active_workers[ticker]                # remove so it gets restarted below
                        continue

                    if ticker not in selected_set:              # this market is no longer in our top selection
                        logger.warning(f"Draining deselected ticker {ticker}: stop worker then cancel resting orders")
                        is_clean = stop_worker_then_cancel(     # tell the worker to stop and cancel all its open orders
                            ticker,
                            stop_event,
                            future,
                            dynamic_config,
                            logger,
                        )
                        if is_clean:                            # worker stopped and orders cancelled successfully
                            del active_workers[ticker]          # remove it from our tracking dict
                        else:
                            logger.error(
                                f"Could not fully clean up {ticker}; keeping worker state for next retry cycle"
                            )                                   # something went wrong — try again next cycle

                for ticker in selected_tickers:                 # for each market we want to trade
                    if ticker not in active_workers:            # no worker running for it yet
                        logger.info(f"Starting worker for selected ticker {ticker}")
                        stop_event = threading.Event()          # create a stop signal for this worker
                        future = executor.submit(               # launch the worker on a background thread
                            run_market_worker,                  # the function to run
                            ticker,                             # which market to trade
                            dynamic_config,                     # full config
                            stop_event,                         # so we can stop it later
                            shared_risk_state,                  # shared risk info
                        )
                        active_workers[ticker] = (stop_event, future)  # track this worker

                # Orphan order watchdog: cancel resting orders on tickers NOT actively managed.
                # Runs on its own faster cycle (orphan_check_seconds) independent of refresh_seconds.
                now_ts = time.time()
                if now_ts - last_orphan_check_time >= orphan_check_seconds:
                    last_orphan_check_time = now_ts
                    try:
                        managed = set(active_workers.keys())
                        all_resting = selector_api.list_all_resting_orders()
                        orphan_orders = [o for o in all_resting if o.get("ticker") not in managed]
                        if orphan_orders:
                            logger.warning(f"Orphan watchdog: {len(orphan_orders)} orders on unmanaged tickers")
                            for o in orphan_orders:
                                try:
                                    ok = selector_api.cancel_order(o["order_id"])
                                    if ok:
                                        logger.warning(f"Orphan cancelled: {o.get('ticker')} {o.get('action')} {o['order_id']}")
                                    else:
                                        logger.warning(f"Orphan cancel deferred (trading paused): {o.get('ticker')} {o['order_id']}")
                                except Exception as e:
                                    logger.warning(f"Orphan cancel error for {o.get('ticker')} {o['order_id']}: {e}")
                    except Exception as e:
                        logger.error(f"Orphan watchdog error: {e}")

                # Interruptible sleep: a worker can set _selector_wake to break out early
                # (e.g. when global_stop fires) instead of waiting the full refresh window.
                _selector_wake.wait(timeout=refresh_seconds)
                _selector_wake.clear()

        except KeyboardInterrupt:  # user pressed Ctrl+C
            logger.info("Received keyboard interrupt, shutting down dynamic strategy")
        finally:                   # always runs on exit, even after Ctrl+C
            for ticker in list(active_workers.keys()):          # shut down every running worker
                stop_event, future = active_workers[ticker]
                logger.warning(f"Final shutdown cleanup for {ticker}")
                stop_worker_then_cancel(ticker, stop_event, future, dynamic_config, logger)  # cancel all open orders before exiting
            selector_api.logout()  # log out of the API cleanly
