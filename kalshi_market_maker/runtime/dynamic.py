import time  # used for sleeping between cycles
from concurrent.futures import ThreadPoolExecutor  # lets us run multiple markets in parallel on separate threads
from typing import Dict, List, Tuple  # type hints
import threading  # used to create stop signals for each worker

import requests  # HTTP library — used to catch network errors

from ..factories import create_api           # builds the Kalshi API client from config
from ..logging_utils import build_logger     # creates a named logger for terminal output
from ..selection.scoring import safe_float, select_top_markets  # market ranking functions
from .cleanup import stop_worker_then_cancel  # gracefully shuts down a worker and cancels its orders
from .workers import run_market_worker        # the function each market thread runs


def run_dynamic_strategy(dynamic_config: Dict):
    logger = build_logger("DynamicSelector", dynamic_config.get("log_level", "INFO"))  # create a logger named "DynamicSelector"

    selector_cfg = dynamic_config.get("market_selector", {})                 # pull the market selector section from config
    refresh_seconds = safe_float(selector_cfg.get("refresh_seconds", 20), 20.0)  # how often to re-scan all markets (default 20s)
    series_ticker = selector_cfg.get("series_ticker")                         # optional filter — only look at one event series (e.g. "FED")
    mve_filter = selector_cfg.get("mve_filter", "exclude")                    # tell Kalshi API to exclude combo markets
    page_limit = int(selector_cfg.get("page_limit", 250))                     # how many markets to fetch per API page
    max_pages = int(selector_cfg.get("max_pages", 5))                         # max pages to fetch (250 x 5 = 1250 markets)
    max_markets = int(selector_cfg.get("max_markets", 1250))                  # hard cap on total markets to consider

    selector_api = create_api(dynamic_config.get("api", {}), logger, market_ticker="DYNAMIC")  # create one API client just for market scanning
    active_workers: Dict[str, Tuple[threading.Event, object]] = {}            # tracks which markets have a running worker thread
    max_workers = int(selector_cfg.get("top_n", 8)) + 1                       # thread pool size = top_n markets + 1 for the selector
    last_selected_tickers: List[str] = []                                     # remembers last selection in case the API fails
    shared_risk_state = {"active_markets": 1}                                 # shared dict — tells each worker how many markets are running
    selector_backoff_seconds = 5.0                                            # how long to wait after a failed API call
    max_selector_backoff_seconds = 120.0                                      # maximum wait time if errors keep happening

    with ThreadPoolExecutor(max_workers=max_workers) as executor:  # create a pool of threads to run workers
        try:
            while True:  # run forever until Ctrl+C
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

                for ticker in list(active_workers.keys()):     # check every currently running worker
                    stop_event, future = active_workers[ticker] # get its stop signal and thread handle
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

                time.sleep(refresh_seconds)  # wait before the next market scan cycle

        except KeyboardInterrupt:  # user pressed Ctrl+C
            logger.info("Received keyboard interrupt, shutting down dynamic strategy")
        finally:                   # always runs on exit, even after Ctrl+C
            for ticker in list(active_workers.keys()):          # shut down every running worker
                stop_event, future = active_workers[ticker]
                logger.warning(f"Final shutdown cleanup for {ticker}")
                stop_worker_then_cancel(ticker, stop_event, future, dynamic_config, logger)  # cancel all open orders before exiting
            selector_api.logout()  # log out of the API cleanly
