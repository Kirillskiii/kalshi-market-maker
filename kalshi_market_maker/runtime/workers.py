from datetime import datetime, timezone
from typing import Dict
import threading

from ..factories import create_api, create_market_maker
from ..logging_utils import build_logger
from ..selection.scoring import ticker_event_date


def _is_unsupported_market(ticker: str, market_data: Dict) -> bool:
    """Reject MVE (multivariate event) and other unsupported market types.

    MVE markets are combo/parlay-style contracts that may not be sellable
    or behave differently from standard binary markets.
    """
    # Hard ticker-prefix gate: all MVE tickers start with KXMVE
    if ticker.upper().startswith("KXMVE"):
        return True

    # API response fields
    if market_data.get("mve_collection_ticker"):
        return True
    mve_legs = market_data.get("mve_selected_legs")
    if mve_legs is not None and len(mve_legs) > 0:
        return True

    # MVE combos use functional strikes
    strike_type = str(market_data.get("strike_type", "")).lower()
    if strike_type == "functional":
        return True

    # Non-binary markets
    market_type = str(market_data.get("market_type", "binary")).lower()
    if market_type != "binary":
        return True

    return False


def run_market_worker(
    ticker: str,
    dynamic_config: Dict,
    stop_event: threading.Event,
    shared_risk_state: Dict | None = None,
):
    logger = build_logger(f"Worker_{ticker}", dynamic_config.get("log_level", "INFO"))

    api = create_api(dynamic_config.get("api", {}), logger, market_ticker=ticker)

    # Safety gate: verify market is tradeable before committing
    try:
        market_response = api.get_market(ticker)
        market_data = market_response.get("market", {})
        if _is_unsupported_market(ticker, market_data):
            logger.error(
                f"BLOCKED: {ticker} is an unsupported market type "
                f"(market_type={market_data.get('market_type')}, "
                f"strike_type={market_data.get('strike_type')}, "
                f"mve_collection={market_data.get('mve_collection_ticker')}). "
                f"Refusing to trade."
            )
            api.logout()
            return
    except Exception as check_err:
        logger.error(f"Failed to verify market type for {ticker}, refusing to trade: {check_err}")
        api.logout()
        return

    # Determine the effective close time for proximity-factor spread widening.
    # Use the earliest of: API close_time, API expiration_time, and the date
    # embedded in the ticker (more reliable for Kalshi daily event markets).
    close_time: float | None = None
    for field in ("close_time", "expiration_time"):
        ts = market_data.get(field)
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                t = dt.timestamp()
                if close_time is None or t < close_time:
                    close_time = t
            except (ValueError, TypeError):
                pass
    event_dt = ticker_event_date(ticker)
    if event_dt is not None:
        event_ts = event_dt.timestamp()
        if close_time is None or event_ts < close_time:
            close_time = event_ts

    market_maker = create_market_maker(
        dynamic_config.get("market_maker", {}),
        api,
        logger,
        dynamic_config.get("risk", {}),
        shared_risk_state,
        close_time=close_time,
    )
    dt = dynamic_config.get("dt", 2.0)

    try:
        logger.info(f"Starting market maker worker for {ticker}")
        market_maker.run(dt, stop_event=stop_event)
    finally:
        api.logout()  # always clean up, whether normal exit or crash
