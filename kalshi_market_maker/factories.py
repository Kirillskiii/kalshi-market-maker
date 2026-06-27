import os
from typing import Dict

from .core.avellaneda import AvellanedaMarketMaker
from .core.kalshi_api import KalshiTradingAPI


def create_api(api_config: Dict, logger, market_ticker: str | None = None) -> KalshiTradingAPI:
    ticker = market_ticker if market_ticker is not None else api_config.get("market_ticker", "DYNAMIC")
    base_url = os.getenv("KALSHI_BASE_URL")
    if not base_url:
        raise ValueError("KALSHI_BASE_URL environment variable is required")

    return KalshiTradingAPI(
        api_key_id=os.getenv("KALSHI_API_KEY_ID"),
        private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        market_ticker=ticker,
        base_url=base_url,
        logger=logger,
    )


def create_market_maker(
    mm_config: Dict,
    api,
    logger,
    risk_config: Dict | None = None,
    shared_risk_state: Dict | None = None,
    close_time: float | None = None,
) -> AvellanedaMarketMaker:
    risk_config = risk_config or {}

    return AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=mm_config.get("gamma", 0.1),
        k=mm_config.get("k", 1.5),
        sigma=mm_config.get("sigma", 0.5),
        T=mm_config.get("T", 3600),
        max_position=mm_config.get("max_position", 100),
        order_expiration=mm_config.get("order_expiration", 300),
        min_spread=mm_config.get("min_spread", 0.01),
        position_limit_buffer=mm_config.get("position_limit_buffer", 0.1),
        inventory_skew_factor=mm_config.get("inventory_skew_factor", 0.01),
        trade_side=mm_config.get("trade_side", "yes"),
        max_global_contracts=risk_config.get("max_global_contracts"),
        max_contracts_per_market=risk_config.get("max_contracts_per_market"),
        reserve_contracts_buffer=risk_config.get("reserve_contracts_buffer", 0),
        shared_risk_state=shared_risk_state,
        target_order_dollars=mm_config.get("target_order_dollars"),
        max_exposure_dollars=mm_config.get("max_exposure_dollars"),
        stop_loss_dollars=mm_config.get("stop_loss_dollars"),
        consecutive_fill_threshold=int(mm_config.get("consecutive_fill_threshold", 3)),
        drift_window=int(mm_config.get("drift_window", 15)),
        adverse_selection_window=int(mm_config.get("adverse_selection_window", 20)),
        price_spike_threshold=float(mm_config.get("price_spike_threshold", 0.05)),
        price_spike_pause_ticks=int(mm_config.get("price_spike_pause_ticks", 10)),
        close_time=close_time,
        global_drawdown_limit=risk_config.get("global_drawdown_limit"),
        kalshi_fee_rate=float(mm_config.get("kalshi_fee_rate", 0.07)),
    )
