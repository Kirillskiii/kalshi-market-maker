import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional


_TICKER_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})-")
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def ticker_event_date(ticker: str) -> Optional[datetime]:
    """Parse the embedded event date from a Kalshi ticker like KXFOO-26JUN25-BAR.
    Returns a UTC midnight datetime if the date can be parsed, else None.
    Format: YY=year-suffix (26→2026), MON=month, DD=day.
    """
    m = _TICKER_DATE_RE.search(ticker.upper())
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    month = _MONTH_MAP.get(mon)
    if month is None:
        return None
    try:
        year = 2000 + int(yy)
        day = int(dd)
        return datetime(year, month, day, tzinfo=timezone.utc)
    except ValueError:
        return None


def safe_float(value, default=0.0):
    try:
        return float(value)          # try to convert value to a decimal number
    except (TypeError, ValueError):
        return default               # if it fails (None, empty string, etc.) return the default instead


def compute_spread_cents(market: Dict) -> float:
    # Live API returns yes_bid_dollars / yes_ask_dollars (0.0–1.0 dollar range).
    # Tests supply yes_bid / yes_ask (integer cents, 0–100).
    bid_dollars = market.get("yes_bid_dollars")   # try to get the bid price in dollars (live API format)
    ask_dollars = market.get("yes_ask_dollars")   # try to get the ask price in dollars (live API format)
    if bid_dollars is not None and ask_dollars is not None:                          # live API format found
        return (safe_float(ask_dollars, -1) - safe_float(bid_dollars, -1)) * 100    # convert dollar spread to cents (e.g. 0.03 → 3 cents)
    yes_bid = safe_float(market.get("yes_bid"), -1)   # fallback: try integer-cent format used in tests
    yes_ask = safe_float(market.get("yes_ask"), -1)   # same for ask
    if yes_bid < 0 or yes_ask < 0:                    # if either price is missing
        return -1                                      # return -1 to signal "unusable market"
    return yes_ask - yes_bid                           # spread in cents (e.g. ask=52, bid=50 → 2 cents)


def compute_mid_price(market: Dict) -> float:
    bid_dollars = market.get("yes_bid_dollars")
    ask_dollars = market.get("yes_ask_dollars")
    if bid_dollars is not None and ask_dollars is not None:
        return (safe_float(bid_dollars, -1) + safe_float(ask_dollars, -1)) / 2
    yes_bid = safe_float(market.get("yes_bid"), -1)
    yes_ask = safe_float(market.get("yes_ask"), -1)
    if yes_bid < 0 or yes_ask < 0:
        return -1
    return (yes_bid + yes_ask) / 200  # convert cents to dollars


def is_supported_binary_market(market: Dict) -> bool:
    market_type = str(market.get("market_type", "binary")).lower()  # get the market type, default to "binary"
    ticker = str(market.get("ticker", ""))                           # get the market ticker symbol

    if market_type != "binary":   # only trade simple yes/no binary markets
        return False               # reject scalar markets, range markets, etc.

    # Reject MVE (multivariate event) markets — these are complex combo bets we can't safely price
    if ticker.upper().startswith("KXMVE"):   # all MVE tickers start with "KXMVE"
        return False                          # reject by ticker prefix

    if market.get("mve_collection_ticker"):  # this field is set when a market belongs to a combo collection
        return False                          # reject it

    mve_legs = market.get("mve_selected_legs")           # this field lists the legs of a combo bet
    if mve_legs is not None and len(mve_legs) > 0:       # if it has any legs, it's a combo
        return False                                       # reject it

    strike_type = str(market.get("strike_type", "")).lower()  # get the strike type
    if strike_type == "functional":   # "functional" means it's an MVE combo that slipped through the other checks
        return False                   # reject it

    return True  # passed all checks — this is a normal binary market we can trade


def select_top_markets(markets: List[Dict], selector_cfg: Dict) -> List[Tuple[str, float, float, float]]:
    min_volume_24h = safe_float(selector_cfg.get("min_volume_24h", 100))
    min_spread_cents = safe_float(selector_cfg.get("min_spread_cents", 1))
    max_spread_cents = safe_float(selector_cfg.get("max_spread_cents", 20))  # avoid illiquid markets where nobody trades
    min_hours_to_close = safe_float(selector_cfg.get("min_hours_to_close", 4))
    min_mid_price = safe_float(selector_cfg.get("min_mid_price", 0.0))
    max_mid_price = safe_float(selector_cfg.get("max_mid_price", 1.0))
    top_n = int(selector_cfg.get("top_n", 8))
    volume_weight = safe_float(selector_cfg.get("volume_weight", 0.5))
    spread_weight = safe_float(selector_cfg.get("spread_weight", 0.5))
    allowed_series = [s.upper() for s in selector_cfg.get("allowed_series", [])]

    now = datetime.now(timezone.utc)

    def collect_candidates(ignore_thresholds: bool) -> List[Dict]:
        collected = []                                    # will hold all markets that pass the filters
        for market in markets:                            # go through every market Kalshi returned
            if not is_supported_binary_market(market):   # skip non-binary or MVE markets
                continue

            ticker = market.get("ticker")   # get the market's unique identifier
            if not ticker:                  # skip markets with no ticker (shouldn't happen but safety check)
                continue

            if allowed_series:
                series = str(market.get("series_ticker", "")).upper()
                if not any(series == a or series.startswith(a) or ticker.upper().startswith(a) for a in allowed_series):
                    continue

            # Reject markets whose ticker encodes today's date or an already-past date.
            # Kalshi sets close_time to after verification (days later), so close_time
            # alone doesn't catch same-day event markets like KXTRUMPMENTION-26JUN25-X.
            event_date = ticker_event_date(ticker)
            if event_date is not None:
                today_utc = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if event_date <= today_utc:
                    continue  # event date is today or past — skip, informed traders dominate

            # Use the soonest of close_time / expiration_time — sports markets often set
            # close_time to end-of-tournament but expiration_time to game day
            earliest_hours = None
            for field in ("close_time", "expiration_time"):
                ts = market.get(field)
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        hours = (dt - now).total_seconds() / 3600
                        if earliest_hours is None or hours < earliest_hours:
                            earliest_hours = hours
                    except (ValueError, TypeError):
                        pass
            if earliest_hours is not None and earliest_hours < min_hours_to_close:
                continue  # skip markets expiring too soon — catches same-day sports games

            mid_price = compute_mid_price(market)
            if mid_price < 0:
                continue  # skip if mid price can't be computed
            if not ignore_thresholds and not (min_mid_price <= mid_price <= max_mid_price):
                continue  # skip markets outside the target price range — too cheap = volatile long-shots, too expensive = near-certain

            volume_24h = safe_float(market.get("volume_24h_fp", market.get("volume_24h", market.get("volume", 0))))  # get 24h volume (tries multiple field names)
            spread_cents = compute_spread_cents(market)  # calculate the bid-ask spread in cents
            if spread_cents < 0:                         # spread couldn't be calculated (missing prices)
                continue                                  # skip this market

            if not ignore_thresholds and (volume_24h < min_volume_24h or spread_cents < min_spread_cents or spread_cents > max_spread_cents):
                continue

            collected.append(
                {
                    "ticker": ticker,           # save the ticker
                    "volume_24h": volume_24h,   # save the volume
                    "spread_cents": spread_cents, # save the spread
                }
            )
        return collected  # return all markets that passed

    candidates = collect_candidates(ignore_thresholds=False)  # first try with strict filters
    if not candidates:                                         # if nothing passes the strict filters
        candidates = collect_candidates(ignore_thresholds=True)  # relax all thresholds and try again (fallback)
    if not candidates:    # still nothing — completely empty market
        return []          # give up and return empty list

    volumes = [market["volume_24h"] for market in candidates]    # list of all volumes
    spreads = [market["spread_cents"] for market in candidates]  # list of all spreads
    min_volume, max_volume = min(volumes), max(volumes)          # find the range of volumes
    min_spread, max_spread = min(spreads), max(spreads)          # find the range of spreads

    def normalize(value: float, low: float, high: float) -> float:
        if high == low:               # all values are the same — can't normalize
            return 1.0                # give everyone a perfect score
        return (value - low) / (high - low)  # scale value to 0.0–1.0 range

    ranked = []
    for market in candidates:
        volume_norm = normalize(market["volume_24h"], min_volume, max_volume)  # 1.0 = most liquid
        # Wider spread = more profit per fill for a market maker, so score higher.
        spread_norm = normalize(market["spread_cents"], min_spread, max_spread)  # 1.0 = widest spread
        score = volume_weight * volume_norm + spread_weight * spread_norm
        ranked.append((market["ticker"], score, market["volume_24h"], market["spread_cents"]))  # store ticker + score + raw stats

    ranked.sort(key=lambda row: row[1], reverse=True)  # sort by score, highest first
    return ranked[:top_n]                               # return only the top N markets
