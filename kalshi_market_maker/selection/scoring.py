from typing import Dict, List, Tuple  # type hints so Python knows what kind of data we're working with


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
    min_volume_24h = safe_float(selector_cfg.get("min_volume_24h", 100))   # minimum 24h trading volume to qualify
    min_spread_cents = safe_float(selector_cfg.get("min_spread_cents", 1)) # minimum spread in cents to qualify
    top_n = int(selector_cfg.get("top_n", 8))                              # how many markets to return
    volume_weight = safe_float(selector_cfg.get("volume_weight", 0.5))     # how much volume matters in scoring
    spread_weight = safe_float(selector_cfg.get("spread_weight", 0.5))     # how much spread matters in scoring

    def collect_candidates(ignore_thresholds: bool) -> List[Dict]:
        collected = []                                    # will hold all markets that pass the filters
        for market in markets:                            # go through every market Kalshi returned
            if not is_supported_binary_market(market):   # skip non-binary or MVE markets
                continue

            ticker = market.get("ticker")   # get the market's unique identifier
            if not ticker:                  # skip markets with no ticker (shouldn't happen but safety check)
                continue

            volume_24h = safe_float(market.get("volume_24h_fp", market.get("volume_24h", market.get("volume", 0))))  # get 24h volume (tries multiple field names)
            spread_cents = compute_spread_cents(market)  # calculate the bid-ask spread in cents
            if spread_cents < 0:                         # spread couldn't be calculated (missing prices)
                continue                                  # skip this market

            if not ignore_thresholds and (volume_24h < min_volume_24h or spread_cents < min_spread_cents):  # doesn't meet minimum requirements
                continue  # skip if below volume or spread thresholds (unless we're in fallback mode)

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
        volume_norm = normalize(market["volume_24h"], min_volume, max_volume)          # 1.0 = most liquid, 0.0 = least liquid
        spread_norm = 1.0 - normalize(market["spread_cents"], min_spread, max_spread)  # 1.0 = tightest spread, 0.0 = widest spread (inverted — tighter is better)
        score = volume_weight * volume_norm + spread_weight * spread_norm              # weighted blend of both scores
        ranked.append((market["ticker"], score, market["volume_24h"], market["spread_cents"]))  # store ticker + score + raw stats

    ranked.sort(key=lambda row: row[1], reverse=True)  # sort by score, highest first
    return ranked[:top_n]                               # return only the top N markets
