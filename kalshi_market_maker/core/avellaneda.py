import math  # gives us math.log for the spread formula
import time  # used to track how long the bot has been running
from typing import Dict, List, Tuple, Optional  # type hints so python knows what kind of data each variable holds

from .interfaces import AbstractTradingAPI  # the blueprint that any trading API must follow


class AvellanedaMarketMaker:
    def __init__(
        self,
        logger,                                      # prints status messages to the terminal
        api: AbstractTradingAPI,                     # the object that actually talks to Kalshi
        gamma: float,                                # risk aversion — higher = wider spreads, less risk
        k: float,                                    # how fast the order book fills up (market depth)
        sigma: float,                                # how volatile the price is expected to be
        T: float,                                    # total time window in seconds the bot trades for
        max_position: int,                           # max contracts allowed to hold at once
        order_expiration: int,                       # how many seconds an order stays open before auto-cancelling
        min_spread: float = 0.01,                    # smallest allowed gap between bid and ask (1 cent)
        position_limit_buffer: float = 0.1,          # fraction of max position used as base order size
        inventory_skew_factor: float = 0.01,         # how aggressively to skew prices when holding inventory
        trade_side: str = "yes",                     # whether to trade "yes" or "no" contracts
        max_global_contracts: Optional[int] = None,  # hard cap on total contracts across ALL markets
        max_contracts_per_market: Optional[int] = None,  # hard cap per individual market
        reserve_contracts_buffer: int = 0,           # contracts to hold back as a safety buffer
        shared_risk_state: Optional[Dict] = None,    # shared dict telling us how many markets are active
        target_order_dollars: Optional[float] = None,  # target dollar value per order; overrides position_limit_buffer sizing
    ):
        self.api = api                                                      # save the API object for later use
        self.logger = logger                                                # save the logger
        self.base_gamma = gamma                                             # save base risk aversion
        self.k = k                                                          # save market depth constant
        self.sigma = sigma                                                  # save volatility estimate
        self.T = T                                                          # save total trading window
        self.max_position = max_position                                    # save position limit
        self.order_expiration = order_expiration                            # save order lifetime
        self.min_spread = min_spread                                        # save minimum spread floor
        self.position_limit_buffer = position_limit_buffer                  # save buffer fraction
        self.inventory_skew_factor = inventory_skew_factor                  # save skew aggressiveness
        self.trade_side = trade_side                                        # save which side (yes/no)
        self.max_global_contracts = max_global_contracts                    # save global cap
        self.max_contracts_per_market = max_contracts_per_market            # save per-market cap
        self.reserve_contracts_buffer = max(0, int(reserve_contracts_buffer))  # ensure buffer is non-negative
        self.shared_risk_state = shared_risk_state or {"active_markets": 1}    # default to 1 active market if none given
        self.target_order_dollars = target_order_dollars                    # save dollar target per order

    def run(self, dt: float, stop_event=None):
        start_time = time.time()                                     # record when we started
        while time.time() - start_time < self.T:                    # keep looping until the time window T is up
            if stop_event is not None and stop_event.is_set():      # check if someone told us to stop
                self.logger.info("Stop signal received, shutting down market maker loop")
                break                                                # exit the loop immediately

            current_time = time.time() - start_time                 # how many seconds since we started
            mid_prices = self.api.get_price()                       # fetch current bid/ask from Kalshi
            mid_price = mid_prices[self.trade_side]                 # grab just the "yes" (or "no") mid price
            inventory = self.api.get_position()                     # how many contracts we currently hold

            reservation_price = self.calculate_reservation_price(mid_price, inventory, current_time)   # our "fair value" adjusted for inventory
            bid_price, ask_price = self.calculate_asymmetric_quotes(mid_price, inventory, current_time) # the actual prices we'll post
            current_orders = self.api.get_orders()                  # fetch our currently open orders
            buy_size, sell_size = self.calculate_order_sizes(inventory, current_orders, mid_price)      # how many contracts to bid/ask

            self.logger.info(
                f"t={current_time:.2f}s mid={mid_price:.4f} inventory={inventory} "
                f"reservation={reservation_price:.4f} bid={bid_price:.4f} ask={ask_price:.4f}"
            )  # print a status line every loop so we can see what the bot is doing

            self.manage_orders(bid_price, ask_price, buy_size, sell_size, current_orders)  # cancel stale orders and place fresh ones
            time.sleep(dt)                                           # wait dt seconds before the next loop

        self.logger.info("Avellaneda market maker finished running")  # log that this market is done

    def calculate_asymmetric_quotes(self, mid_price: float, inventory: int, elapsed_time: float) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, inventory, elapsed_time)  # get our skewed fair value
        base_spread = self.calculate_optimal_spread(elapsed_time, inventory)                       # get the total spread width

        effective_max_position = self.get_effective_max_position()           # max contracts we're allowed
        position_ratio = inventory / effective_max_position                  # how full our position is (0=empty, 1=maxed out)
        spread_adjustment = base_spread * abs(position_ratio) * 3           # widen the spread more the more inventory we hold

        if inventory > 0:                                                    # we're holding too many — need to sell
            bid_spread = base_spread / 2 + spread_adjustment                # push bid lower to discourage more buying
            ask_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)  # pull ask closer to attract sellers
        else:                                                                # we're short or flat — need to buy
            bid_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)  # pull bid closer to attract buyers
            ask_spread = base_spread / 2 + spread_adjustment                # push ask higher to discourage more selling

        bid_price = max(0.01, min(mid_price, reservation_price - bid_spread))   # final bid: never below 1 cent, never above mid
        ask_price = min(0.99, max(mid_price, reservation_price + ask_spread))   # final ask: never above 99 cents, never below mid

        return bid_price, ask_price  # return both prices

    def calculate_reservation_price(self, mid_price: float, inventory: int, elapsed_time: float) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)                              # get risk aversion scaled by inventory
        inventory_skew = -inventory * self.inventory_skew_factor * mid_price                # negative skew when long, positive when short
        time_remaining = max(0.0, 1 - elapsed_time / self.T)                                # fraction of time left (1.0 = just started, 0.0 = done)
        return mid_price + inventory_skew - inventory * dynamic_gamma * (self.sigma ** 2) * time_remaining  # reservation = mid adjusted for inventory risk

    def calculate_optimal_spread(self, elapsed_time: float, inventory: int) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)       # risk aversion grows with inventory
        time_remaining = max(0.0, 1 - elapsed_time / self.T)         # fraction of time left
        base_spread = (
            dynamic_gamma * (self.sigma ** 2) * time_remaining        # first term: risk cost (wider when more time remains)
            + (2 / dynamic_gamma) * math.log(1 + (dynamic_gamma / self.k))  # second term: profit from market depth
        )
        effective_max_position = self.get_effective_max_position()    # max contracts allowed
        position_ratio = min(1.0, abs(inventory) / effective_max_position)  # how loaded we are (capped at 1.0)
        spread_multiplier = 1 + 2.0 * position_ratio                 # spread grows up to 3x when fully loaded
        return max(base_spread * spread_multiplier, self.min_spread)  # never go below the minimum spread floor

    def calculate_dynamic_gamma(self, inventory: int) -> float:
        effective_max_position = self.get_effective_max_position()    # max contracts allowed
        position_ratio = abs(inventory) / effective_max_position      # how full we are (0.0 to 1.0)
        return self.base_gamma * (1 + (position_ratio**2) * 4)        # gamma grows up to 5x when maxed — makes us much more cautious

    def get_effective_max_position(self) -> int:
        if self.max_contracts_per_market is not None:                 # if a per-market cap is set
            configured_market_cap = max(1, int(self.max_contracts_per_market))  # use it (minimum 1)
        else:
            configured_market_cap = max(1, int(self.max_position))   # otherwise fall back to general max_position

        if self.max_global_contracts is None:                         # if no global cap is set
            return configured_market_cap                              # just return the per-market cap

        active_markets = max(1, int(self.shared_risk_state.get("active_markets", 1)))       # how many markets are running right now
        global_budget = max(1, int(self.max_global_contracts) - self.reserve_contracts_buffer)  # total allowed minus the safety buffer
        equal_weight_cap = max(1, global_budget // active_markets)   # divide the budget equally across all markets
        return max(1, min(configured_market_cap, equal_weight_cap))  # take the tighter of the two limits

    def get_global_remaining_capacity(self) -> int:
        if self.max_global_contracts is None:                         # no global cap set
            return 10**9                                              # return a huge number meaning "unlimited"

        try:
            positions = self.api.list_all_positions()                 # fetch all positions across every market
            total_abs_position = sum(abs(int(float(position.get("position", position.get("position_fp", 0))))) for position in positions)  # sum up total contracts held (absolute value, counts both long and short)
            remaining = int(self.max_global_contracts) - self.reserve_contracts_buffer - total_abs_position  # how many more we're allowed to take on
            return max(0, remaining)                                  # never return a negative number
        except Exception as global_exception:
            self.logger.error(f"Global risk snapshot failed, blocking new risk: {global_exception}")
            return 0                                                  # if anything goes wrong, block all new trades to be safe

    def extract_pending_exposure(self, current_orders: List[Dict]) -> Tuple[int, int]:
        pending_buy = 0   # total contracts in open buy orders not yet filled
        pending_sell = 0  # total contracts in open sell orders not yet filled

        for order in current_orders:                                         # look at each open order
            if order.get("side") != self.trade_side:                         # skip orders on the wrong side (e.g. "no" when we trade "yes")
                continue
            remaining = int(float(order.get("remaining_count", order.get("remaining_count_fp", 0))))  # how many contracts still unfilled
            if order.get("action") == "buy":                                 # it's a buy order
                pending_buy += remaining                                     # add to pending buy total
            elif order.get("action") == "sell":                              # it's a sell order
                pending_sell += remaining                                     # add to pending sell total

        return pending_buy, pending_sell  # return both totals

    def calculate_order_sizes(self, inventory: int, current_orders: List[Dict], mid_price: float = 0.5) -> Tuple[int, int]:
        effective_max_position = self.get_effective_max_position()           # max we're allowed to hold
        pending_buy, pending_sell = self.extract_pending_exposure(current_orders)  # contracts already committed in open orders
        effective_inventory = inventory + pending_buy - pending_sell         # inventory including what's in-flight

        local_remaining_capacity = max(0, effective_max_position - abs(effective_inventory))  # how many more we can take on in this market
        global_remaining_capacity = self.get_global_remaining_capacity()    # how many more we can take on across all markets

        if self.target_order_dollars is not None and mid_price > 0:
            base_size = max(1, round(self.target_order_dollars / mid_price))  # contracts needed to spend target_order_dollars at current price
        else:
            base_size = max(1, int(effective_max_position * self.position_limit_buffer))  # smallest order size (at least 1)
        accumulation_size = min(base_size, local_remaining_capacity, global_remaining_capacity)  # how much to add — limited by both caps
        if global_remaining_capacity <= 0:                                   # if we've hit the global cap
            accumulation_size = 0                                            # stop adding new positions entirely

        reduction_size = max(1, min(effective_max_position, max(base_size, abs(effective_inventory))))  # how aggressively to reduce — bigger when more loaded

        if effective_inventory > 0:       # we're long (holding more yes contracts than we want)
            buy_size = accumulation_size  # buy cautiously
            sell_size = reduction_size    # sell aggressively to reduce
        elif effective_inventory < 0:     # we're short (holding more no / sold more than we bought)
            buy_size = reduction_size     # buy aggressively to reduce
            sell_size = accumulation_size # sell cautiously
        else:                             # flat — no inventory
            buy_size = accumulation_size  # buy and sell equally
            sell_size = accumulation_size

        return buy_size, sell_size  # return both sizes

    def manage_orders(
        self,
        bid_price: float,    # the price we want to buy at
        ask_price: float,    # the price we want to sell at
        buy_size: int,       # how many contracts to bid for
        sell_size: int,      # how many contracts to offer
        current_orders: Optional[List[Dict]] = None,  # existing open orders (fetched if not provided)
    ):
        if current_orders is None:              # if no orders were passed in
            current_orders = self.api.get_orders()  # fetch them from Kalshi now

        buy_orders: List[Dict] = []   # will hold all our current buy orders
        sell_orders: List[Dict] = []  # will hold all our current sell orders

        for order in current_orders:                          # go through every open order
            if order["side"] == self.trade_side:              # only look at orders on our side (yes/no)
                if order["action"] == "buy":                  # it's a buy order
                    buy_orders.append(order)                  # add to buy list
                elif order["action"] == "sell":               # it's a sell order
                    sell_orders.append(order)                 # add to sell list

        self.handle_order_side("buy", buy_orders, bid_price, buy_size)    # manage the buy side
        self.handle_order_side("sell", sell_orders, ask_price, sell_size) # manage the sell side

    def handle_order_side(self, action: str, orders: List[Dict], desired_price: float, desired_size: int):
        keep_order = None  # tracks which existing order (if any) is still good and shouldn't be cancelled

        for order in orders:                                              # go through each existing order on this side
            if self.trade_side == "yes":                                  # trading yes contracts
                raw = order.get("yes_price_dollars", order.get("yes_price"))  # get price (dollars or cents format)
                current_price = float(raw) if "." in str(raw) else float(raw) / 100  # convert to dollars if needed
            else:                                                         # trading no contracts
                raw = order.get("no_price_dollars", order.get("no_price"))
                current_price = float(raw) if "." in str(raw) else float(raw) / 100
            remaining = int(float(order.get("remaining_count", order.get("remaining_count_fp", 0))))  # contracts still unfilled
            if (
                keep_order is None                                        # haven't found a good order yet
                and abs(current_price - desired_price) < 0.01            # price is close enough (within 1 cent)
                and remaining == desired_size                             # size matches exactly
            ):
                keep_order = order                                        # this order is still good — keep it
            else:
                self.api.cancel_order(order["order_id"])                 # stale or wrong price/size — cancel it

        if desired_size <= 0:   # if we don't want any position on this side
            return               # nothing to place, we're done

        current_price = self.api.get_price()[self.trade_side]            # fetch the latest market price
        should_place = (action == "buy" and desired_price < current_price) or (
            action == "sell" and desired_price > current_price
        )  # only place a buy below market price, or a sell above market price (limit orders only)

        if keep_order is None and should_place:          # no existing good order and conditions are right
            self.api.place_order(
                action,                                  # "buy" or "sell"
                self.trade_side,                         # "yes" or "no"
                desired_price,                           # the price to post at
                desired_size,                            # number of contracts
                int(time.time()) + self.order_expiration,  # timestamp when the order expires
            )
