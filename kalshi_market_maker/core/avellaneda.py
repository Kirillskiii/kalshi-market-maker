import math
import time
from typing import Dict, List, Tuple, Optional

from .interfaces import AbstractTradingAPI


class AvellanedaMarketMaker:
    def __init__(
        self,
        logger,
        api: AbstractTradingAPI,
        gamma: float,
        k: float,
        sigma: float,
        T: float,
        max_position: int,
        order_expiration: int,
        min_spread: float = 0.01,
        position_limit_buffer: float = 0.1,
        inventory_skew_factor: float = 0.01,
        trade_side: str = "yes",
        max_global_contracts: Optional[int] = None,
        max_contracts_per_market: Optional[int] = None,
        reserve_contracts_buffer: int = 0,
        shared_risk_state: Optional[Dict] = None,
        target_order_dollars: Optional[float] = None,
        max_exposure_dollars: Optional[float] = None,
        stop_loss_dollars: Optional[float] = None,
        consecutive_fill_threshold: int = 3,
        drift_window: int = 15,
        # Glosten-Milgrom-style informed-trader protections
        adverse_selection_window: int = 20,   # rolling fill window for adverse selection rate
        price_spike_threshold: float = 0.05,  # mid jump ≥ this triggers a quoting pause
        price_spike_pause_ticks: int = 10,    # ticks to cancel + pause after a spike
        close_time: Optional[float] = None,   # unix timestamp of market resolution
        # Portfolio-level risk
        global_drawdown_limit: Optional[float] = None,  # stop all trading if portfolio P&L drops below -this
        kalshi_fee_rate: float = 0.07,        # Kalshi fee = rate × P × (1−P) per contract per side
    ):
        self.api = api
        self.logger = logger
        self.base_gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.max_position = max_position
        self.order_expiration = order_expiration
        self.min_spread = min_spread
        self.position_limit_buffer = position_limit_buffer
        self.inventory_skew_factor = inventory_skew_factor
        self.trade_side = trade_side
        self.max_global_contracts = max_global_contracts
        self.max_contracts_per_market = max_contracts_per_market
        self.reserve_contracts_buffer = max(0, int(reserve_contracts_buffer))
        self.shared_risk_state = shared_risk_state or {"active_markets": 1}
        self.target_order_dollars = target_order_dollars
        self.max_exposure_dollars = max_exposure_dollars
        self.stop_loss_dollars = stop_loss_dollars
        self.consecutive_fill_threshold = consecutive_fill_threshold
        self.drift_window = max(5, int(drift_window))
        self.adverse_selection_window = max(5, int(adverse_selection_window))
        self._price_spike_threshold = price_spike_threshold
        self.price_spike_pause_ticks = price_spike_pause_ticks
        self._close_time = close_time
        self.global_drawdown_limit = global_drawdown_limit
        self.kalshi_fee_rate = kalshi_fee_rate

        self._prev_inventory: Optional[int] = None
        self._avg_cost: float = 0.0
        self._realized_pnl: float = 0.0

        self._global_cap_cache: Optional[int] = None
        self._global_cap_cache_time: float = 0.0
        self._global_cap_ttl: float = 30.0

        # Rolling price history for realized vol estimation
        self._price_history: List[float] = []

        # One-sided fill detection: tracks contracts filled per side since last opposite fill
        self._fill_streak: Dict[str, int] = {"buy": 0, "sell": 0}
        self._paused_side: Optional[str] = None
        self._paused_ticks: int = 0

        # OFI: rolling fill direction (+1 = seller hit our bid, -1 = buyer hit our ask)
        self._recent_fills: List[int] = []
        self._ofi_window: int = 10

        # Adverse selection: was each observed fill followed by an adverse price move?
        self._fill_adversity: List[int] = []   # 1=adverse, 0=non-adverse, rolling window
        self._last_fill_direction: Optional[int] = None  # stored for next-tick check
        self._last_mid_price: Optional[float] = None

        # Price spike pause (proxy for news event / volume spike)
        self._spike_paused_ticks: int = 0

        # Last observed bid/ask — used as fill price proxy when a fill is detected
        self._last_bid: Optional[float] = None
        self._last_ask: Optional[float] = None

    # ── Price history & realized vol ──────────────────────────────────────────

    def _update_price_history(self, mid_price: float) -> None:
        self._price_history.append(mid_price)
        if len(self._price_history) > self.drift_window:
            self._price_history.pop(0)

    def _realized_sigma(self) -> float:
        """Realized vol from recent price history, blended with configured sigma floor."""
        if len(self._price_history) < 5:
            return self.sigma
        moves = [abs(self._price_history[i] - self._price_history[i - 1])
                 for i in range(1, len(self._price_history))]
        realized = sum(moves) / len(moves)
        return min(0.50, max(self.sigma, realized * 5))

    # ── Glosten-Milgrom informed-trader protections ───────────────────────────

    def _adverse_selection_rate(self) -> float:
        """Fraction of recent fills where price subsequently moved against us (0–1)."""
        if not self._fill_adversity:
            return 0.0
        return sum(self._fill_adversity) / len(self._fill_adversity)

    def _effective_sigma(self) -> float:
        """
        Realized vol amplified by adverse selection rate.
        At 0% adverse: base sigma. At 100% adverse: 4× base sigma.
        Informed counterparties are effectively raising our cost of market making.
        """
        base = self._realized_sigma()
        asr = self._adverse_selection_rate()
        return min(0.50, base * (1.0 + 3.0 * asr))

    def _ofi_skew(self) -> float:
        """
        Directional bias from order flow imbalance.
        Seller-initiated fills (we bought) → bearish → negative skew.
        Buyer-initiated fills (we sold)    → bullish → positive skew.
        Falls back to price momentum when no fills are observed.
        """
        if self._recent_fills:
            # Negate because fill=+1 (we bought) means a seller hit our bid → bearish
            ofi = -sum(self._recent_fills) / len(self._recent_fills)  # −1 to +1
            return ofi * 0.005  # ±0.5¢ max per unit of OFI
        # Fallback: short-term price momentum
        n = len(self._price_history)
        if n < 5:
            return 0.0
        third = max(2, n // 3)
        old_avg = sum(self._price_history[:third]) / third
        new_avg = sum(self._price_history[-third:]) / third
        return (new_avg - old_avg) / n

    def _proximity_factor(self) -> float:
        """
        Spread multiplier that grows as market resolution approaches.
        Informed trader dominance increases sharply near the event date.
        1.0× at ≥48 h, ~2.5× at 2 h, 3.0× past close.
        """
        if self._close_time is None:
            return 1.0
        hours_left = (self._close_time - time.time()) / 3600
        if hours_left >= 48:
            return 1.0
        if hours_left <= 0:
            return 3.0
        if hours_left >= 2:
            return 1.0 + 1.5 * (1.0 - hours_left / 48.0)
        return 3.0

    def _fee_floor_spread(self, mid_price: float) -> float:
        """
        Minimum spread for a round trip to break even on Kalshi fees.
        Fee = kalshi_fee_rate × P × (1−P) per contract per side.
        Adds a 0.5¢ profit margin on top.
        """
        fee_per_side = self.kalshi_fee_rate * mid_price * (1.0 - mid_price)
        return 2.0 * fee_per_side + 0.005

    def _update_adversarial_tracking(self, new_mid: float) -> None:
        """
        Called every tick (before _update_pnl).
        1. Checks if the previous fill was adversely selected (price moved against us).
        2. Detects large mid-price jumps that signal a news event → sets spike pause.
        3. Advances _last_mid_price.
        """
        if self._last_mid_price is not None:
            price_move = new_mid - self._last_mid_price

            # Adverse selection: did the previous fill direction predict the wrong move?
            if self._last_fill_direction is not None and abs(price_move) > 0.005:
                # adverse when: we bought (+1) and price fell (<0), or sold (-1) and price rose (>0)
                adverse = int(self._last_fill_direction * price_move < 0)
                self._fill_adversity.append(adverse)
                if len(self._fill_adversity) > self.adverse_selection_window:
                    self._fill_adversity.pop(0)

            # Price spike → news event → pause quoting
            if abs(price_move) >= self._price_spike_threshold and self._spike_paused_ticks == 0:
                self.logger.warning(
                    f"Price spike: {self._last_mid_price:.3f} → {new_mid:.3f} "
                    f"({price_move:+.3f}). Pausing {self.price_spike_pause_ticks} ticks."
                )
                self._spike_paused_ticks = self.price_spike_pause_ticks

        self._last_mid_price = new_mid

    # ── P&L tracking ─────────────────────────────────────────────────────────

    def _update_portfolio_pnl(self, delta: float) -> None:
        """Thread-safe update of the shared portfolio P&L counter."""
        lock = self.shared_risk_state.get("_lock")
        if lock is not None:
            with lock:
                self.shared_risk_state["total_realized_pnl"] = (
                    self.shared_risk_state.get("total_realized_pnl", 0.0) + delta
                )
        else:
            self.shared_risk_state["total_realized_pnl"] = (
                self.shared_risk_state.get("total_realized_pnl", 0.0) + delta
            )

    def _update_pnl(
        self,
        inventory: int,
        mid_price: float,
        fill_bid: Optional[float] = None,
        fill_ask: Optional[float] = None,
    ) -> None:
        if self._prev_inventory is None:
            self._prev_inventory = inventory
            return

        delta = inventory - self._prev_inventory
        self._prev_inventory = inventory

        if delta > 0:
            # We bought — fill price was our posted bid (approximated by last observed bid)
            buy_price = fill_bid if fill_bid is not None else mid_price
            if inventory > 0:
                prev_long = max(0, inventory - delta)
                self._avg_cost = (self._avg_cost * prev_long + delta * buy_price) / inventory
            else:
                contracts_covered = min(delta, abs(inventory - delta))
                pnl_delta = contracts_covered * (self._avg_cost - buy_price)
                self._realized_pnl += pnl_delta
                self._update_portfolio_pnl(pnl_delta)
                if inventory == 0:
                    self._avg_cost = 0.0
            self._fill_streak["buy"] += abs(delta)
            self._fill_streak["sell"] = 0
            if self._fill_streak["buy"] >= self.consecutive_fill_threshold and self._paused_side != "buy":
                self._paused_side = "buy"
                self._paused_ticks = 20
                self.logger.warning(
                    f"One-sided fill streak ({self._fill_streak['buy']} buys). Pausing buy side for 20 ticks."
                )
            # seller hit our bid → bearish OFI signal
            self._recent_fills.extend([+1] * abs(delta))
            self._last_fill_direction = +1

        elif delta < 0:
            # We sold — fill price was our posted ask (approximated by last observed ask)
            sell_price = fill_ask if fill_ask is not None else mid_price
            if inventory < 0:
                prev_short = min(0, inventory - delta)
                self._avg_cost = (self._avg_cost * abs(prev_short) + abs(delta) * sell_price) / abs(inventory)
            else:
                contracts_sold = abs(delta)
                pnl_delta = contracts_sold * (sell_price - self._avg_cost)
                self._realized_pnl += pnl_delta
                self._update_portfolio_pnl(pnl_delta)
                if inventory == 0:
                    self._avg_cost = 0.0
            self._fill_streak["sell"] += abs(delta)
            self._fill_streak["buy"] = 0
            if self._fill_streak["sell"] >= self.consecutive_fill_threshold and self._paused_side != "sell":
                self._paused_side = "sell"
                self._paused_ticks = 20
                self.logger.warning(
                    f"One-sided fill streak ({self._fill_streak['sell']} sells). Pausing sell side for 20 ticks."
                )
            # buyer hit our ask → bullish OFI signal
            self._recent_fills.extend([-1] * abs(delta))
            self._last_fill_direction = -1

        else:
            self._last_fill_direction = None  # no fill this tick

        if len(self._recent_fills) > self._ofi_window:
            self._recent_fills = self._recent_fills[-self._ofi_window:]

    # ── Emergency close ───────────────────────────────────────────────────────

    def _emergency_close(self, inventory: int, mid_price: float) -> None:
        try:
            for o in self.api.get_orders():
                try:
                    self.api.cancel_order(o["order_id"])
                except Exception:
                    pass
        except Exception:
            pass
        if inventory == 0:
            return
        if inventory > 0:
            close_price = max(0.01, round(mid_price - 0.05, 2))
            self.api.place_order("sell", self.trade_side, close_price, abs(inventory), int(time.time()) + 120)
        else:
            close_price = min(0.99, round(mid_price + 0.05, 2))
            self.api.place_order("buy", self.trade_side, close_price, abs(inventory), int(time.time()) + 120)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, dt: float, stop_event=None):
        start_time = time.time()
        while time.time() - start_time < self.T:
            if stop_event is not None and stop_event.is_set():
                self.logger.info("Stop signal received, shutting down market maker loop")
                break
            if self.shared_risk_state.get("global_stop", False):
                self.logger.warning("Global stop signal received. Exiting market maker loop.")
                break

            try:
                current_time = time.time() - start_time
                prices = self.api.get_price()
                mid_price = prices[self.trade_side]

                # Detect market close/settlement before touching orders
                status = prices.get("status", "open")
                if status not in ("open", "active"):
                    self.logger.warning(
                        f"Market is no longer open (status={status!r}). "
                        f"Cancelling all orders and exiting."
                    )
                    try:
                        for o in self.api.get_orders():
                            try:
                                self.api.cancel_order(o["order_id"])
                            except Exception:
                                pass
                    except Exception:
                        pass
                    break

                # Capture bid/ask BEFORE _update_pnl so the previous tick's prices are
                # used as fill prices (fills happen between ticks, at our last posted prices).
                fill_bid = self._last_bid
                fill_ask = self._last_ask
                bid_key = "yes_bid" if self.trade_side == "yes" else "no_bid"
                ask_key = "yes_ask" if self.trade_side == "yes" else "no_ask"
                self._last_bid = prices.get(bid_key)
                self._last_ask = prices.get(ask_key)

                inventory = self.api.get_position()

                self._update_price_history(mid_price)
                self._update_adversarial_tracking(mid_price)  # must precede _update_pnl
                self._update_pnl(inventory, mid_price, fill_bid=fill_bid, fill_ask=fill_ask)

                unrealized_pnl = inventory * (mid_price - self._avg_cost)
                total_pnl = self._realized_pnl + unrealized_pnl

                # Per-market stop-loss
                if self.stop_loss_dollars is not None and total_pnl <= -self.stop_loss_dollars:
                    self.logger.warning(
                        f"Stop-loss triggered: total_pnl=${total_pnl:+.4f} <= -${self.stop_loss_dollars:.2f}. "
                        f"Closing position (inventory={inventory}) and exiting."
                    )
                    self._emergency_close(inventory, mid_price)
                    break

                # Portfolio-level drawdown limit — signals all workers to stop
                if self.global_drawdown_limit is not None:
                    portfolio_pnl = self.shared_risk_state.get("total_realized_pnl", 0.0)
                    if portfolio_pnl <= -self.global_drawdown_limit:
                        self.logger.warning(
                            f"Global drawdown limit hit: portfolio_pnl=${portfolio_pnl:+.4f} "
                            f"<= -${self.global_drawdown_limit:.2f}. Signalling all workers to stop."
                        )
                        self.shared_risk_state["global_stop"] = True
                        wake = self.shared_risk_state.get("_selector_wake")
                        if wake is not None:
                            wake.set()
                        self._emergency_close(inventory, mid_price)
                        break

                # News-event spike pause: cancel everything and wait for volatility to settle
                if self._spike_paused_ticks > 0:
                    self._spike_paused_ticks -= 1
                    try:
                        for o in self.api.get_orders():
                            try:
                                self.api.cancel_order(o["order_id"])
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self.logger.warning(f"Spike pause: {self._spike_paused_ticks} ticks remaining, all orders cancelled.")
                    time.sleep(dt)
                    continue

                if self._paused_side is not None:
                    self._paused_ticks -= 1
                    if self._paused_ticks <= 0:
                        self.logger.info(f"Resuming {self._paused_side} side after pause.")
                        self._paused_side = None
                        self._fill_streak = {"buy": 0, "sell": 0}

                reservation_price = self.calculate_reservation_price(mid_price, inventory, current_time)
                bid_price, ask_price = self.calculate_asymmetric_quotes(mid_price, inventory, current_time)
                current_orders = self.api.get_orders()
                buy_size, sell_size = self.calculate_order_sizes(inventory, current_orders, mid_price)

                self.logger.info(
                    f"t={current_time:.2f}s mid={mid_price:.4f} inventory={inventory} "
                    f"reservation={reservation_price:.4f} bid={bid_price:.4f} ask={ask_price:.4f} "
                    f"ofi={self._ofi_skew():+.4f} sigma={self._effective_sigma():.3f} "
                    f"asr={self._adverse_selection_rate():.2f} proximity={self._proximity_factor():.2f} "
                    f"realized=${self._realized_pnl:+.4f} unrealized=${unrealized_pnl:+.4f} "
                    f"total=${total_pnl:+.4f}"
                )

                self.manage_orders(bid_price, ask_price, buy_size, sell_size, current_orders, mid_price)
                time.sleep(dt)

            except Exception as tick_error:
                self.logger.error(f"Tick error (will retry next cycle): {tick_error}")
                time.sleep(dt)

        self.logger.info("Avellaneda market maker finished running")

    # ── Quote calculation ─────────────────────────────────────────────────────

    def calculate_asymmetric_quotes(self, mid_price: float, inventory: int, elapsed_time: float) -> Tuple[float, float]:
        reservation_price = self.calculate_reservation_price(mid_price, inventory, elapsed_time)
        base_spread = self.calculate_optimal_spread(elapsed_time, inventory, mid_price)

        effective_max_position = self.get_effective_max_position()
        position_ratio = inventory / effective_max_position
        spread_adjustment = base_spread * abs(position_ratio) * 3

        if inventory > 0:
            bid_spread = base_spread / 2 + spread_adjustment
            ask_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
        else:
            bid_spread = max(base_spread / 2 - spread_adjustment, self.min_spread / 2)
            ask_spread = base_spread / 2 + spread_adjustment

        bid_price = max(0.01, min(mid_price, reservation_price - bid_spread))
        ask_price = min(0.99, max(mid_price, reservation_price + ask_spread))

        return bid_price, ask_price

    def calculate_reservation_price(self, mid_price: float, inventory: int, elapsed_time: float) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        effective_sigma = self._effective_sigma()
        inventory_skew = -inventory * self.inventory_skew_factor * mid_price
        time_remaining = max(0.0, 1 - elapsed_time / self.T)
        # OFI-based directional adjustment: tilt toward where informed flow is pointing,
        # amplified by inventory size (more inventory = more exposure to the signal).
        ofi_adjustment = self._ofi_skew() * (1 + abs(inventory))
        return (
            mid_price
            + ofi_adjustment
            + inventory_skew
            - inventory * dynamic_gamma * (effective_sigma ** 2) * time_remaining
        )

    def calculate_optimal_spread(self, elapsed_time: float, inventory: int, mid_price: float = 0.5) -> float:
        dynamic_gamma = self.calculate_dynamic_gamma(inventory)
        effective_sigma = self._effective_sigma()
        time_remaining = max(0.0, 1 - elapsed_time / self.T)
        base_spread = (
            dynamic_gamma * (effective_sigma ** 2) * time_remaining
            + (2 / dynamic_gamma) * math.log(1 + (dynamic_gamma / self.k))
        )
        effective_max_position = self.get_effective_max_position()
        position_ratio = min(1.0, abs(inventory) / effective_max_position)
        spread_multiplier = 1 + 2.0 * position_ratio
        # Three-way floor: model minimum, configured minimum, fee break-even
        raw_spread = max(base_spread * spread_multiplier, self.min_spread, self._fee_floor_spread(mid_price))
        # Widen further as resolution approaches — informed traders dominate near the event
        return raw_spread * self._proximity_factor()

    def calculate_dynamic_gamma(self, inventory: int) -> float:
        effective_max_position = self.get_effective_max_position()
        position_ratio = abs(inventory) / effective_max_position
        return self.base_gamma * (1 + (position_ratio**2) * 4)

    # ── Risk / sizing ─────────────────────────────────────────────────────────

    def get_effective_max_position(self) -> int:
        if self.max_contracts_per_market is not None:
            configured_market_cap = max(1, int(self.max_contracts_per_market))
        else:
            configured_market_cap = max(1, int(self.max_position))

        if self.max_global_contracts is None:
            return configured_market_cap

        active_markets = max(1, int(self.shared_risk_state.get("active_markets", 1)))
        global_budget = max(1, int(self.max_global_contracts) - self.reserve_contracts_buffer)
        equal_weight_cap = max(1, global_budget // active_markets)
        return max(1, min(configured_market_cap, equal_weight_cap))

    def get_global_remaining_capacity(self) -> int:
        if self.max_global_contracts is None:
            return 10**9

        now = time.time()
        if self._global_cap_cache is not None and (now - self._global_cap_cache_time) < self._global_cap_ttl:
            return self._global_cap_cache

        try:
            positions = self.api.list_all_positions()
            total_abs_position = sum(
                abs(int(float(p.get("position", p.get("position_fp", 0))))) for p in positions
            )
            remaining = max(0, int(self.max_global_contracts) - self.reserve_contracts_buffer - total_abs_position)
            self._global_cap_cache = remaining
            self._global_cap_cache_time = now
            return remaining
        except Exception as global_exception:
            self.logger.error(f"Global risk snapshot failed, blocking new risk: {global_exception}")
            return 0

    def extract_pending_exposure(self, current_orders: List[Dict]) -> Tuple[int, int]:
        pending_buy = 0
        pending_sell = 0
        for order in current_orders:
            if order.get("side") != self.trade_side:
                continue
            remaining = int(float(order.get("remaining_count", order.get("remaining_count_fp", 0))))
            if order.get("action") == "buy":
                pending_buy += remaining
            elif order.get("action") == "sell":
                pending_sell += remaining
        return pending_buy, pending_sell

    def calculate_order_sizes(self, inventory: int, current_orders: List[Dict], mid_price: float = 0.5) -> Tuple[int, int]:
        effective_max_position = self.get_effective_max_position()
        pending_buy, pending_sell = self.extract_pending_exposure(current_orders)
        effective_inventory = inventory + pending_buy - pending_sell

        global_remaining_capacity = self.get_global_remaining_capacity()

        if self.target_order_dollars is not None and mid_price > 0:
            base_size = max(1, round(self.target_order_dollars / mid_price))
        else:
            base_size = max(1, int(effective_max_position * self.position_limit_buffer))

        inventory_fraction = min(1.0, abs(effective_inventory) / effective_max_position)
        accumulation_scale = max(0.0, 1.0 - inventory_fraction)
        accumulation_size = max(0, min(
            round(base_size * accumulation_scale),
            global_remaining_capacity,
        ))

        if global_remaining_capacity <= 0:
            accumulation_size = 0

        if self.max_exposure_dollars is not None and mid_price > 0:
            current_exposure = (inventory + pending_buy) * mid_price
            if current_exposure >= self.max_exposure_dollars:
                accumulation_size = 0

        buy_accum = 0 if self._paused_side == "buy" else accumulation_size
        sell_accum = 0 if self._paused_side == "sell" else accumulation_size

        reduction_size = max(1, min(effective_max_position, max(base_size, abs(effective_inventory))))

        if effective_inventory > 0:
            buy_size = buy_accum
            sell_size = reduction_size
        elif effective_inventory < 0:
            buy_size = reduction_size
            sell_size = sell_accum
        else:
            buy_size = buy_accum
            sell_size = sell_accum

        return buy_size, sell_size

    # ── Order management ──────────────────────────────────────────────────────

    def manage_orders(
        self,
        bid_price: float,
        ask_price: float,
        buy_size: int,
        sell_size: int,
        current_orders: Optional[List[Dict]] = None,
        mid_price: float = 0.5,
    ):
        if current_orders is None:
            current_orders = self.api.get_orders()

        buy_orders: List[Dict] = []
        sell_orders: List[Dict] = []

        for order in current_orders:
            if order["side"] == self.trade_side:
                if order["action"] == "buy":
                    buy_orders.append(order)
                elif order["action"] == "sell":
                    sell_orders.append(order)

        self.handle_order_side("buy", buy_orders, bid_price, buy_size, mid_price)
        self.handle_order_side("sell", sell_orders, ask_price, sell_size, mid_price)

    def handle_order_side(self, action: str, orders: List[Dict], desired_price: float, desired_size: int, mid_price: float):
        keep_order = None

        for order in orders:
            if self.trade_side == "yes":
                raw = order.get("yes_price_dollars", order.get("yes_price"))
                current_price = float(raw) if "." in str(raw) else float(raw) / 100
            else:
                raw = order.get("no_price_dollars", order.get("no_price"))
                current_price = float(raw) if "." in str(raw) else float(raw) / 100
            remaining = int(float(order.get("remaining_count", order.get("remaining_count_fp", 0))))
            if (
                keep_order is None
                and abs(current_price - desired_price) < 0.01
                and remaining == desired_size
            ):
                keep_order = order
            else:
                self.api.cancel_order(order["order_id"])

        if desired_size <= 0:
            return

        should_place = (action == "buy" and desired_price <= mid_price) or (
            action == "sell" and desired_price >= mid_price
        )

        if keep_order is None and should_place:
            self.api.place_order(
                action,
                self.trade_side,
                desired_price,
                desired_size,
                int(time.time()) + self.order_expiration,
            )
