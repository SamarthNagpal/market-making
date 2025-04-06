from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_HALF_EVEN
from typing import Dict

from hummingbot.core.data_type.common import PriceType, OrderType, TradeType
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory, CandlesConfig
from hummingbot.core.data_type.order_candidate import OrderCandidate

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import math

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN
DECIMAL_PLACES = Decimal("0.0001")
def d(val):
    return Decimal(val).quantize(DECIMAL_PLACES, rounding=ROUND_HALF_UP)

class MyCustomStrategy(ScriptStrategyBase):
    # === CONFIG PARAMETERS ===
    trading_pair = "ETH-USDT"
    base_asset = trading_pair.split('-')[0]
    quote_asset = trading_pair.split('-')[1]

    exchange = "binance_paper_trade"
    candle_exchange = "binance"
    order_refresh_time = 15
    order_amount = d(0.01)
    price_source = PriceType.MidPrice
    base_spread = d(0.01) # 1%
    min_spread = d(0.002) # 0.2% not used yet
    max_spread = d(0.05)   # 5% not used yet
    


    # === INTERNAL STATE ===
    create_timestamp = 0  # used to space out actions every `order_refresh_time` seconds

    # === REQUIRED: define market to connect to ===
    # markets = {exchange: {trading_pair}}
    markets = {exchange: [trading_pair]}



    def __init__(self, connectors: Dict):
        super().__init__(connectors)
        self.connector = self.connectors[self.exchange]
        # Initialize anything you want to start when the strategy starts
        self._candles_cache = {}
        self.logger().info("Strategy initialized.")

    def on_stop(self):
        # Stop anything (feeds, models, etc.) cleanly here
        self.logger().info("Strategy stopped.")

    def on_tick(self):
        # if not (self.connectors.get("binance") and self.connectors["binance"].is_ready):
        #     self.logger().warning("Exchange not ready. Skipping this tick.")
        #     return
        
        if self.current_timestamp >= self.create_timestamp:
            self.cancel_all_orders()

            vol_score = self.volatility_factor()
            vol_score = max(min(vol_score, 1.5), -1.0)               # Clamp vol score into some reasonable range, just in case
            # Optional: Make it smoother with a multiplier or exponential function
            vol_multiplier = d(1) + d(0.5) * d(vol_score)     # adjust 0.5 as sensitivity factor
            volume_score = self.volume_factor()
            volume_score = max(min(volume_score, 1.5), -1.0)
            self.logger().info(f"[Volume Debug] Volume Score: {float(volume_score):.4f}")
            volume_multiplier = d(1) - d(0.3) * d(volume_score)
            adjusted_spread = self.base_spread * vol_multiplier * volume_multiplier      # Apply to base spread
            bid_spread = d(adjusted_spread)
            ask_spread = d(adjusted_spread)

            mid_price = self.connector.get_mid_price(self.trading_pair).quantize(d("0.0001"))

            inventory_skew = d(self.inventory_skew_tanh())/mid_price  # convert to % change
            trend_skew = d(self.short_term_trend()) + d(0.5) * d(self.long_term_trend())
            total_skew = d(inventory_skew + trend_skew)
  
            # Final prices
            if math.isnan(bid_spread) or math.isnan(ask_spread):
                self.logger().warning("Spread is NaN!")
            if math.isnan(total_skew):
                self.logger().warning("Skew is NaN!")
            total_skew = d(max(min(total_skew, 0.3), -0.3))
            # bid_price = mid_price * (d(1) - bid_spread + total_skew)
            # ask_price = mid_price * (d(1) + ask_spread + total_skew)
            


            bid_price = mid_price * (d(1) - bid_spread) * (d(1) + total_skew)
            ask_price = mid_price * (d(1) + ask_spread) * (d(1) + total_skew)

            self.logger().info(
                f"[Skew Debug] inventory_skew={float(inventory_skew):.6f}, short_term_trend={float(self.short_term_trend()):.6f}, long_term_trend={float(self.long_term_trend()):.6f}"
            )



            self.logger().info(f"Mid price: {mid_price:.2f}, Bid: {bid_price:.2f}, Ask: {ask_price:.2f}, Spread: {float(bid_spread):.4f}, Total skew: {total_skew:.5f}")


            buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT, order_side=TradeType.BUY, amount=d(self.order_amount), price=d(bid_price))
            sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT, order_side=TradeType.SELL, amount=d(self.order_amount), price=d(ask_price))
            proposal = self.connector.budget_checker.adjust_candidates([buy_order, sell_order], all_or_none=True)
            self.logger().info(f"Placing orders - Buy @ {bid_price:.2f}, Sell @ {ask_price:.2f}")
            for order in proposal:
                self.place_order(self.exchange, order)

            # Update timestamp to throttle actions
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def format_status(self) -> str:
        # Used by `status` command to print balance / orders / etc.
        if not self.ready_to_trade:
            return "Market connectors are not ready."

        lines = []

        balance_df = self.get_balance_df()
        lines.extend(["", "  Balances:"] + ["    " + line for line in balance_df.to_string(index=False).split("\n")])

        try:
            df = self.active_orders_df()
            lines.extend(["", "  Orders:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        except ValueError:
            lines.append("  No active maker orders.")

        return "\n".join(lines)
    
    def inventory_skew_tanh(self) -> float:
        max_skew_pct = 0.003  # 0.3% of price
        sensitivity = 3

        # base = self.balance_asset(self.base_asset)
        # base = self.markets.get_balance(self.base_asset)
        # quote = self.balance_asset(self.quote_asset)
        # quote = self.markets.get_balance(self.quote_asset)
        # mid_price = self.markets.get_mid_price(self.trading_pair)

        base = self.connector.get_balance(self.base_asset)
        quote = self.connector.get_balance(self.quote_asset)
        mid_price = self.connector.get_mid_price(self.trading_pair)

        if mid_price <= 0:
            return 0.0

        base_equiv = quote / mid_price
        total = base + base_equiv
        if total == 0:
            return 0.0

        ν = (base - base_equiv) / total
        max_skew = max_skew_pct * float(mid_price)
        return max_skew * math.tanh(sensitivity * float(ν))

    
    def get_candles(self, interval: str, max_records: int) -> pd.DataFrame:
        key = f"{self.trading_pair}_{interval}"
        if key not in self._candles_cache:
            config = CandlesConfig(
                connector=self.candle_exchange,
                trading_pair=self.trading_pair,
                interval=interval,
                max_records=max_records
            )
            candle = CandlesFactory.get_candle(config)
            candle.start()
            self._candles_cache[key] = candle

        candles = self._candles_cache[key]
        return candles.candles_df if candles.candles_df is not None else pd.DataFrame()

    
    def short_term_trend(self): # returns how much market is in short term uptrend or downtrend
        interval = '1m'
        max_records = 50

        candles = self.get_candles(interval=interval, max_records=max_records)
        if candles.empty or len(candles) < max_records:
            return 0.0

        close_prices = candles["close"]
        ema20 = close_prices.ewm(span=20).mean().iloc[-1]
        ema50 = close_prices.ewm(span=50).mean().iloc[-1]

        short_term_factor = float((ema20 - ema50) / ema50)
        return d(short_term_factor)
    

    def long_term_trend(self): # returns how likely market is for mean reversion
        if not hasattr(self, "_long_term_trend_cache"):
            interval = "1h"
            max_records = 100
            min_required = 50

            candles = self.get_candles(interval=interval, max_records=max_records)
            if candles.empty or len(candles) < min_required:
                self._long_term_trend_cache = 0.0
                return self._long_term_trend_cache

            close_prices = candles["close"].values
            if np.isnan(close_prices).any():
                self.logger().warning("[LTT Debug] NaNs in close prices!")

            mean = np.mean(close_prices)
            std = np.std(close_prices)
            std = max(std, 1e-3)
            self.logger().info(f"[LTT Debug] First ts: {candles.index[0]}, Last ts: {candles.index[-1]}")

            last_price = close_prices[-1]
            z_score = (last_price - mean) / std if std > 0 else 0.0

            self._long_term_trend_cache = d(z_score)

        return self._long_term_trend_cache

    
    def volatility_factor(self): # returns volatility factor for spread adjustment
        candles = self.get_candles(interval="5m", max_records=100)
        if candles.empty or len(candles) < 50:
            return 0.0

        close_prices = candles["close"].values
        log_returns = np.diff(np.log(close_prices)).reshape(-1, 1)

        # Remove any rows with NaN or Inf from log_returns
        log_returns = log_returns[~np.isnan(log_returns)]
        log_returns = log_returns[~np.isinf(log_returns)]
        log_returns = log_returns.reshape(-1, 1)

        # If after cleaning, log_returns is too short, just return 0.0
        if log_returns.shape[0] < 10:
            return 0.0

        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        model.fit(log_returns)

        hidden_states = model.predict(log_returns)
        state_stdevs = [np.std(log_returns[hidden_states == i]) if np.any(hidden_states == i) else 0.0 for i in range(3)]



        probs = model.predict_proba(log_returns)
        current_state_probs = probs[-1]
        current_volatility = float(np.dot(current_state_probs, state_stdevs))

        mean_stdev = np.mean(state_stdevs)
        volatility_score = (current_volatility - mean_stdev) / mean_stdev

        if any(np.isnan(state_stdevs)):
            return 0.0

        current_volatility = float(np.dot(current_state_probs, state_stdevs))
        mean_stdev = np.mean(state_stdevs)

        if mean_stdev == 0:
            return 0.0
        if np.isnan(volatility_score) or np.isinf(volatility_score):
            return 0.0
        
        return volatility_score
    
    def volume_factor(self):
        interval = "1m"
        max_records = 100
        min_required = 30

        candles = self.get_candles(interval=interval, max_records=max_records)
        if candles.empty or len(candles) < min_required:
            return 0.0

        volumes = candles["volume"]
        latest_volume = volumes.iloc[-1]
        avg_volume = volumes[:-1].mean()  # exclude last volume for fair comparison

        if avg_volume == 0:
            return 0.0

        raw_score = float((latest_volume - avg_volume) / avg_volume)

        # Clamp the score to be within reasonable range
        clamped_score = max(min(raw_score, 1.5), -1.0)
        return d(clamped_score)

    
    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        # if not self.config or not self.config.exchange:
        #     self.logger().warning("Config or exchange not set. Cannot cancel orders.")
        #     return
        self.logger().info("Cancelling all active orders.")
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

