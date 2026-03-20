import ccxt
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class BingXClient:
    def __init__(self):
        self.api_key = os.getenv("BINGX_API_KEY")
        self.secret_key = os.getenv("BINGX_SECRET_KEY")
        self.dry_run = os.getenv("DRY_RUN", "True").lower() == "true"
        
        self.exchange = ccxt.bingx({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'options': {
                'defaultType': 'swap', # Perpetual swaps
            }
        })
        try:
            self.exchange.load_markets()
        except Exception as e:
            print(f"Warning: Failed to load markets: {e}")

    def fetch_balance(self):
        """Fetch USDT balance."""
        balance = self.exchange.fetch_balance()
        return balance['USDT']['total']

    def fetch_klines(self, symbol, timeframe='15m', limit=100):
        """Fetch latest klines and return a DataFrame."""
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df

    def set_leverage(self, symbol, leverage):
        """Set leverage for a symbol."""
        try:
            return self.exchange.set_leverage(leverage, symbol)
        except Exception as e:
            print(f"Error setting leverage: {e}")
            return None

    def place_order(self, symbol, side, amount, price=None, params={}):
        """Place an order. Respect DRY_RUN setting."""
        if self.dry_run:
            print(f"[DRY RUN] Would place {side} order for {amount} {symbol} at {price}")
            return {"id": "dry_run_id", "status": "closed"}
        
        if side.lower() == 'buy':
            return self.exchange.create_market_buy_order(symbol, amount, params)
        else:
            return self.exchange.create_market_sell_order(symbol, amount, params)

    def place_limit_order_with_sl_tp(self, symbol, side, amount, price, sl_price, tp_price):
        """
        Place a limit order with SL and TP.
        Note: BingX might require specific SL/TP params in the initial order or separate calls.
        """
        params = {
            'stopLossPrice': sl_price,
            'takeProfitPrice': tp_price
        }
        
        if self.dry_run:
            print(f"[DRY RUN] Would place {side} LIMIT order for {amount} {symbol} at {price}")
            print(f"[DRY RUN] SL: {sl_price}, TP: {tp_price}")
            return {"id": "dry_run_id", "status": "open"}

        return self.exchange.create_order(symbol, 'limit', side, amount, price, params)

    def fetch_open_positions(self, symbol=None):
        """Fetch current open positions."""
        return self.exchange.fetch_positions(symbols=[symbol] if symbol else None)
    
    def fetch_open_orders(self, symbol=None):
        """Fetch current open orders."""
        return self.exchange.fetch_open_orders(symbol)

    def cancel_order(self, order_id, symbol):
        """Cancel an open order."""
        if self.dry_run:
            print(f"[DRY RUN] Would cancel order {order_id} for {symbol}")
            return True
        return self.exchange.cancel_order(order_id, symbol)

    def fetch_order(self, order_id, symbol):
        """Fetch a specific order by ID."""
        if self.dry_run:
            # Return a mock order for dry run
            return {
                'id': order_id,
                'symbol': symbol,
                'status': 'open', # Default to open for testing state transitions manually if needed
                'filled': 0.0,
                'remaining': 1.0,
            }
        return self.exchange.fetch_order(order_id, symbol)
