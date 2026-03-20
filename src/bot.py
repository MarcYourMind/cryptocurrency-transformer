import os
import time
import logging
import threading
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import ccxt
from logging.handlers import RotatingFileHandler
from .bingx_client import BingXClient
from .strategy import Strategy

load_dotenv()

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("bot.log", maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

from .config import Config

class TradingBot:
    def __init__(self):
        self.client = BingXClient()
        self.strategy = Strategy()
        
        # Multi-symbol support
        # 1. Load from DataDir to match backtest.py
        try:
            files = sorted([f for f in os.listdir(Config.DataDir) if f.endswith(".csv")])
            num_symbols = int(os.getenv("NUM_SYMBOLS", 10))
            if num_symbols > 0:
                files = files[:num_symbols]
            
            self.symbols = []
            for f in files:
                # e.g. BTCUSDT_15m.csv -> BTCUSDT
                raw_symbol = f.split("_")[0]
                # Convert to CCXT format: BTCUSDT -> BTC/USDT
                # Assumption: all symbols end in USDT
                if raw_symbol.endswith("USDT"):
                    base = raw_symbol[:-4]
                    quote = "USDT"
                    self.symbols.append(f"{base}/{quote}")
                else:
                    self.symbols.append(raw_symbol)
            
            logger.info(f"Loaded {len(self.symbols)} symbols from {Config.DataDir} (Top {num_symbols})")
            
        except Exception as e:
            logger.error(f"Error loading symbols from DataDir: {e}")
            # Fallback
            symbols_str = os.getenv("TRADING_SYMBOLS", "BTC/USDT")
            self.symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        
        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", 0.01))
        self.leverage = int(os.getenv("LEVERAGE", 5))
        self.is_running = False
        self.last_check_15m = None
        self.thread = None
        
        # State tracking
        self.state_file = "trades_state.json"
        self.active_trades, self.history = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('active_trades', []), data.get('history', [])
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        return [], []

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({
                    'active_trades': self.active_trades,
                    'history': self.history
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            logger.info("Bot started.")

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("Bot stopped.")

    def _run(self):
        logger.info(f"Monitoring symbols: {', '.join(self.symbols)}...")
        while self.is_running:
            try:
                now = datetime.now()
                
                # 1. 15m Candle Check
                # Run on every candle close (00, 15, 30, 45 minutes)
                if now.minute % 15 == 0 and now.minute != self.last_check_15m:
                    for symbol in self.symbols:
                        self._check_for_signals(symbol)
                    self.last_check_15m = now.minute
                    
                # 2. 5m Management Check
                if now.minute % 5 == 0:
                    self._manage_orders()
                
                # Sleep for 30s to avoid busy wait
                time.sleep(30)
                
            except ccxt.RateLimitExceeded as e:
                logger.error(f"Rate limit exceeded: {e}. Sleeping for 2 minutes.")
                time.sleep(120)
            except ccxt.NetworkError as e:
                logger.error(f"Network error: {e}. Retrying in 10s.")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Error in bot loop: {e}")
                time.sleep(60)

    def _check_for_signals(self, symbol):
        logger.info(f"Checking for new signals on {symbol}...")
        try:
            df = self.client.fetch_klines(symbol)
            signal = self.strategy.get_signal(df)
            
            if signal:
                logger.info(f"Signal found for {symbol}: {signal}")
                self._execute_trade(symbol, signal)
            else:
                logger.info(f"No signal identified for {symbol}.")
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")

    def _execute_trade(self, symbol, signal):
        try:
            balance = self.client.fetch_balance()
            entry_price = signal['entry_price']
            sl_price = signal['sl_price']
            tp_price = signal['tp_price']
            
            # Risk Management
            # Risk Management
            sl_dist = abs(entry_price - sl_price)
            if sl_dist == 0: return
            
            # 1. Enforce Minimum SL Distance 0.1% for Sizing
            min_sl_dist = entry_price * 0.001
            clamped_sl_dist = max(sl_dist, min_sl_dist)
            
            risk_amount = balance * self.risk_per_trade
            raw_amount_to_trade = risk_amount / clamped_sl_dist
            
            # 2. Cap Position Size at 10% of Equity
            max_pos_size_value = balance * 0.1
            max_amount_to_trade = max_pos_size_value / entry_price
            
            amount_to_trade = min(raw_amount_to_trade, max_amount_to_trade)
            
            # 3. Precision Handling
            # Ensure price and amount are valid for the exchange
            try:
                amount_to_trade = self.client.exchange.amount_to_precision(symbol, amount_to_trade)
                entry_price = float(self.client.exchange.price_to_precision(symbol, entry_price))
                sl_price = float(self.client.exchange.price_to_precision(symbol, sl_price))
                tp_price = float(self.client.exchange.price_to_precision(symbol, tp_price))
            except Exception as e:
                logger.error(f"Precision handling error: {e}")
                # Fallback to basic rounding if needed, but preferable to rely on ccxt
                return
            
            # Convert amount back to float for logging/params as ccxt returns string usually
            amount_to_trade = float(amount_to_trade)

            logger.info(f"Risk Calc: Bal={balance}, Risk={risk_amount:.2f}, SL Dist={sl_dist:.4f} (Clamped={clamped_sl_dist:.4f})")
            logger.info(f"Pos Size: Raw={raw_amount_to_trade:.4f}, Max={max_amount_to_trade:.4f} -> Final={amount_to_trade}")
            
            # Set Leverage
            self.client.set_leverage(symbol, self.leverage)
            
            # Place Limit Order
            logger.info(f"Placing LIMIT BUY on {symbol} at {entry_price} with SL={sl_price} and TP={tp_price}")
            order = self.client.place_limit_order_with_sl_tp(
                symbol, 'buy', amount_to_trade, entry_price, sl_price, tp_price
            )
            
            if order:
                self.active_trades.append({
                    'symbol': symbol,
                    'order_id': order.get('id'),
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'amount': amount_to_trade,
                    'status': 'open',
                    'timestamp': datetime.now().isoformat()
                })
                self._save_state()
        except Exception as e:
            logger.error(f"Execution failed: {e}")

    def _manage_orders(self):
        """Check status of open orders/positions."""
        if not self.active_trades:
            return
            
        logger.info("Managing open orders...")
        updated_trades = []
        state_changed = False

        try:
            # Fetch current positions
            positions = self.client.fetch_open_positions()
            # open_orders = self.client.fetch_open_orders(self.symbol)  # Removed: unused and buggy
            # open_order_ids = [o['id'] for o in open_orders]           # Removed: unused

            for trade in self.active_trades:
                symbol = trade['symbol']
                order_id = trade['order_id']
                
                # Case 1: Trade is still an open limit order
                if trade['status'] == 'open':
                    try:
                        order = self.client.fetch_order(order_id, symbol)
                        status = order['status']
                        
                        if status == 'open':
                            updated_trades.append(trade)
                        elif status == 'closed' or status == 'filled':
                            # Order filled, verify position exists just to be sure
                            pos = next((p for p in positions if p['symbol'] == symbol), None)
                            if pos and float(pos['contracts']) > 0:
                                logger.info(f"Order {order_id} filled. Now tracking position for {symbol}.")
                                trade['status'] = 'filled'
                                updated_trades.append(trade)
                                state_changed = True
                            else:
                                # Could happen if filled then immediately closed, or if api lag
                                logger.warning(f"Order {order_id} filled but no position found. Marking as closed.")
                                trade['status'] = 'closed'
                                trade['closed_at'] = datetime.now().isoformat()
                                self.history.append(trade)
                                state_changed = True
                        elif status in ['canceled', 'rejected', 'expired']:
                            logger.info(f"Order {order_id} was {status}.")
                            state_changed = True
                        else:
                            # Unknown status, keep tracking
                            logger.warning(f"Unknown order status for {order_id}: {status}")
                            updated_trades.append(trade)

                    except ccxt.OrderNotFound:
                        logger.error(f"Order {order_id} not found. Assuming cancelled.")
                        state_changed = True
                    except Exception as e:
                        logger.error(f"Error fetching order {order_id}: {e}")
                        updated_trades.append(trade)
                
                # Case 2: Trade is an active position
                elif trade['status'] == 'filled':
                    pos = next((p for p in positions if p['symbol'] == symbol), None)
                    if not pos or float(pos['contracts']) == 0:
                        logger.info(f"Position for {symbol} closed.")
                        trade['status'] = 'closed'
                        trade['closed_at'] = datetime.now().isoformat()
                        self.history.append(trade)
                        state_changed = True
                    else:
                        updated_trades.append(trade)

            if state_changed:
                self.active_trades = updated_trades
                self._save_state()

        except Exception as e:
            logger.error(f"Error managing orders: {e}")

    def get_status(self):
        return {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "active_trades_count": len(self.active_trades),
            "last_check_15m": self.last_check_15m
        }
