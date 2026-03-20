### ✅ COMPLETED: Live Bot Risk Management Synchronized with Backtest

**Changes Implemented:**
1. **Minimum SL Distance (0.1%)**: Position sizing now uses `max(sl_distance, entry_price * 0.001)` to prevent excessive leverage in low-volatility conditions.
2. **Position Size Cap (10% Equity)**: Hard limit ensures no single position exceeds 10% of account equity, matching backtest safety constraints.
3. **Precision Handling**: All order prices and amounts are properly formatted using exchange precision requirements via CCXT.

**Files Modified:**
- `src/bot.py` - Core risk calculation in `_execute_trade()` method
- `src/bingx_client.py` - Added `load_markets()` for precision data

**Result:**
The live bot now replicates the backtest risk management exactly, preventing account blow-up scenarios from tight stop losses or oversized positions.

See [walkthrough documentation](file:///c:/Users/Nexus/.gemini/antigravity/brain/e4d72324-3063-4c75-8fde-f868f4cf21fb/walkthrough.md) for detailed implementation review.
