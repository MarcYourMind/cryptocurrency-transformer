import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from .config import Config
from .train_gpt import VolumeProfileTransformer, load_data, TradingDataset, is_sideways, compute_volume_profile, get_context_features
from torch.utils.data import DataLoader

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Global variables for workers
worker_model = None
worker_scaler = None
worker_device = None

def init_worker():
    global worker_model, worker_scaler, worker_device
    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    worker_model = VolumeProfileTransformer(Config).to(worker_device)
    worker_model.load_state_dict(torch.load(os.path.join(Config.ResultsDir, "best_model.pth"), map_location=worker_device))
    worker_model.eval()
    worker_scaler = joblib.load(os.path.join(Config.ResultsDir, "scaler.pkl"))

def process_single_symbol(f, threshold):
    global worker_model, worker_scaler, worker_device
    symbol = f.split("_")[0]
    
    try:
        df = pd.read_csv(os.path.join(Config.DataDir, f))
    except Exception as e:
        return symbol, None, f"Error reading {f}: {e}"
        
    if df.empty or len(df) < Config.LookbackWindow + Config.ForwardLook:
        return symbol, [], 0
        
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    start_idx = int(len(df) * (Config.TrainSplit + Config.ValSplit))
    
    windows_data = []
    # Optimization: pre-extract columns
    df_low = df['low'].values
    df_high = df['high'].values
    df_ts = df['timestamp'].values
    
    for i in range(start_idx, len(df) - Config.LookbackWindow - Config.ForwardLook, 4):
        window = df.iloc[i : i + Config.LookbackWindow]
        if not is_sideways(window):
            continue
            
        future_lows = df_low[i + Config.LookbackWindow : i + Config.LookbackWindow + Config.ForwardLook]
        future_highs = df_high[i + Config.LookbackWindow : i + Config.LookbackWindow + Config.ForwardLook]
        current_time = df_ts[i + Config.LookbackWindow - 1]
        
        try:
            prof = compute_volume_profile(window, Config.NumVolumeBins)
            base_ctx = get_context_features(window)
        except:
            continue

        atr = base_ctx[0]
        low = window['low'].min()
        high = window['high'].max()
        price_range = high - low
        if price_range <= 0: continue
        bin_width = price_range / Config.NumVolumeBins
        
        windows_data.append({
            'idx': i,
            'time': current_time,
            'prof': prof,
            'base_ctx': base_ctx,
            'atr': atr,
            'low': low,
            'bin_width': bin_width,
            'future_lows': future_lows,
            'future_highs': future_highs
        })

    if not windows_data:
        return symbol, [], 0

    all_batch_prof = []
    all_batch_ctx = []
    
    for win in windows_data:
        for bin_idx in range(Config.NumVolumeBins):
            ctx_item = win['base_ctx'] + [bin_idx / Config.NumVolumeBins]
            all_batch_prof.append(win['prof'])
            all_batch_ctx.append(ctx_item)
    
    # Process in smaller chunks to further save memory if needed
    all_probs = []
    with torch.no_grad():
        # Scale in chunks to avoid large intermediate arrays
        batch_size = 4096
        for i in range(0, len(all_batch_prof), batch_size):
            chunk_prof = all_batch_prof[i:i+batch_size]
            chunk_ctx = all_batch_ctx[i:i+batch_size]
            
            t_prof = torch.tensor(np.array(chunk_prof), dtype=torch.float32).to(worker_device)
            t_ctx = torch.tensor(worker_scaler.transform(np.array(chunk_ctx)), dtype=torch.float32).to(worker_device)
            
            logits = worker_model(t_prof, t_ctx)
            probs = torch.sigmoid(logits).squeeze().tolist()
            if isinstance(probs, float): probs = [probs]
            all_probs.extend(probs)
            
            # Explicitly clear chunk tensors
            del t_prof, t_ctx
    
    symbol_trades = []
    missed_count = 0
    
    all_best_probs = []
    for win_idx, win in enumerate(windows_data):
        start_p = win_idx * Config.NumVolumeBins
        end_p = start_p + Config.NumVolumeBins
        win_probs = all_probs[start_p:end_p]
        
        candidates = []
        for bin_idx, prob in enumerate(win_probs):
            candidates.append({
                'prob': prob,
                'price': win['low'] + (bin_idx + 0.5) * win['bin_width']
            })
        
        candidates.sort(key=lambda x: x['prob'], reverse=True)
        best_candidate = candidates[0]
        
        all_best_probs.append(best_candidate['prob'])
        
        if best_candidate['prob'] < threshold:
            continue
        
        e_price = best_candidate['price']
        f_lows = win['future_lows']
        f_highs = win['future_highs']
        
        entry_idx = -1
        for f_idx in range(len(f_lows)):
            if f_lows[f_idx] <= e_price <= f_highs[f_idx]:
                entry_idx = f_idx
                break
        
        if entry_idx == -1:
            missed_count += 1
        else:
            atr = win['atr']
            sl_price = e_price - (atr * Config.StopLossMultiplierATR)
            tp_price = e_price + (atr * Config.StopLossMultiplierATR * Config.RiskRewardRatio)
            
            outcome_val = 0
            for j in range(entry_idx, len(f_lows)):
                if f_lows[j] <= sl_price:
                    outcome_val = -1
                    break
                if f_highs[j] >= tp_price:
                    outcome_val = 1
                    break
            
            if outcome_val != 0:
                # Find exit time and price
                exit_idx = entry_idx
                exit_price = e_price
                for j in range(entry_idx, len(f_lows)):
                    if f_lows[j] <= sl_price:
                        exit_price = sl_price
                        exit_idx = j
                        outcome_val = -1
                        break
                    if f_highs[j] >= tp_price:
                        exit_price = tp_price
                        exit_idx = j
                        outcome_val = 1
                        break
                
                # If still open, close at the last candle
                if outcome_val == 0:
                    exit_idx = len(f_lows) - 1
                    exit_price = f_lows[exit_idx] # Or close, but we have lows/highs here
                    outcome_val = 1 if exit_price > e_price else -1

                exit_time = win['time'] + pd.Timedelta(minutes=(exit_idx + 1) * 15)
                
                # Capture floating prices at 1h intervals for equity curve
                # Time 0 is entry_idx. exit_idx is the end.
                floating_prices = []
                # Absolute index in df_low
                abs_start_idx = win['idx'] + Config.LookbackWindow
                
                for step_idx in range(entry_idx, exit_idx + 1, 4): # Every 1h (4 * 15m)
                    t_at_step = win['time'] + pd.Timedelta(minutes=(step_idx + 1) * 15)
                    # Use midpoint for "floating" price to be more representative than just low
                    p_at_step = (df_low[abs_start_idx + step_idx] + df_high[abs_start_idx + step_idx]) / 2
                    floating_prices.append((t_at_step, p_at_step))
                
                symbol_trades.append({
                    'entry_time': win['time'] + pd.Timedelta(minutes=(entry_idx + 1) * 15),
                    'exit_time': exit_time,
                    'symbol': symbol,
                    'prob': best_candidate['prob'],
                    'entry_price': e_price,
                    'exit_price': exit_price,
                    'outcome': outcome_val,
                    'atr': win['atr'],
                    'floating_prices': floating_prices
                })
                
    return symbol, symbol_trades, {'missed_count': missed_count, 'all_best_probs': all_best_probs}

def run_backtest(num_symbols=10):
    print(f"--- Starting Optimized Parallel Backtest (Symbols: {num_symbols}) ---")
    start_time = datetime.now()
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(os.path.join(Config.ResultsDir, "best_model.pth")):
        print("Model not found! Run training first.")
        return

    # threshold = 0.518
    threshold = 0.5
    initial_equity = 10000
    risk_per_trade = 0.01
    
    files = sorted([f for f in os.listdir(Config.DataDir) if f.endswith(".csv")])
    if num_symbols > 0:
        files = files[:num_symbols]
    
    all_trades = []
    all_probs_dist = []
    total_missed = 0
    
    print(f"Launching parallel processing for {len(files)} symbols with 4 workers...")
    
    # Use 2 workers to stay safe on memory
    with ProcessPoolExecutor(max_workers=2, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_symbol, f, threshold): f for f in files}
        
        for future in tqdm(as_completed(futures), total=len(files), desc="Backtesting"):
            try:
                symbol, symbol_trades, extra_info = future.result()
                
                if isinstance(extra_info, str):
                    print(f"\n[{symbol}] {extra_info}")
                    continue
                    
                if symbol_trades:
                    all_trades.extend(symbol_trades)
                
                if isinstance(extra_info, dict):
                    total_missed += extra_info.get('missed_count', 0)
                    all_probs_dist.extend(extra_info.get('all_best_probs', []))
            except Exception as e:
                print(f"\nWorker error: {e}")

    if not all_trades:
        print("No trades generated.")
        return

    # Aggregate metrics
    wins = sum(1 for t in all_trades if t['outcome'] == 1)
    losses = sum(1 for t in all_trades if t['outcome'] == -1)
    total_trades = len(all_trades)
    
    # Sort trades by entry time
    all_trades.sort(key=lambda x: x['entry_time'])
    
    # 1. Calculate PnL and Position Sizes with Compounding
    from heapq import heappush, heappop
    exit_queue = [] # (exit_time, pnl)
    current_realized_balance = initial_equity
    
    for t in all_trades:
        # Process all exits that happened before or at this entry
        while exit_queue and exit_queue[0][0] <= t['entry_time']:
            _, p_pnl = heappop(exit_queue)
            current_realized_balance += p_pnl
        
        # Risk is based on realized balance at the moment of entry
        risk_amount = current_realized_balance * risk_per_trade
        
        # Position size = risk / sl_distance
        sl_distance = max(t['atr'] * Config.StopLossMultiplierATR, t['entry_price'] * 0.001) # Min 0.1% SL
        desired_pos_size = risk_amount / sl_distance if sl_distance > 0 else 0
        # Cap position size to 10% equity exposure
        t['pos_size'] = min(desired_pos_size, current_realized_balance * 0.1 / t['entry_price'])
        
        # Calculate final PnL based on actual price delta and (possibly capped) pos_size
        t['final_pnl'] = (t['exit_price'] - t['entry_price']) * t['pos_size']
        
        heappush(exit_queue, (t['exit_time'], t['final_pnl']))
    
    # --- Equity Curve Calculation (Highly Accurate) ---
    print("\nCalculating high-accuracy time-based equity curve...")
    
    test_start = min(t['entry_time'] for t in all_trades)
    test_end = max(t['exit_time'] for t in all_trades)
    
    # Create a 1-hour grid for visibility
    sampling_freq = "1h"
    equity_times = pd.date_range(start=test_start, end=test_end, freq=sampling_freq)
    
    equity_curve = []
    
    # Pre-calculate realized balance milestones to speed up
    # (time, total_realized_pnl)
    pnl_milestones = []
    acc_pnl = 0
    # Collect all exit events
    exit_events = sorted([(t['exit_time'], t['final_pnl']) for t in all_trades])
    for et, ep in exit_events:
        acc_pnl += ep
        pnl_milestones.append((et, acc_pnl))
    
    for et in tqdm(equity_times, desc="Generating Curve"):
        # Realized Balance at et
        # Find latest milestone <= et
        idx = 0
        current_acc_pnl = 0
        # Simple search for now, could be binary search
        for m_t, m_p in pnl_milestones:
            if m_t <= et:
                current_acc_pnl = m_p
            else:
                break
        
        current_rb = initial_equity + current_acc_pnl
        
        # Unrealized PnL at et
        current_upnl = 0
        for t in all_trades:
            if t['entry_time'] <= et < t['exit_time']:
                # Find the closest price captured at or before et
                p_at_t = t['entry_price']
                for pt, pp in t['floating_prices']:
                    if pt <= et:
                        p_at_t = pp
                    else:
                        break
                
                # PnL = (current_price - entry_price) * pos_size
                # Note: For shorts it would be (entry - current), but our code handles 
                # outcome value globally. For simplicity, we assume 'outcome' is 
                # positive for wins. In this strategy, we are only doing longs (entry - atr = SL).
                current_upnl += (p_at_t - t['entry_price']) * t['pos_size']
        
        equity_curve.append(current_rb + current_upnl)

    final_equity = initial_equity + sum(t['final_pnl'] for t in all_trades)
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    print(f"\nFinal Equity: ${final_equity:.2f}")
    print(f"Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Missed: {total_missed}")
    print(f"Win Rate: {win_rate*100:.1f}%")
    
    # Classification Metrics
    y_true = [1 if t['outcome'] == 1 else 0 for t in all_trades]
    y_scores = [t['prob'] for t in all_trades]
    
    accuracy = accuracy_score(y_true, [1 if p > 0.5 else 0 for p in y_scores]) if y_true else 0
    f1 = f1_score(y_true, [1 if p > 0.5 else 0 for p in y_scores]) if y_true else 0
    try:
        auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0
    except:
        auc = 0
    
    print(f"Accuracy: {accuracy:.2f}, F1: {f1:.2f}, AUC: {auc:.2f}")

    # Save and Plot
    results = {
        'timestamp': timestamp_str,
        'metrics': {
            'initial_equity': initial_equity,
            'final_equity': final_equity,
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'missed': total_missed,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        },
        'equity_curve': equity_curve,
        'equity_times': [et.strftime("%Y-%m-%d %H:%M") for et in equity_times]
    }
    
    json_path = os.path.join(Config.ResultsDir, f"results_{timestamp_str}.json")
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=4)
    
    plt.figure(figsize=(12, 7))
    plt.plot(equity_times, equity_curve, label="Equity")
    plt.title(f"High-Accuracy Equity Curve over TIME\nFinal: ${final_equity:.2f} (WR: {win_rate*100:.1f}%)")
    plt.ylabel("Portfolio Equity ($)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    
    # Highlight start/end
    plt.scatter([equity_times[0], equity_times[-1]], [equity_curve[0], equity_curve[-1]], color='red')
    plt.text(equity_times[0], equity_curve[0], f" Start: ${initial_equity}", verticalalignment='bottom')
    plt.text(equity_times[-1], equity_curve[-1], f" End: ${final_equity:.2f}", verticalalignment='top')
    
    plt.gcf().autofmt_xdate() # Format dates
    plt.savefig(os.path.join(Config.ResultsDir, "equity_curve.png"))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_probs_dist, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
    plt.title(f"Prediction Distribution (Total Windows: {len(all_probs_dist)})")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(Config.ResultsDir, "prediction_dist.png"))
    plt.close()

    print(f"Backtest completed in {datetime.now() - start_time}")

def main():
    parser = argparse.ArgumentParser(description='Run backtest on a specified number of symbols.')
    parser.add_argument('--symbols', '-n', type=int, default=10, help='Number of symbols to run the backtest on (default: 10). Use 0 for all symbols.')
    args = parser.parse_args()
    
    run_backtest(num_symbols=args.symbols)

if __name__ == "__main__":
    main()
