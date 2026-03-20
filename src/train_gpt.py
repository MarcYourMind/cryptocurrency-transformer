import os
import time
import requests
import pandas as pd
import numpy as np
import torch
from .config import Config, Utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from joblib import Parallel, delayed


# ==========================================
# 2. DATA FETCHING
# ==========================================
def fetch_top_symbols(limit=50):
    print("Fetching top symbols from Binance...")
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)

        data = response.json()
        
        # List of stablecoins to exclude
        stablecoins = ['USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'USDD', 'USDP', 
                       'FDUSD', 'FRAX', 'GUSD', 'USDJ', 'USDN', 'USTC', 
                       'UST', 'SUSD', 'HUSD', 'PAX', 'PAXG', 'USD1']
        
        # Filter USDT pairs and exclude stablecoin pairs
        symbols = []
        for x in data:
            if x['symbol'].endswith('USDT'):
                # Extract base asset (part before USDT)
                base_asset = x['symbol'][:-4]  # Remove 'USDT' suffix
                # Only include if base asset is not a stablecoin
                if base_asset not in stablecoins:
                    symbols.append(x)
        
        # Sort by volume (quote volume)
        symbols = sorted(symbols, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        top_symbols = [x['symbol'] for x in symbols[:limit]]
        print(f"Top {len(top_symbols)} symbols: {top_symbols[:5]}...")
        return top_symbols
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def download_klines(symbol):
    filename = os.path.join(Config.DataDir, f"{symbol}_{Config.INTERVAL}.csv")
    if os.path.exists(filename):
        # In a real system we might update, for now skip if exists to save time
        print(f"Data for {symbol} already exists. Skipping download.")
        return

    print(f"Downloading data for {symbol}...")
    base_url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    # Start time: Let's get ~3 months of data. 
    # 3 months * 30 days * 24 hours * 60 mins = 129600 mins.
    # 15m interval => 129600 / 15 = 8640 candles
    # start_time = int(time.time() * 1000) - (90 * 24 * 60 * 60 * 1000)
    
    # For robust training we normally want years, but let's do 10000 candles for now
    limit = 1000
    end_time = int(time.time() * 1000)
    
    # We will fetch backwards or just start from a fixed point. 
    # Let's fetch last 10000 candles (increased from 5000)
    # Because Binance API standard is start_time -> forward.
    ncandles = 900*24*4  # 900 days in 15min candles
    
    start_time = end_time - (ncandles * 15 * 60 * 1000)
    
    current_start = start_time
    
    while True:
        params = {
            'symbol': symbol,
            'interval': Config.INTERVAL,
            'startTime': current_start,
            'limit': limit
        }
        res = requests.get(base_url, params=params)
        data = res.json()
        
        if not data or not isinstance(data, list):
            break
            
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        
        if len(data) < limit or current_start > end_time:
            break
        
        time.sleep(0.1) 
        
    if not all_klines:
        return

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    
    # Numeric conversion
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.to_csv(filename, index=False)

# ==========================================
# 3. PREPROCESSING & FEATURE ENGINEERING
# ==========================================
def load_data():
    market_data = {}
    files = sorted([f for f in os.listdir(Config.DataDir) if f.endswith(".csv")])
    
    # Limit to Config.SYMBOL_LIMIT
    files = files[:Config.SYMBOL_LIMIT]
    
    for f in files:
        symbol = f.split("_")[0]
        path = os.path.join(Config.DataDir, f)
        df = pd.read_csv(path)
        market_data[symbol] = df
    return market_data

def compute_volume_profile(window_df, bins=64):
    low = window_df['low'].min()
    high = window_df['high'].max()
    
    if high == low:
        return np.zeros(bins)
        
    price_range = high - low
    bin_size = price_range / bins
    
    # Bin indices
    # We want to distribute volume. 
    # A simple approach: use 'close' price of the candle to assign volume.
    # Better: assumes uniform distribution between low and high of the candle?
    # Let's stick to simple close-price binning for speed or mid-price
    
    mid_prices = (window_df['high'] + window_df['low']) / 2
    volumes = window_df['volume']
    
    # Digitize
    bins_edges = np.linspace(low, high, bins + 1)
    
    # np.histogram equivalent but weighted by volume
    profile, _ = np.histogram(mid_prices, bins=bins_edges, weights=volumes)
    
    # Normalize
    if profile.max() > 0:
        profile = profile / profile.max()
        
    return profile

def get_context_features(window_df):
    # ATR (approximate for the window)
    # Volatility (std dev of returns)
    # Avg Volume
    # Trend Slope of the window itself
    
    closes = window_df['close'].values
    returns = np.diff(closes) / closes[:-1]
    
    volatility = np.std(returns)
    avg_vol = window_df['volume'].mean()
    
    # Trend slope
    x = np.arange(len(closes)).reshape(-1, 1)
    y = (closes - closes.min()) / (closes.max() - closes.min() + 1e-9) # Normalize price for slope
    reg = LinearRegression().fit(x, y)
    slope = reg.coef_[0]
    
    # ATR - simplified: average (High - Low)
    atr = (window_df['high'] - window_df['low']).mean()
    
    return [atr, volatility, avg_vol, slope]

def is_sideways(window_df):
    # Linear regression on Close prices
    y = window_df['close'].values
    x = np.arange(len(y)).reshape(-1, 1)
    
    # Normalize y to be percentage change from start to make slope comparable across assets
    y_norm = (y - y[0]) / y[0]
    
    reg = LinearRegression().fit(x, y_norm)
    slope = reg.coef_[0]
    r2 = reg.score(x, y_norm)
    
    # Condition: Low slope AND (Low R2 OR Low Slope)
    # Basically if it's flat, slope is near 0.
    if abs(slope) < Config.SlopeThreshold:
        return True
        
    return False


def process_dataframe(df):
    """
    Generates samples from a single dataframe (e.g. one symbol's train split).
    """
    X_profiles = []
    X_context = []
    y_probs = []
    
    if len(df) < Config.LookbackWindow + Config.ForwardLook:
        return np.array([]), np.array([]), np.array([])
        
    # Sliding window logic
    # Stride 4 (1 hour) to reduce redundancy
    
    for i in range(0, len(df) - Config.LookbackWindow - Config.ForwardLook, 4):
        
        # Window
        window = df.iloc[i : i + Config.LookbackWindow]
        future = df.iloc[i + Config.LookbackWindow : i + Config.LookbackWindow + Config.ForwardLook]
        
        # 1. Check Regime
        if not is_sideways(window):
            continue
            
        # 2. Base Features
        prof = compute_volume_profile(window, Config.NumVolumeBins)
        base_ctx = get_context_features(window) # [atr, volatility, avg_volume, slope]
        
        atr = base_ctx[0]
        
        # 3. Iterate over 64 entry points
        low = window['low'].min()
        high = window['high'].max()
        price_range = high - low
        if price_range <= 0 or atr <= 0: continue
        
        bin_width = price_range / Config.NumVolumeBins
        
        # Pre-calculate future arrays
        f_lows = future['low'].values
        f_highs = future['high'].values
        
        for bin_idx in range(Config.NumVolumeBins):
            # Entry price is center of the bin
            entry_price = low + (bin_idx + 0.5) * bin_width
            
            # Check if future hits entry_price
            entry_idx_in_future = -1
            
            for f_idx in range(len(f_lows)):
                if f_lows[f_idx] <= entry_price <= f_highs[f_idx]:
                    entry_idx_in_future = f_idx
                    break
            
            if entry_idx_in_future == -1:
                continue
            
            # If entry hit, check outcome
            sl_price = entry_price - (atr * Config.StopLossMultiplierATR)
            tp_price = entry_price + (atr * Config.StopLossMultiplierATR * Config.RiskRewardRatio)
            
            outcome = -1
            for j in range(entry_idx_in_future, len(f_lows)):
                l = f_lows[j]
                h = f_highs[j]
                
                if l <= sl_price:
                    outcome = 0
                    break
                if h >= tp_price:
                    outcome = 1
                    break
            
            if outcome != -1:
                X_profiles.append(prof)
                # Add normalized bin index to context
                ctx = base_ctx + [bin_idx / Config.NumVolumeBins]
                X_context.append(ctx)
                y_probs.append(outcome)
    
    if len(y_probs) == 0:
         return np.array([]), np.array([]), np.array([])
         
    return np.array(X_profiles), np.array(X_context), np.array(y_probs)

def generate_datasets_split(market_data):
    """
    Splits data PER SYMBOL, then generates samples for Train/Val/Test.
    Returns (X_p_train, X_c_train, y_train), (X_p_val, ...), (X_p_test, ...)
    """
    print("Generating datasets with PER-SYMBOL splitting...")
    
    # Containers
    train_data = {'p': [], 'c': [], 'y': []}
    val_data = {'p': [], 'c': [], 'y': []}
    test_data = {'p': [], 'c': [], 'y': []}
    
    total_symbols = len(market_data)
    for s_idx, (symbol, df) in enumerate(market_data.items()):
        print(f"[{s_idx+1}/{total_symbols}] Processing {symbol} (len={len(df)})...")
        
        # Split indices
        total_len = len(df)
        train_end = int(total_len * Config.TrainSplit)
        val_end = int(total_len * (Config.TrainSplit + Config.ValSplit))
        
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]
        
        # Process patches
        # NOTE: We might lose a few samples at the boundaries due to lookback, 
        # but this ensures strict separation.
        
        # Train
        p, c, y = process_dataframe(df_train)
        if len(y) > 0:
            train_data['p'].append(p)
            train_data['c'].append(c)
            train_data['y'].append(y)
            
        # Val
        p, c, y = process_dataframe(df_val)
        if len(y) > 0:
            val_data['p'].append(p)
            val_data['c'].append(c)
            val_data['y'].append(y)
            
        # Test
        p, c, y = process_dataframe(df_test)
        if len(y) > 0:
            test_data['p'].append(p)
            test_data['c'].append(c)
            test_data['y'].append(y)
            
    # Concatenate
    def concat_arrays(list_of_arrays):
        if not list_of_arrays:
            return np.array([])
        return np.concatenate(list_of_arrays, axis=0)
        
    X_train = (concat_arrays(train_data['p']), concat_arrays(train_data['c']), concat_arrays(train_data['y']))
    X_val = (concat_arrays(val_data['p']), concat_arrays(val_data['c']), concat_arrays(val_data['y']))
    X_test = (concat_arrays(test_data['p']), concat_arrays(test_data['c']), concat_arrays(test_data['y']))
    
    return X_train, X_val, X_test

def generate_datasets_split_parallel(market_data):
    """
    Splits data PER SYMBOL, then generates samples for Train/Val/Test in parallel.
    """
    print(f"Generating datasets for {len(market_data)} symbols in PARALLEL...")
    
    def process_single_symbol(symbol, df):
        total_len = len(df)
        train_end = int(total_len * Config.TrainSplit)
        val_end = int(total_len * (Config.TrainSplit + Config.ValSplit))
        
        df_train = df.iloc[:train_end]
        df_val = df.iloc[train_end:val_end]
        df_test = df.iloc[val_end:]
        
        res = {
            'train': process_dataframe(df_train),
            'val': process_dataframe(df_val),
            'test': process_dataframe(df_test)
        }
        return res

    results = Parallel(n_jobs=-1)(
        delayed(process_single_symbol)(s, df) for s, df in tqdm(market_data.items(), desc="Symbols")
    )
    
    # Aggregate
    train_data = {'p': [], 'c': [], 'y': []}
    val_data = {'p': [], 'c': [], 'y': []}
    test_data = {'p': [], 'c': [], 'y': []}
    
    for res in results:
        # Train
        p, c, y = res['train']
        if len(y) > 0:
            train_data['p'].append(p)
            train_data['c'].append(c)
            train_data['y'].append(y)
            
        # Val
        p, c, y = res['val']
        if len(y) > 0:
            val_data['p'].append(p)
            val_data['c'].append(c)
            val_data['y'].append(y)
            
        # Test
        p, c, y = res['test']
        if len(y) > 0:
            test_data['p'].append(p)
            test_data['c'].append(c)
            test_data['y'].append(y)
            
    # Concatenate
    def concat_arrays(list_of_arrays):
        if not list_of_arrays:
            return np.empty(0)
        return np.concatenate(list_of_arrays, axis=0)
        
    X_train = (concat_arrays(train_data['p']), concat_arrays(train_data['c']), concat_arrays(train_data['y']))
    X_val = (concat_arrays(val_data['p']), concat_arrays(val_data['c']), concat_arrays(val_data['y']))
    X_test = (concat_arrays(test_data['p']), concat_arrays(test_data['c']), concat_arrays(test_data['y']))
    
    return X_train, X_val, X_test


# ==========================================
# 4. MODEL (Transformer)
# ==========================================
class VolumeProfileTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.bin_embedding = nn.Linear(1, config.EmbedDim)
        self.pos_encoder = nn.Parameter(torch.randn(1, config.NumVolumeBins, config.EmbedDim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.EmbedDim, nhead=config.NumHeads, dropout=config.Dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.NumLayers)
        
        self.context_projection = nn.Linear(config.ContextDim, config.EmbedDim)
        
        self.head = nn.Sequential(
            nn.Linear(config.EmbedDim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # No Sigmoid here!
        )
        
    def forward(self, profiles, context):
        x = profiles.unsqueeze(-1).float()
        x = self.bin_embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        x_pool = x.mean(dim=1)
        c = self.context_projection(context.float())
        combined = torch.cat([x_pool, c], dim=1)
        return self.head(combined)

class TradingDataset(Dataset):
    def __init__(self, X_p, X_c, y):
        self.X_p = X_p
        self.X_c = X_c
        self.y = y
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X_p[idx], self.X_c[idx], self.y[idx]

# ==========================================
# 5. TRAINING & EVALUATION
# ==========================================
def train(model, loader, optimizer, criterion, epoch, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [TRAIN]", 
                file=sys.stdout, dynamic_ncols=True)
    
    batch_count = 0
    accumulated_loss = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    for batch_idx, (profiles, context, labels) in enumerate(pbar):
        profiles = profiles.to(device)
        context = context.to(device)
        labels = labels.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(profiles, context).squeeze()
            loss = criterion(logits, labels.float())
            loss = loss / Config.GradientAccumSteps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        accumulated_loss += loss.item()
        
        # Update weights every GradientAccumSteps
        if (batch_idx + 1) % Config.GradientAccumSteps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
            total_loss += accumulated_loss
            accumulated_loss = 0
            batch_count += 1
        
        # Calculate accuracy
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        if batch_idx % Config.LogEveryN == 0:
            avg_loss = total_loss / max(batch_count, 1)
            acc = 100. * correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{acc:.2f}%',
                'batch': f'{batch_idx}/{len(loader)}'
            })
    
    # Handle any remaining accumulated gradients
    if accumulated_loss > 0:
        optimizer.step()
        optimizer.zero_grad()
        total_loss += accumulated_loss
        batch_count += 1
    
    avg_loss = total_loss / max(batch_count, 1)
    accuracy = 100. * correct / total if total > 0 else 0
    
    pbar.close()
    return avg_loss, accuracy

def evaluate(model, loader, criterion, desc="VAL", device='cpu'):
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    
    pbar = tqdm(loader, desc=f"[{desc}]", file=sys.stdout, dynamic_ncols=True)
    
    with torch.no_grad():
        for profiles, context, labels in pbar:
            profiles = profiles.to(device)
            context = context.to(device)
            labels = labels.to(device)
            
            logits = model(profiles, context).squeeze()
            loss = criterion(logits, labels.float())
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            avg_loss = total_loss / (pbar.n + 1) if hasattr(pbar, 'n') else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    pbar.close()
    return total_loss / len(loader), preds, targets

def sample_dataset(X_p, X_c, y, max_samples):
    """
    Sample dataset to reduce size while maintaining class balance.
    """
    if len(y) <= max_samples:
        print(f"Dataset size ({len(y)}) <= max_samples ({max_samples}), using full dataset.")
        return X_p, X_c, y
    
    print(f"Sampling dataset from {len(y)} to {max_samples} samples...")
    
    # Get class counts
    unique, counts = np.unique(y, return_counts=True)
    class_balance = dict(zip(unique, counts))
    
    # Sample proportionally from each class
    indices = []
    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        cls_proportion = counts[list(unique).index(cls)] / len(y)
        n_samples_cls = int(max_samples * cls_proportion)
        
        # Sample randomly
        if len(cls_indices) > n_samples_cls:
            sampled = np.random.choice(cls_indices, n_samples_cls, replace=False)
        else:
            sampled = cls_indices
        
        indices.extend(sampled)
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    
    print(f"Sampled {len(indices)} samples (original balance preserved)")
    return X_p[indices], X_c[indices], y[indices]

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("--- Starting TopTrader GPT Pipeline ---")
    
    # Set matmul precision for potential GPU utilization
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        print("Set float32 matmul precision to 'high'")

    Utils.ensure_dirs()
    
    # 1. Fetch
    symbols = fetch_top_symbols(Config.SYMBOL_LIMIT)
    for s in symbols:
        download_klines(s)
        
    # 2. Preprocess
    print("Loading and processing data...")
    import joblib
    
    train_path = os.path.join(Config.DataDir, "train.npz")
    val_path = os.path.join(Config.DataDir, "val.npz")
    test_path = os.path.join(Config.DataDir, "test.npz")
    scaler_path = os.path.join(Config.ResultsDir, "scaler.pkl")
    
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path) and os.path.exists(scaler_path):
        print("Loading cached splits...")
        train_d = np.load(train_path)
        val_d = np.load(val_path)
        test_d = np.load(test_path)
        
        X_train_p, X_train_c, y_train = train_d['p'], train_d['c'], train_d['y']
        X_val_p, X_val_c, y_val = val_d['p'], val_d['c'], val_d['y']
        X_test_p, X_test_c, y_test = test_d['p'], test_d['c'], test_d['y']
        
        scaler = joblib.load(scaler_path)
        print("Data loaded.")
    else:
        market_data = load_data()
        (X_train_p, X_train_c, y_train), (X_val_p, X_val_c, y_val), (X_test_p, X_test_c, y_test) = generate_datasets_split_parallel(market_data)
        
        print("Fitting scaler on TRAIN only...")
        scaler = StandardScaler()
        if len(X_train_c) > 0:
            X_train_c = scaler.fit_transform(X_train_c)
        
        # Apply to Val/Test
        if len(X_val_c) > 0:
            X_val_c = scaler.transform(X_val_c)
        if len(X_test_c) > 0:
            X_test_c = scaler.transform(X_test_c)
            
        print("Saving splits and scaler...")
        np.savez_compressed(train_path, p=X_train_p, c=X_train_c, y=y_train)
        np.savez_compressed(val_path, p=X_val_p, c=X_val_c, y=y_val)
        np.savez_compressed(test_path, p=X_test_p, c=X_test_c, y=y_test)
        joblib.dump(scaler, scaler_path)
        
    print(f"Train Shape: {X_train_p.shape}, Val Shape: {X_val_p.shape}, Test Shape: {X_test_p.shape}")
    
    if len(y_train) == 0:
        print("No training samples generated.")
        return
    
    # Apply sampling if enabled
    if Config.UseSampling and len(y_train) > Config.MaxSamples:
        X_train_p, X_train_c, y_train = sample_dataset(X_train_p, X_train_c, y_train, Config.MaxSamples)
        print(f"After sampling - Train Shape: {X_train_p.shape}")
    
    # Check Balance (on Train)
    unique, counts = np.unique(y_train, return_counts=True)
    balance = dict(zip(unique, counts))
    print(f"Train Class Balance: {balance}")
    
    pos_count = balance.get(1, 0)
    neg_count = balance.get(0, 0)
    
    # Simple heuristic to avoid zero
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Positive Weight: {pos_weight_val:.2f}")

    # Create Datasets
    train_data = TradingDataset(X_train_p, X_train_c, y_train)
    val_data = TradingDataset(X_val_p, X_val_c, y_val)
    test_data = TradingDataset(X_test_p, X_test_c, y_test)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=Config.BatchSize, 
        shuffle=True, 
        num_workers=Config.NumWorkers, 
        pin_memory=torch.cuda.is_available(),
        persistent_workers=Config.PersistentWorkers if Config.NumWorkers > 0 else False
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=Config.BatchSize, 
        num_workers=Config.NumWorkers, 
        pin_memory=torch.cuda.is_available(),
        persistent_workers=Config.PersistentWorkers if Config.NumWorkers > 0 else False
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=Config.BatchSize, 
        num_workers=Config.NumWorkers, 
        pin_memory=torch.cuda.is_available(),
        persistent_workers=Config.PersistentWorkers if Config.NumWorkers > 0 else False
    )
    
    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Model
    model = VolumeProfileTransformer(Config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LearningRate)
    
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Check for checkpoint
    checkpoint_path = os.path.join(Config.ResultsDir, "checkpoint.pth")
    start_epoch = 0
    
    if Config.ResumeFromCheckpoint and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # 4. Train with Early Stopping
    print(f"\n--- Starting Training ({Config.Epochs} epochs, Batch Size {Config.BatchSize}) ---")
    print(f"Total batches per epoch: {len(train_loader)}")
    print(f"Estimated time per epoch: ~{len(train_loader) / 20 / 60:.1f} minutes (at 20 batches/sec)")
    
    best_loss = float('inf')
    patience = 5
    counter = 0
    
    for epoch in range(start_epoch, Config.Epochs):
        t_loss, t_acc = train(model, train_loader, optimizer, criterion, epoch, device)
        v_loss, _, _ = evaluate(model, val_loader, criterion, "VAL", device)
        
        print(f"\nEpoch {epoch+1}/{Config.Epochs} Summary:")
        print(f"  Train Loss: {t_loss:.4f} | Train Acc: {t_acc:.2f}%")
        print(f"  Val Loss: {v_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': t_loss,
            'val_loss': v_loss,
        }
        torch.save(checkpoint, checkpoint_path)
        
        if v_loss < best_loss:
            best_loss = v_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(Config.ResultsDir, "best_model.pth"))
            print(f"  → New best model saved!")
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break
                
    # 5. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(os.path.join(Config.ResultsDir, "best_model.pth")))
    _, preds, targets = evaluate(model, test_loader, criterion, "TEST", device)
    
    print(f"Max Pred Prob: {max(preds):.4f}")
    
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    
    print("\n--- Evaluation on Test Set ---")
    print(classification_report(targets, preds_binary))
    
    # Save Metrics Plot (Calibration)
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(targets, preds, n_bins=10)
    
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Win Rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(os.path.join(Config.ResultsDir, "calibration.png"))
    print("Calibration curve saved.")

    # Save artifacts
    print("Saving scaler...")
    import joblib
    joblib.dump(scaler, os.path.join(Config.ResultsDir, "scaler.pkl"))
    
    print("Pipeline Complete.")

if __name__ == "__main__":
    main()
