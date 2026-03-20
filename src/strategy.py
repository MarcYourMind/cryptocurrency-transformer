import os
import torch
import numpy as np
import pandas as pd
import joblib
from .config import Config
from .train_gpt import VolumeProfileTransformer, is_sideways, compute_volume_profile, get_context_features

class Strategy:
    def __init__(self, model_path="results/best_model.pth", scaler_path="results/scaler.pkl"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Model
        self.model = VolumeProfileTransformer(Config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model not found at {model_path}. Please train the model first.")
        self.model.eval()
        
        # Load Scaler
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Please train the model first.")
            self.scaler = None

    def get_signal(self, df, threshold=0.5):
        """
        Processes latest klines and returns a signals if a trade opportunity is found.
        Returns: (entry_price, sl_price, tp_price, prob) or None
        """
        if len(df) < Config.LookbackWindow:
            return None
            
        # Get the latest window
        window = df.iloc[-Config.LookbackWindow:]
        
        # 1. Regime Detection
        if not is_sideways(window):
            return None
            
        # 2. Extract Features
        try:
            prof = compute_volume_profile(window, Config.NumVolumeBins)
            base_ctx = get_context_features(window)
        except Exception as e:
            print(f"Error computing features: {e}")
            return None
            
        atr = base_ctx[0]
        low = window['low'].min()
        high = window['high'].max()
        price_range = high - low
        if price_range <= 0 or atr <= 0:
            return None
            
        bin_width = price_range / Config.NumVolumeBins
        
        # 3. Model Inference for all 64 entry points
        all_batch_prof = []
        all_batch_ctx = []
        
        for bin_idx in range(Config.NumVolumeBins):
            ctx_item = base_ctx + [bin_idx / Config.NumVolumeBins]
            all_batch_prof.append(prof)
            all_batch_ctx.append(ctx_item)
            
        if self.scaler:
            try:
                # Scaler expects 2D array
                scaled_ctx = self.scaler.transform(np.array(all_batch_ctx))
            except Exception as e:
                print(f"Scaling error: {e}")
                return None
        else:
            scaled_ctx = np.array(all_batch_ctx)
            
        with torch.no_grad():
            t_prof = torch.tensor(np.array(all_batch_prof), dtype=torch.float32).to(self.device)
            t_ctx = torch.tensor(scaled_ctx, dtype=torch.float32).to(self.device)
            
            logits = self.model(t_prof, t_ctx)
            probs = torch.sigmoid(logits).squeeze().tolist()
            
        # 4. Find Best Candidate
        candidates = []
        for bin_idx, prob in enumerate(probs):
            entry_price = low + (bin_idx + 0.5) * bin_width
            candidates.append({
                'prob': prob,
                'price': entry_price
            })
            
        candidates.sort(key=lambda x: x['prob'], reverse=True)
        best = candidates[0]
        
        if best['prob'] >= threshold:
            e_price = best['price']
            sl_price = e_price - (atr * Config.StopLossMultiplierATR)
            tp_price = e_price + (atr * Config.StopLossMultiplierATR * Config.RiskRewardRatio)
            
            return {
                'entry_price': round(e_price, 4),
                'sl_price': round(sl_price, 4),
                'tp_price': round(tp_price, 4),
                'prob': round(best['prob'], 4)
            }
            
        return None
