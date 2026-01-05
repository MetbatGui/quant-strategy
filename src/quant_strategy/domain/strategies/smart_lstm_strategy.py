import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy
from quant_strategy.infrastructure.data_loader import MarketDataLoader

# LSTM Model Definition (Must match training script)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

class SmartLstmStrategy(SmartWhaleStrategy):
    """
    [Smart LSTM Strategy] (Deep Learning)
    - "시간의 흐름을 읽는 AI"
    - Input: 20일치 시계열 데이터 (가격, 거래량, 수급, 거시경제 지표)
    - Output: 상승 확률 (0~1)
    """
    
    def __init__(self, window=60, model_path='models/lstm_whale_v1.pth'):
        super().__init__(window)
        self.model = None
        self.device = torch.device('cpu') # Inference on CPU is fast enough
        self.seq_len = 20
        self.input_dim = 14 # Must match training
        
        # Load Model
        if os.path.exists(model_path):
            try:
                self.model = LSTMModel(input_size=self.input_dim).to(self.device)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"🧠 LSTM Neural Network Loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ LSTM Model Load Failed: {e}")
        else:
            print(f"⚠️ LSTM Model not found at {model_path}")

        # Macro Data Cache (Updated daily in real system, here loaded once)
        loader = MarketDataLoader()
        self.macro_df = loader.fetch_macro_data("2020-01-01", "2024-12-31")

    def add_indicators(self, df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        # Base Indicators
        df = super().add_indicators(df, macro_df)
        
        # LSTM Features (Must match training logic exactly)
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
        df['Dist_MA60'] = df['Close'] / df['MA60'] - 1
        df['Vol_Roll'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Normalized Techs
        if 'RSI' in df.columns: df['RSI_Norm'] = df['RSI'] / 100.0
        else: df['RSI_Norm'] = 0.5
            
        if 'Stoch_K' in df.columns: df['Stoch_K_Norm'] = df['Stoch_K'] / 100.0
        else: df['Stoch_K_Norm'] = 0.5
            
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Smart Z
        roll_mean = df['Smart_Sum_20'].rolling(60).mean()
        roll_std = df['Smart_Sum_20'].rolling(60).std()
        df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
        
        # Macro Merge (Join by Date Index)
        # Assuming df has Datetime Index
        # We need to map macro data to df index
        # Simple implementation: reindex macro to df
        
        # Note: add_indicators is called per ticker. We need to be careful with index alignment.
        # But for backtest, we can't join macro easily here without complex lookups.
        # Alternatively, assume macro columns already added or do crude ffill.
        
        # Simplification for Inference:
        # We are only predicting NEXT STEP, so we usually have the latest macro.
        # For backtest, we need historical macro.
        
        # Let's try to join efficiently
        if not self.macro_df.empty:
            # Reindex macro to match df (FFILL)
            # Ensure TZ awareness matches
            if df.index.tz is not None and self.macro_df.index.tz is None:
                 self.macro_df.index = self.macro_df.index.tz_localize(df.index.tz)
            elif df.index.tz is None and self.macro_df.index.tz is not None:
                 self.macro_df.index = self.macro_df.index.tz_localize(None)

            # Join
            # Using reindex with method='ffill' is safer than join for potentially missing dates
            macro_reindexed = self.macro_df.reindex(df.index, method='ffill')
            df['Oil'] = macro_reindexed['CL=F']
            df['Gold'] = macro_reindexed['GC=F']
            df['Bond'] = macro_reindexed['^TNX']
            df['KRW'] = macro_reindexed['KRW=X']
            
            df['Oil_Chg'] = df['Oil'].pct_change().fillna(0)
            df['Gold_Chg'] = df['Gold'].pct_change().fillna(0)
            df['Bond_Yield'] = df['Bond'] / 100.0
            df['USD_KRW'] = df['KRW'].pct_change().fillna(0)
        else:
            df['Oil_Chg'] = 0
            df['Gold_Chg'] = 0
            df['Bond_Yield'] = 0
            df['USD_KRW'] = 0
            
        # OBV Chg
        df['OBV_Chg'] = df['OBV'].pct_change().fillna(0)

        # Remove infs
        df = df.replace([np.inf, -np.inf], 0)
        
        # Run AI Inference
        df = self.run_inference_batch(df)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool, entry_price: float = 0) -> str:
        # 1. Whale Signal
        signal = super().check_signals(curr_row, prev_row, has_position, entry_price)
        
        # 2. LSTM Filter
        if signal == 'BUY' and self.model is not None:
            # Prepare Sequence (Last 20 days)
            # We need access to history... 
            # `check_signals` only gives current row.
            # This is a limitation of current architecture.
            # However, `curr_row` is a Series from the DataFrame.
            # We can pass the full DF or access it?
            # Actually, `engine.py` iterates rows.
            # To fix this, we need `BacktestService` to pass `history`.
            # BUT, changing Engine is risky.
            
            # Alternative: Assume `add_indicators` pre-calculated the LSTM Probability!
            # YES. Calculate `AI_Prob` for ALL rows in `add_indicators` as a vectorized operation (or rolling apply).
            # This is much faster than running LSTM row-by-row in Python loop.
            
            # Check AI Prob (with lower threshold)
            prob = curr_row.get('AI_Prob', 0.5)
            
            # Debug: rare print
            # if np.random.rand() < 0.001: print(f"AI Prob: {prob:.4f}")

            if prob > 0.35: # Lower threshold to 0.35
                return 'BUY'
            else:
                return 'HOLD'
                
        return signal

    def run_inference_batch(self, df):
        """
        Runs LSTM on the entire DataFrame to generate 'AI_Prob' column.
        """
        if self.model is None or len(df) < self.seq_len:
            df['AI_Prob'] = 0.5
            return df
            
        # Prepare Feature Tensor
        feature_cols = [
            'Log_Ret', 'Dist_MA20', 'Dist_MA60', 'Vol_Roll', 
            'RSI_Norm', 'Stoch_K_Norm', 'BB_Width', 'MACD_Hist', 
            'Smart_Z', 
            'Oil_Chg', 'Gold_Chg', 'Bond_Yield', 'USD_KRW',
            'OBV_Chg'
        ]
        
        data_np = df[feature_cols].values
        
        # Normalize (Simple local standardization)
        # Note: In training we did global or per-stock. Here we do per-stock (since df is one stock).
        mean = np.mean(data_np, axis=0)
        std = np.std(data_np, axis=0) + 1e-9
        data_norm = (data_np - mean) / std
        
        # Sliding Window 
        # (N, Features) -> (N-Seq+1, Seq, Features)
        # Using stride_tricks for efficiency
        shape = (len(data_norm) - self.seq_len + 1, self.seq_len, self.input_dim)
        strides = (data_norm.strides[0], data_norm.strides[0], data_norm.strides[1])
        X = np.lib.stride_tricks.as_strided(data_norm, shape=shape, strides=strides)
        
        # Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Inference (Batch)
        ai_probs = []
        with torch.no_grad():
            # Process in chunks if too large
            preds = self.model(X_tensor)
            ai_probs = preds.cpu().numpy().flatten()
            
        # Pad initial rows with 0.5
        pad = np.full(self.seq_len - 1, 0.5)
        df['AI_Prob'] = np.concatenate([pad, ai_probs])
        
        return df

    # Override run from Engine? No, engine calls add_indicators usually via strategy.
    # But add_indicators is modifying DF.
    # We can perform inference INSIDE add_indicators.
    
    def add_indicators_and_inference(self, df):
        df = self.add_indicators(df) # Calculate features
        df = self.run_inference_batch(df) # Calculate Prob
        return df
