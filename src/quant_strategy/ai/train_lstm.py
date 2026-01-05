import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

# === Configuration ===
WINDOW_SIZE = 20    # 20일치 데이터를 보고 판단
PREDICT_DAYS = 5    # 5일 뒤 예측
TARGET_RETURN = 0.05 # 5% 수익 목표 (Aggressive)
BATCH_SIZE = 32
EPOCHS = 100        # Deep Learning needs epochs
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

def create_sequences(data, seq_length, target_col_idx):
    xs, ys = [], []
    for i in range(len(data) - seq_length - PREDICT_DAYS):
        # X: All features except the last column (Target)
        x = data[i:(i + seq_length), :-1] 
        # Target: Future Return check
        y = data[i + seq_length, -1] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def prepare_data(tickers):
    loader = MarketDataLoader()
    strategy = SmartWhaleStrategy()
    
    # 1. Fetch Macro Data globally
    print("🌍 Fetching Macro Data (Oil, Gold, Yields)...")
    macro_df = loader.fetch_macro_data("2020-01-01", "2024-12-31")
    
    all_X = []
    all_y = []
    
    scaler = StandardScaler()
    
    # We train on normalized features.
    # But scaling must be done per stock or globally? 
    # Global scaling for macro, Local scaling for price-based.
    # To simplify: calculate percent changes or relative ratios for everything.
    
    print(f"🔄 Processing {len(tickers)} stocks for LSTM...")
    
    for ticker in tickers:
        try:
            df = loader.fetch_data(ticker)
            if df.empty: continue
            
            # Merge Macro
            df = df.join(macro_df, how='left').ffill().dropna()
            
            # Use Strategy logic to calculate Smart_Sum_20, etc.
            df = strategy.add_indicators(df)
            
            # --- Feature Engineering (Deep) ---
            # Instead of raw prices, we use Ratios & Changes (Stationary)
            
            # 1. Price Features
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
            df['Dist_MA60'] = df['Close'] / df['MA60'] - 1
            df['Vol_Roll'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # 2. Tech Features
            df['RSI'] = df['RSI'] / 100.0 # 0~1 Scale
            df['Stoch_K'] = df['Stoch_K'] / 100.0
            df['BB_Width'] = df['BB_Width']
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # 3. Smart Money
            # Normalize Smart Sum: Z-Score with 60d window
            roll_mean = df['Smart_Sum_20'].rolling(60).mean()
            roll_std = df['Smart_Sum_20'].rolling(60).std()
            df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
            
            # 4. Macro Features (Change rates)
            # CL=F, GC=F, ^TNX, ^NDX, KRW=X
            if 'CL=F' in df.columns:
                df['Oil_Chg'] = df['CL=F'].pct_change()
                df['Gold_Chg'] = df['GC=F'].pct_change()
                df['Bond_Yield'] = df['^TNX'] / 100.0 # Already %
                df['USD_KRW'] = df['KRW=X'].pct_change()
            else:
                # Fallback if macro fetch failed
                df['Oil_Chg'] = 0
                df['Gold_Chg'] = 0
                df['Bond_Yield'] = 0
                df['USD_KRW'] = 0

            # 5. Target Creation (Must be done BEFORE dropping NaNs)
            # Future 5-day Return
            future_ret = df['Close'].shift(-PREDICT_DAYS) / df['Close'] - 1
            df['Target'] = np.where(future_ret > TARGET_RETURN, 1.0, 0.0)
            
            # Drop NaNs created by rolling/diff
            feature_cols = [
                'Log_Ret', 'Dist_MA20', 'Dist_MA60', 'Vol_Roll', 
                'RSI', 'Stoch_K', 'BB_Width', 'MACD_Hist', 'OBV',
                'Smart_Z', 
                'Oil_Chg', 'Gold_Chg', 'Bond_Yield', 'USD_KRW'
            ]
            
            # OBV needs normalization (It's cumulative) -> OBV Change 
            df['OBV_Chg'] = df['OBV'].pct_change()
            feature_cols = [f for f in feature_cols if f != 'OBV'] + ['OBV_Chg']
            
            data_df = df[feature_cols + ['Target']].dropna()
            
            if len(data_df) < WINDOW_SIZE: continue
            
            # Convert to numpy
            data_np = data_df.values
            
            # Normalize Features (Fit on this stock's history? No, better robust scaler globally? 
            # Or Z-Score per stock. Let's do Z-Score per stock to handle price diffs.)
            # Target (last col) should not be scaled
            features = data_np[:, :-1]
            targets = data_np[:, -1]
            
            # Quick standardization per stock
            feat_mean = np.mean(features, axis=0)
            feat_std = np.std(features, axis=0) + 1e-9
            features_norm = (features - feat_mean) / feat_std
            
            # Concat back
            data_norm = np.column_stack([features_norm, targets])
            
            # Create Sequences
            X, y = create_sequences(data_norm, WINDOW_SIZE, -1)
            
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                
            print(f"✅ {ticker}: {len(X)} seq")
            
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
            import traceback
            traceback.print_exc()

    if not all_X: return None, None, None

    # Merge all stocks
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    return X_all, y_all, len(feature_cols)

def train_lstm():
    tickers = [
        "005930.KS", "000660.KS", "042700.KS", "005830.KS", # Semi
        "373220.KS", "006400.KS", "051910.KS", "003670.KQ", "247540.KQ", "086520.KQ", # Battery
        "005380.KS", "000270.KS", "012330.KS", # Auto
        "207940.KS", "068270.KS", "028300.KQ", "196170.KQ", "087010.KQ", # Bio
        "035420.KS", "035720.KS", "352820.KS", "251270.KQ", # Platform
        "012450.KS", "047810.KS", "010120.KS", "034020.KS" # Industry
    ]
    
    print("🚀 [Phase 1] Data Preparation (Deep Learning Mode)")
    X, y, input_dim = prepare_data(tickers)
    
    if X is None:
        print("❌ Data prep failed.")
        return

    print(f"📊 Dataset Shape: X={X.shape}, y={y.shape}")
    print(f"🧩 Input Features: {input_dim}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    train_dataset = StockDataset(X_train, y_train)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model Setup
    model = LSTMModel(input_size=input_dim).to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"🧠 [Phase 2] Training LSTM on {DEVICE}...")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == batch_y).sum().item()
        
        val_avg_loss = val_loss / len(test_loader)
        accuracy = correct / len(y_test)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}, Acc: {accuracy:.4f}")
            
        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            # Save Checkpoint
            torch.save(model.state_dict(), 'models/lstm_whale_v1.pth')
            
    print("💾 Training Complete. Best Model Saved.")

if __name__ == "__main__":
    train_lstm()
