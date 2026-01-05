import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
import joblib
import os
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

def prepare_data(tickers):
    loader = MarketDataLoader()
    strategy = SmartWhaleStrategy() 
    
    # Pre-fetch macro
    print("🌍 Fetching Global Macro Data...")
    macro_df = loader.fetch_macro_data("2020-01-01", "2024-12-31")
    
    all_data = []
    print(f"🔄 Fetching data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            df = loader.fetch_data(ticker)
            if df.empty: continue
            
            # Merge Macro
            df = df.join(macro_df, how='left').ffill().dropna()
            
            # Indicators (Base + Advanced)
            # Smart Sum calculated here? No, fetch_data doesn't do it.
            # Use strategy logic.
            df = strategy.add_indicators(df)
            
            # --- Feature Engineering (Deep) ---
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
            df['Dist_MA60'] = df['Close'] / df['MA60'] - 1
            df['Vol_Roll'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Tech
            if 'RSI' in df.columns: df['RSI_Norm'] = df['RSI'] / 100.0
            if 'Stoch_K' in df.columns: df['Stoch_K_Norm'] = df['Stoch_K'] / 100.0
            df['BB_Width'] = df['BB_Width']
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Smart Z
            roll_mean = df['Smart_Sum_20'].rolling(60).mean()
            roll_std = df['Smart_Sum_20'].rolling(60).std()
            df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
            
            # Macro Changes
            if 'CL=F' in df.columns:
                df['Oil_Chg'] = df['CL=F'].pct_change()
                df['Gold_Chg'] = df['GC=F'].pct_change()
                df['Bond_Yield'] = df['^TNX'] / 100.0
                df['USD_KRW'] = df['KRW=X'].pct_change()
            else:
                df['Oil_Chg'] = 0; df['Gold_Chg'] = 0; df['Bond_Yield'] = 0; df['USD_KRW'] = 0
                
            # OBV Change
            df['OBV_Chg'] = df['OBV'].pct_change()
            
            # Feature List
            features = [
                'Log_Ret', 'Dist_MA20', 'Dist_MA60', 'Vol_Roll', 
                'RSI_Norm', 'Stoch_K_Norm', 'BB_Width', 'MACD_Hist', 'OBV_Chg',
                'Smart_Z', 
                'Oil_Chg', 'Gold_Chg', 'Bond_Yield', 'USD_KRW'
            ]
            
            # Create Target
            # 5-day return > 3%
            future_ret = df['Close'].shift(-5) / df['Close'] - 1
            df['Target'] = np.where(future_ret > 0.03, 1, 0)
            
            df_final = df[features + ['Target']].replace([np.inf, -np.inf], np.nan).dropna()
            all_data.append(df_final)
            print(f"✅ {ticker}: {len(df_final)} rows")
            
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data)

def train_refined():
    tickers = [
        "005930.KS", "000660.KS", "042700.KS", "005830.KS", # Semi
        "373220.KS", "006400.KS", "051910.KS", "003670.KQ", "247540.KQ", "086520.KQ", # Battery
        "005380.KS", "000270.KS", "012330.KS", # Auto
        "207940.KS", "068270.KS", "028300.KQ", "196170.KQ", "087010.KQ", # Bio
        "035420.KS", "035720.KS", "352820.KS", "251270.KQ", # Platform
        "012450.KS", "047810.KS", "010120.KS", "034020.KS" # Industry
    ]
    
    print("🚀 [Step 1] Loading Deep Data (Macro + Advanced Tech)...")
    df = prepare_data(tickers)
    
    if df.empty: return

    features = [c for c in df.columns if c != 'Target']
    X = df[features]
    y = df['Target']
    
    neg, pos = np.bincount(y)
    scale = neg / pos
    print(f"📊 Class Ratio: Neg {neg} / Pos {pos} (Weight: {scale:.2f})")
    
    # XGBoost V3
    print("🧠 [Step 2] Training XGBoost V3 (Deep Features)...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale * 1.2, # Slightly aggressive
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False) # early_stopping_rounds removed (deprecated in some versions or handled differently)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"🏆 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature Importance
    print("\n🔑 Final Feature Importance:")
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    print(fi.sort_values(by='Importance', ascending=False))
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/xgb_whale_v3_deep.pkl')
    print("💾 Model saved to models/xgb_whale_v3_deep.pkl")

if __name__ == "__main__":
    train_refined()
