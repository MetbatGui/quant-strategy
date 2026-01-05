import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

def resample_weekly(df):
    """
    Generate Weekly Indicators and merge back to Daily (FFILL)
    """
    logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    w_df = df.resample('W').agg(logic)
    
    # Weekly RSI
    delta = w_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    w_df['Weekly_RSI'] = 100 - (100 / (1 + gain/loss))
    
    # Weekly MACD
    exp12 = w_df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = w_df['Close'].ewm(span=26, adjust=False).mean()
    w_df['Weekly_MACD'] = exp12 - exp26
    
    # Weekly Trend (SMA 20)
    w_df['Weekly_MA20'] = w_df['Close'].rolling(20).mean()
    w_df['Weekly_Trend'] = np.where(w_df['Close'] > w_df['Weekly_MA20'], 1, 0)
    
    # Select cols
    w_feat = w_df[['Weekly_RSI', 'Weekly_MACD', 'Weekly_Trend']]
    
    return w_feat

def prepare_data(tickers):
    loader = MarketDataLoader()
    strategy = SmartWhaleStrategy()
    
    print("🌍 Fetching Macro...")
    macro_df = loader.fetch_macro_data("2020-01-01", "2024-12-31")
    
    all_data = []
    print(f"🔄 Processing {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            df = loader.fetch_data(ticker)
            if df.empty: continue
            
            # 1. Weekly Features
            w_feat = resample_weekly(df)
            
            # Merge Weekly back to Daily (FFILL to avoid lookahead? No, FFILL propagates known weekly value forward)
            # Safe way: Shift weekly by 1 week? 
            # Resample puts date at END of week (Sunday).
            # If we FFILL, Monday uses Sunday's value. 
            # But Sunday's value assumes Friday close.
            # So on Monday morning, we KNOW last week's close. This is valid.
            df = df.join(w_feat.reindex(df.index, method='ffill'))
            
            # 2. Merge Macro
            df = df.join(macro_df, how='left').ffill().dropna()
            
            # 3. Smart Indicators
            df = strategy.add_indicators(df)
            
            # 4. Feature Engineering (The Ultimate Set)
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
            df['Vol_Roll'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Default Indicators
            df['RSI_Norm'] = df['RSI'] / 100.0
            df['Stoch_K_Norm'] = df['Stoch_K'] / 100.0
            df['BB_Width'] = df['BB_Width']
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # New Indicators (Ichimoku, CCI, Williams)
            # Ichimoku: Close > SpanA? Close > SpanB?
            # Interaction Feature: Cloud Breakout
            df['Ichi_Bull'] = np.where((df['Close'] > df['Ichimoku_SpanA']) & (df['Close'] > df['Ichimoku_SpanB']), 1, 0)
            df['CCI_Norm'] = df['CCI'] / 100.0 # Standardize somewhat
            df['WilliamsR_Norm'] = (df['WilliamsR'] + 50) / 50.0 # -100~0 -> -1~1
            
            # Smart Money
            roll_mean = df['Smart_Sum_20'].rolling(60).mean()
            roll_std = df['Smart_Sum_20'].rolling(60).std()
            df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
            
            # Macro
            if 'CL=F' in df.columns:
                df['Oil_Chg'] = df['CL=F'].pct_change()
                df['Bond_Yield'] = df['^TNX'] / 100.0
                df['USD_KRW'] = df['KRW=X'].pct_change()
            else:
                df['Oil_Chg'] = 0; df['Bond_Yield'] = 0; df['USD_KRW'] = 0
            
            # OBV
            df['OBV_Chg'] = df['OBV'].pct_change()
            
            # Weekly Features (Normalized)
            df['Weekly_RSI_Norm'] = df['Weekly_RSI'] / 100.0
            df['Weekly_MACD'] = df['Weekly_MACD']
            df['Weekly_Trend'] = df['Weekly_Trend']
            
            # Feature List (20 Features)
            cols = [
                'Log_Ret', 'Dist_MA20', 'Vol_Roll', 
                'RSI_Norm', 'Stoch_K_Norm', 'BB_Width', 'MACD_Hist',
                'Ichi_Bull', 'CCI_Norm', 'WilliamsR_Norm',
                'OBV_Chg', 'Smart_Z',
                'Oil_Chg', 'Bond_Yield', 'USD_KRW',
                'Weekly_RSI_Norm', 'Weekly_MACD', 'Weekly_Trend'
            ]
            
            # Target (5 days, > 3%)
            future_ret = df['Close'].shift(-5) / df['Close'] - 1
            df['Target'] = np.where(future_ret > 0.03, 1, 0)
            
            df_final = df[cols + ['Target']].replace([np.inf, -np.inf], np.nan).dropna()
            all_data.append(df_final)
            print(f"✅ {ticker}: {len(df_final)}")
            
        except Exception as e:
            print(f"Error {ticker}: {e}")
            import traceback; traceback.print_exc() 
            
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data)

def train_ensemble():
    tickers = [
        "005930.KS", "000660.KS", "042700.KS", "373220.KS", "006400.KS", "051910.KS", 
        "005380.KS", "000270.KS", "207940.KS", "068270.KS", "035420.KS", "035720.KS", 
        "012450.KS", "010120.KS"
    ] # Selected representative tickers
    
    print("🚀 Loading Ultimate Data Set...")
    df = prepare_data(tickers)
    
    if df.empty:
        print("No data.")
        return

    features = [c for c in df.columns if c != 'Target']
    X = df[features]
    y = df['Target']
    
    neg, pos = np.bincount(y)
    scale = neg / pos
    print(f"Class Weight: {scale:.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 1. XGBoost (The Brain)
    clf1 = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, scale_pos_weight=scale, random_state=42)
    
    # 2. Random Forest (The Consensus)
    clf2 = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight='balanced', random_state=42)
    
    # 3. Gradient Boosting (The Specialist)
    clf3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    
    # Ensemble (Soft Voting)
    eclf = VotingClassifier(estimators=[('xgb', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
    
    print("🧠 Training Ensemble Model (XGB + RF + GB)...")
    eclf.fit(X_train, y_train)
    
    y_pred = eclf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"🏆 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(eclf, 'models/ensemble_whale_v4.pkl')
    print("💾 Saved models/ensemble_whale_v4.pkl")

if __name__ == "__main__":
    train_ensemble()
