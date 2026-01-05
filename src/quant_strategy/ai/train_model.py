import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

def prepare_data(tickers):
    loader = MarketDataLoader()
    # Strategy class used just to calculate indicators easily
    strategy = SmartWhaleStrategy() 
    
    all_data = []
    
    print(f"🔄 Fetching data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            # 1. Load Data
            df = loader.fetch_data(ticker)
            if df.empty: continue
            
            # 2. Add Indicators (Using SmartWhale logic)
            # This gives us: MA5, MA20, MA60, Smart_Sum_20, ATR, Macro_Bull...
            df = strategy.add_indicators(df)
            
            # 3. Create Additional Features for AI
            # AI needs normalized/relative values, not raw prices.
            
            # (1) 이격도 (Disparity): Price vs MA
            df['Dist_MA20'] = df['Close'] / df['MA20'] - 1
            df['Dist_MA60'] = df['Close'] / df['MA60'] - 1
            
            # (2) 수급 강도 (Smart Money Strength)
            # Smart_Sum_20은 절대값이므로, 시가총액/거래 대금 대비 비율이 좋지만
            # 간단하게 '변화율'이나 '부호' 위주로 사용.
            # 여기서는 Normalized된 값을 만들기 위해 z-score 비슷하게 변환하거나 binary로 씀.
            # 일단, Smart_Sum을 그대로 쓰면 종목별 스케일이 다르므로 -> "0보다 크냐 작냐" (Binary) + "변화율"
            df['Smart_Signal'] = np.where(df['Smart_Sum_20'] > 0, 1, 0)
            
            # (3) 변동성 (Volatility)
            df['Vol_ATR'] = df['ATR'] / df['Close']
            
            # 4. Create Target (Y)
            # 목표: "향후 5일 뒤 수익률이 +3% 이상인가?" (Binary Classification)
            df['Future_Return'] = df['Close'].shift(-5) / df['Close'] - 1
            df['Target'] = np.where(df['Future_Return'] > 0.03, 1, 0) # 3% 수익 목표
            
            # 5. Clean up (Drop NaNs mainly due to rolling & shift)
            df = df.dropna()
            
            all_data.append(df)
            print(f"✅ {ticker}: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data)

def train():
    # 1. Define Universe (Training Data)
    tickers = [
        "005930.KS", "000660.KS", "042700.KS", "005830.KS", # Semi
        "373220.KS", "006400.KS", "051910.KS", "003670.KQ", "247540.KQ", "086520.KQ", # Battery
        "005380.KS", "000270.KS", # Auto
        "207940.KS", "068270.KS", "028300.KQ", "196170.KQ", "087010.KQ", # Bio
        "035420.KS", "035720.KS", # Platform
        "012450.KS", "047810.KS" # Defense
    ]
    
    print("🚀 [Step 1] Loading Data & Engineering Features...")
    df = prepare_data(tickers)
    
    if df.empty:
        print("❌ No data loaded.")
        return

    # 2. Select Features (X) & Target (Y)
    features = [
        'Dist_MA20', 'Dist_MA60', # 기술적 위치
        'Smart_Signal', 'Smart_Sum_20', # 수급 (세력)
        'Vol_ATR', # 변동성
        'Macro_Bull' # 시장 상황
    ]
    
    X = df[features]
    y = df['Target']
    
    print(f"📊 Total Samples: {len(X)}")
    print(f"🎯 Target Distribution: {y.value_counts(normalize=True)}") # 클래스 불균형 확인
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # 시계열 데이터이므로 shuffle=False가 좋지만, 여러 종목이 섞여있어서 
    # 엄밀하게 하려면 종목별로 시간 순으로 잘라야 함. (일단 단순하게 갑니다)
    
    # 4. Train XGBoost
    print("🧠 [Step 2] Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("📝 [Step 3] Evaluation")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"🏆 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Feature Importance
    print("\n🔑 Feature Importance:")
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
    print(fi.sort_values(by='Importance', ascending=False))
    
    # 6. Save Model
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, 'models/xgb_whale_v1.pkl')
    print("💾 Model saved to models/xgb_whale_v1.pkl")

if __name__ == "__main__":
    train()
