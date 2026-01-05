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
    
    all_data = []
    print(f"🔄 Fetching data for {len(tickers)} stocks...")
    
    for ticker in tickers:
        try:
            df = loader.fetch_data(ticker)
            if df.empty: continue
            
            # --- Feature Engineering ---
            df = strategy.add_indicators(df)
            
            # 1. 기술적 지표 (Technicals)
            df['Dist_MA20'] = df['Close'] / df['MA20'] - 1  # 이격도
            df['Dist_MA60'] = df['Close'] / df['MA60'] - 1
            df['Vol_ATR'] = df['ATR'] / df['Close']         # 변동성
            
            # 2. 수급 지표 (Smart Money)
            # 수급의 '변화량'과 '강도'를 정규화
            # 20일 평균 거래량 대비 당일 거래량
            df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            # 시가총액 대비 수급 유입 (근사치로 Volume * Close 사용)
            # Smart_Sum_20 그대로 쓰면 종목간 스케일 차이 심함 -> z-score 변환
            roll_mean = df['Smart_Sum_20'].rolling(60).mean()
            roll_std = df['Smart_Sum_20'].rolling(60).std()
            df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
            
            # 3. 모멘텀 (Momentum)
            # RSI는 이미 있음.
            df['RSI_Change'] = df['RSI'].diff(5) # 5일간 RSI 변화
            
            # --- Target Engineering ---
            # 목표: 5일 뒤 수익률 > 3% (기존) -> 5%로 상향 (확실한 것만 잡기 위해)
            # 대신 0/1 비율 불균형은 scale_pos_weight로 해결
            future_ret = df['Close'].shift(-5) / df['Close'] - 1
            df['Target'] = np.where(future_ret > 0.05, 1, 0)
            
            df = df.dropna()
            all_data.append(df)
            print(f"✅ {ticker}: {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error {ticker}: {e}")
            
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data)

def train_intensive():
    tickers = [
        "005930.KS", "000660.KS", "042700.KS", "005830.KS", # Semi
        "373220.KS", "006400.KS", "051910.KS", "003670.KQ", "247540.KQ", "086520.KQ", # Battery
        "005380.KS", "000270.KS", # Auto
        "207940.KS", "068270.KS", "028300.KQ", "196170.KQ", "087010.KQ", # Bio
        "035420.KS", "035720.KS", # Platform
        "012450.KS", "047810.KS" # Defense
    ]
    
    print("🚀 [Step 1] Loading Massive Data...")
    df = prepare_data(tickers)
    
    if df.empty: return

    features = [
        'Dist_MA20', 'Dist_MA60', 'Vol_ATR',
        'Vol_Ratio', 'Smart_Z', 'RSI', 'RSI_Change',
        'Macro_Bull'
    ]
    
    X = df[features]
    y = df['Target']
    
    # 클래스 비율 확인 (scale_pos_weight 계산용)
    neg, pos = np.bincount(y)
    scale_weight = neg / pos
    print(f"📊 Class Ratio - Neg: {neg}, Pos: {pos}, Recommended Weight: {scale_weight:.2f}")

    # TimeSeriesSplit (시계열 교차 검증)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Grid Search Parameters
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300, 500],
        'scale_pos_weight': [scale_weight, scale_weight * 1.5], # Recall을 높이기 위해 가중치 부여
        'gamma': [0, 0.1, 0.2], # 과적합 방지
        'subsample': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        tree_method='hist' # 속도 최적화
    )
    
    print("🧠 [Step 2] Grid Search (Deep Thinking)... This may take a while.")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1', # Accuracy가 아니라 F1 score(조화평균)를 최적화
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"🏆 Best Params: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Save Best Model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_model, 'models/xgb_whale_v2_optimized.pkl')
    
    # Final Evaluation (using last split logic implicit in TSCV or manual split if needed)
    # 여기서는 전체 데이터에 대한 학습 점수를 보여주지만, 실제로는 GridSearch 점수가 중요.
    print(f"📝 Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Feature Importance visualization
    fi = pd.DataFrame({'Feature': features, 'Importance': best_model.feature_importances_})
    print("\n🔑 Final Feature Importance:")
    print(fi.sort_values(by='Importance', ascending=False))
    print("💾 Optimized Model saved.")

if __name__ == "__main__":
    train_intensive()
