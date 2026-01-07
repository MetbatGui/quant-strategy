"""
XGBoost 거래 수익 예측 모델
과거 패턴과 시장 조건으로 다음 거래의 수익률 예측
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from quant_strategy.domain.indicators.technical import calculate_atr, calculate_rsi, calculate_volume_ratio


def create_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    거래 예측을 위한 특징 생성
    
    Args:
        df: OHLCV 데이터
        lookback: 과거 참조 기간
    
    Returns:
        특징 데이터프레임
    """
    features = pd.DataFrame(index=df.index)
    
    # 1. 가격 모멘텀 (과거 N일 수익률)
    for i in range(1, lookback + 1):
        features[f'return_{i}d'] = df['Close'].pct_change(i) * 100
    
    # 2. 변동성
    atr = calculate_atr(df, period=14)
    features['atr'] = atr / df['Close'] * 100
    features['atr_change'] = features['atr'].pct_change() * 100
    
    # 3. RSI
    features['rsi'] = calculate_rsi(df['Close'], period=14)
    features['rsi_change'] = features['rsi'].diff()
    
    # 4. 거래량
    if 'Volume' in df.columns:
        vol_ratio = calculate_volume_ratio(df, period=20)
        features['volume_ratio'] = vol_ratio
        features['volume_change'] = vol_ratio.pct_change() * 100
    else:
        features['volume_ratio'] = 1.0
        features['volume_change'] = 0.0
    
    # 5. 캔들 패턴
    body = abs(df['Close'] - df['Open'])
    range_val = df['High'] - df['Low']
    features['body_ratio'] = body / range_val
    features['upper_shadow'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / range_val
    features['lower_shadow'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / range_val
    
    # 6. 시간 특징
    features['day_of_week'] = df.index.dayofweek
    features['day_of_month'] = df.index.day
    
    # 7. 추세 지표
    features['ma5'] = df['Close'].rolling(5).mean()
    features['ma20'] = df['Close'].rolling(20).mean()
    features['price_to_ma5'] = (df['Close'] / features['ma5'] - 1) * 100
    features['price_to_ma20'] = (df['Close'] / features['ma20'] - 1) * 100
    
    return features.dropna()


def create_target(df: pd.DataFrame, entry_k: float = 0.5) -> pd.Series:
    """
    타겟 변수 생성: 다음 거래의 수익률
    
    Args:
        df: OHLCV 데이터
        entry_k: 진입 기준 k 값
    
    Returns:
        수익률 시리즈
    """
    # 1. 진입가 계산 (변동성 돌파)
    # Target(T) = Open(T) + Range(T-1) * k
    lev_rng = (df['High'] - df['Low']).shift(1)
    target_price = df['Open'] + (lev_rng * entry_k)
    
    # 2. 청산가 계산 (익일 시가)
    # Exit(T+1) = Open(T+1)
    # shift(-1) of Open gives Open(T+1)
    exit_price = df['Open'].shift(-1)
    
    # 3. 수익률 계산
    # Return = (Exit / Entry - 1)
    returns = (exit_price / target_price - 1 - 0.0005) * 100
    
    return returns


def train_xgboost_model(df: pd.DataFrame, test_size: float = 0.2, entry_k: float = 0.5):
    """
    XGBoost 모델 학습
    
    Args:
        df: OHLCV 데이터
        test_size: 테스트 비율
        entry_k: 진입 k 값
    
    Returns:
        (model, feature_names, metrics)
    """
    # 특징 및 타겟 생성
    X = create_features(df)
    
    # Target Shift:
    # X(T)는 T일 종가 기준 특징.
    # 우리는 X(T)를 보고 T+1일의 거래(T+1 진입 -> T+2 청산)를 예측하고 싶음.
    # 따라서 y를 -1 shift하여 X(T)와 매핑해야 함.
    y = create_target(df, entry_k=entry_k).shift(-1)
    
    # 공통 인덱스
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    
    # NaN 제거
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 50:
        raise ValueError("Not enough training data after cleaning")
    
    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # XGBoost 파라미터
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.05,
        'n_estimators': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # 모델 학습
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # 평가
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    y_pred = model.predict(X_test)
    
    # 방향성 정확도
    direction_acc = ((y_pred > 0) == (y_test > 0)).mean()
    
    metrics = {
        'train_r2': train_score,
        'test_r2': test_score,
        'direction_accuracy': direction_acc,
        'n_samples': len(X)
    }
    
    return model, X.columns.tolist(), metrics


def predict_trade_return(df: pd.DataFrame, date, model, feature_names) -> float:
    """
    특정 날짜의 거래 수익률 예측
    
    Args:
        df: OHLCV 데이터
        date: 예측 날짜
        model: 학습된 모델
        feature_names: 특징 이름
    
    Returns:
        예측 수익률 (%)
    """
    try:
        features = create_features(df)
        
        if date not in features.index:
            return 0.0
        
        feature_vector = features.loc[date][feature_names].values.reshape(1, -1)
        prediction = model.predict(feature_vector)[0]
        
        return prediction
        
    except:
        return 0.0
