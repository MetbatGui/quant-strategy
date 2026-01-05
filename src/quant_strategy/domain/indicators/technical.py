"""
기술적 지표 계산 유틸리티
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Volume Ratio
"""

import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI(Relative Strength Index) 계산
    
    Args:
        prices: 종가 시리즈
        period: RSI 기간 (기본 14)
    
    Returns:
        RSI 값 (0~100)
    """
    delta = prices.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(Average True Range) 계산
    
    Args:
        df: OHLC 데이터프레임
        period: ATR 기간 (기본 14)
    
    Returns:
        ATR 값
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    return atr


def calculate_atr_percentile(df: pd.DataFrame, atr_period: int = 14, lookback: int = 60) -> pd.Series:
    """
    ATR의 백분위수 계산 (변동성 체제 판단용)
    
    Args:
        df: OHLC 데이터프레임
        atr_period: ATR 계산 기간
        lookback: 백분위 계산 기간
    
    Returns:
        ATR 백분위 (0~100)
    """
    atr = calculate_atr(df, period=atr_period)
    
    def percentile_rank(window):
        if len(window) < 2:
            return 50.0
        current = window.iloc[-1]
        return (window < current).sum() / len(window) * 100
    
    atr_pct = atr.rolling(window=lookback, min_periods=lookback).apply(percentile_rank, raw=False)
    
    return atr_pct


def calculate_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    거래량 비율 계산 (현재 거래량 / 평균 거래량)
    
    Args:
        df: 거래량 포함 데이터프레임
        period: 평균 거래량 계산 기간
    
    Returns:
        거래량 비율 (1.0 = 평균)
    """
    avg_volume = df['Volume'].rolling(window=period, min_periods=period).mean()
    volume_ratio = df['Volume'] / avg_volume
    
    return volume_ratio


def is_valid_volatility_regime(atr_percentile: float, min_pct: float = 20.0, max_pct: float = 80.0) -> bool:
    """
    정상 변동성 체제인지 확인
    
    Args:
        atr_percentile: ATR 백분위
        min_pct: 최소 백분위 (기본 20%)
        max_pct: 최대 백분위 (기본 80%)
    
    Returns:
        정상 범위 여부
    """
    if pd.isna(atr_percentile):
        return False
    return min_pct < atr_percentile < max_pct


def has_sufficient_volume(volume_ratio: float, threshold: float = 1.2) -> bool:
    """
    충분한 거래량이 있는지 확인
    
    Args:
        volume_ratio: 거래량 비율
        threshold: 최소 비율 (기본 1.2 = 평균대비 20% 이상)
    
    Returns:
        충분한 거래량 여부
    """
    if pd.isna(volume_ratio):
        return False
    return volume_ratio >= threshold
