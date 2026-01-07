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


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14, smooth_k: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator 계산
    
    Args:
        high: 고가
        low: 저가
        close: 종가
        period: 기간 (기본 14)
        smooth_k: %K 스무딩 (기본 3)
    
    Returns:
        DataFrame with 'k', 'd' columns
    """
    # Fast %K
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    fast_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Slow %K (Fast %D)
    stoch_k = fast_k.rolling(window=smooth_k).mean()
    
    # Slow %D
    stoch_d = stoch_k.rolling(window=smooth_k).mean()
    
    return pd.DataFrame({'k': stoch_k, 'd': stoch_d})


def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands 계산
    
    Args:
        close: 종가
        period: 이동평균 기간 (기본 20)
        std_dev: 표준편차 승수 (기본 2.0)
    
    Returns:
        DataFrame with 'upper', 'middle', 'lower', 'width', 'percent_b'
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # Band Width (변동성 지표)
    width = (upper - lower) / middle
    
    # %B (밴드 내 위치)
    percent_b = (close - lower) / (upper - lower)
    
    return pd.DataFrame({
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'width': width,
        'percent_b': percent_b
    })

def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) 계산
    
    Args:
        close: 종가 시리즈
        fast: 단기 EMA 기간 (기본 12)
        slow: 장기 EMA 기간 (기본 26)
        signal: 시그널 EMA 기간 (기본 9)
    
    Returns:
        DataFrame with 'macd', 'signal', 'hist'
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'hist': hist
    })


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX (Average Directional Index) 계산
    
    Args:
        df: OHLC 데이터프레임
        period: 기간 (기본 14)
    
    Returns:
        ADX 값 시리즈
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm.abs()), 0.0)
    minus_dm = minus_dm.where((minus_dm < 0) & (minus_dm.abs() > plus_dm), 0.0).abs()
    
    tr = calculate_atr(df, period=1) # TR for 1 period
    
    atr_smooth = tr.rolling(window=period).sum() # Initial smoothing
    # Wilders smoothing for better accuracy matching standard libraries could be complex, 
    # but rolling sum/mean is often sufficient for ML features. 
    # Let's use simple rolling sum for DM and TR as approximation or EWMA.
    # Standard ADX uses Wilder's Smoothing.
    
    # Using specific Wilder's smoothing logic:
    # Smooth(t) = Smooth(t-1) - (Smooth(t-1)/n) + Current(t)
    # This is equivalent to EWM with com=period-1
    
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx
