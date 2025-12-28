import pandas as pd
import numpy as np

class GoldenCrossStrategy:
    """
    [골든 크로스 전략]
    - 불장(Bull Market) 전용 추세 추종
    - 매수: 5일 이평선 > 20일 이평선 (정배열 진입)
    - 매도: 5일 이평선 < 20일 이평선 (역배열 청산)
    """
    
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 이동평균선 계산
        df['MA_Short'] = df['Close'].rolling(window=self.short_window).mean()
        df['MA_Long'] = df['Close'].rolling(window=self.long_window).mean()
        
        # 2. ATR (자금 관리 및 손절 계산용)
        # TR 계산
        c_prev = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - c_prev)
        tr3 = abs(df['Low'] - c_prev)
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR 20일 평균
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        # 지표가 없으면 계산 불가
        if pd.isna(curr_row['MA_Short']) or pd.isna(curr_row['MA_Long']):
            return 'HOLD'
        
        # 이전 봉 지표 확인 (크로스 체크용)
        if pd.isna(prev_row.get('MA_Short')) or pd.isna(prev_row.get('MA_Long')):
            return 'HOLD'

        curr_diff = curr_row['MA_Short'] - curr_row['MA_Long']
        prev_diff = prev_row['MA_Short'] - prev_row['MA_Long']

        # === [매수 로직] ===
        if not has_position:
            # 1. 골든 크로스 발생 (아래에서 위로 돌파)
            if prev_diff <= 0 and curr_diff > 0:
                return 'BUY'
            
            # 2. (옵션) 이미 정배열 상태라면 중도 탑승
            # 불장에서는 기다리지 않고 바로 타는 게 유리함
            if curr_diff > 0:
                return 'BUY'

        # === [매도 로직] ===
        if has_position:
            # 데드 크로스 발생 (위에서 아래로 이탈)
            if prev_diff >= 0 and curr_diff < 0:
                return 'SELL'
            
        return 'HOLD'