import pandas as pd
import numpy as np

class TurtleStrategy:
    """
    [심화 터틀 전략 (Turtle System 2 + Trend Filter + ATR)]
    - 진입: 55일 신고가 돌파
    - 청산: 20일 신저가 이탈 OR 60일 이평선 이탈
    - 자금관리용 데이터: ATR(20) 계산 추가
    """
    
    def __init__(self, buy_period=55, sell_period=20, filter_period=60, atr_period=20):
        self.buy_period = buy_period
        self.sell_period = sell_period
        self.filter_period = filter_period
        self.atr_period = atr_period # ATR 계산 기간 (보통 20일)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 돈치안 채널 (Donchian Channel) - Shift 1 필수
        df['Donchian_High'] = df['High'].rolling(window=self.buy_period).max().shift(1)
        df['Donchian_Low'] = df['Low'].rolling(window=self.sell_period).min().shift(1)
        
        # 2. 추세 필터용 이동평균선
        if self.filter_period:
            df['MA_Filter'] = df['Close'].rolling(window=self.filter_period).mean().shift(1)
            
        # 3. ATR (Average True Range) 계산 - 자금 관리 핵심
        # TR = Max(|High-Low|, |High-PrevClose|, |Low-PrevClose|)
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - prev_close)
        tr3 = abs(df['Low'] - prev_close)
        
        # 3가지 중 최댓값을 TR로 선정
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR은 TR의 20일 이동평균
        df['ATR'] = df['TR'].rolling(window=self.atr_period).mean().shift(1) 
        # 주의: 당일 ATR을 보고 진입 수량을 정하려면 전일 기준 ATR(shift 1)을 쓰는 게 보수적/현실적임.
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        current_close = curr_row['Close']
        donchian_high = curr_row['Donchian_High']
        donchian_low = curr_row['Donchian_Low']
        
        # ATR이나 지표가 없으면 계산 불가
        if pd.isna(donchian_high) or pd.isna(donchian_low):
            return 'HOLD'

        # [매수] 55일 신고가 돌파 + (옵션) 이평선 위
        if not has_position:
            is_breakout = current_close > donchian_high
            
            is_uptrend = True
            if self.filter_period and not pd.isna(curr_row.get('MA_Filter')):
                if current_close <= curr_row['MA_Filter']:
                    is_uptrend = False
            
            if is_breakout and is_uptrend:
                return 'BUY'
            
        # [매도] 20일 신저가 이탈 (추세 반전)
        if has_position:
            if current_close < donchian_low:
                return 'SELL'
            
        return 'HOLD'