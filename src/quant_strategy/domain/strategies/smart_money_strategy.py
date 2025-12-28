import pandas as pd
import numpy as np

class SmartMoneyStrategy:
    """
    [스마트 머니 포착 전략]
    - 목적: 외국인(Foreigner)과 기관(Institution)의 수급이 쏠리는 지점 포착
    - 매수: '쌍끌이(양매수)' 발생 + 누적 순매수 증가 + 정배열
    - 특징: 차트는 아직 조용한데 수급이 들어올 때(선취매) 잡기에 유리함.
    """
    
    def __init__(self, window=5):
        self.window = window # 수급 누적 기간 (5일)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 0. 데이터 존재 여부 확인 (수급 데이터가 없으면 에러 방지)
        if 'Foreigner' not in df.columns:
            df['Foreigner'] = 0 # 더미 데이터
        if 'Institution' not in df.columns:
            df['Institution'] = 0 # 더미 데이터

        # 1. 수급 지표 계산
        # 스마트 머니 합계 = 외국인 + 기관
        df['Smart_Money'] = df['Foreigner'] + df['Institution']
        
        # 최근 N일간 스마트 머니 누적 순매수량
        df['Smart_Cumsum'] = df['Smart_Money'].rolling(window=self.window).sum()
        
        # 2. 추세 지표 (20일선)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # 3. ATR (자금 관리용)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        # 데이터 부족 시
        if pd.isna(curr_row['MA20']):
            return 'HOLD'

        # === [매수 로직] ===
        if not has_position:
            # 조건 1: 오늘 외국인과 기관이 동시에 샀는가? (양매수)
            is_twin_buy = (curr_row['Foreigner'] > 0) and (curr_row['Institution'] > 0)
            
            # 조건 2: 최근 5일간 누적으로도 매수세인가? (지속성)
            is_accumulating = curr_row['Smart_Cumsum'] > 0
            
            # 조건 3: 주가가 20일선 위에 있는가? (상승 추세)
            # 수급이 좋아도 역배열이면 '물타기'일 수 있으므로 제외
            is_uptrend = curr_row['Close'] > curr_row['MA20']
            
            if is_twin_buy and is_accumulating and is_uptrend:
                return 'BUY'

        # === [매도 로직] ===
        if has_position:
            # 스마트 머니가 이탈(대량 매도)하거나 추세가 꺾이면 청산
            # 여기서는 '스마트 머니 누적합'이 음수로 돌아서면 매도
            if curr_row['Smart_Cumsum'] < 0:
                return 'SELL'
            
            # 혹은 20일선 이탈 시 매도 (안전장치)
            if curr_row['Close'] < curr_row['MA20']:
                return 'SELL'
            
        return 'HOLD'