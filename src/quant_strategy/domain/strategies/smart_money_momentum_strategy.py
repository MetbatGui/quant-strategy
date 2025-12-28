import pandas as pd
import numpy as np

class SmartMoneyMomentumStrategy:
    """
    [스마트 머니 모멘텀 전략]
    - 목적: 상승장에서 세력과 함께 끝까지 수익을 추구 (Let profits run).
    - 특징: 익절 목표가 없음. 세력 평단가를 지지선으로 삼아 이탈 시에만 매도.
    """
    
    def __init__(self, window=60):
        self.window = window # 세력 평단가 산출 기간

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 데이터 유효성 체크
        if 'Foreigner' not in df.columns or 'Institution' not in df.columns:
             # 컬럼 없으면 0으로 채워서 에러 방지
            df['Foreigner'] = 0
            df['Institution'] = 0

        # 1. 스마트 머니 수급
        df['Smart_Vol'] = df['Foreigner'] + df['Institution']
        df['Smart_Amt'] = df['Close'] * df['Smart_Vol']
        
        # 2. 세력 평단가 (Whale Price) - 매수일 기준 VWAP
        buy_mask = df['Smart_Vol'] > 0
        df['Buy_Vol'] = np.where(buy_mask, df['Smart_Vol'], 0)
        df['Buy_Amt'] = np.where(buy_mask, df['Smart_Amt'], 0)
        
        df['Cum_Amt'] = df['Buy_Amt'].rolling(window=self.window).sum()
        df['Cum_Vol'] = df['Buy_Vol'].rolling(window=self.window).sum()
        
        df['Whale_Price'] = np.where(
            df['Cum_Vol'] > 0, 
            df['Cum_Amt'] / df['Cum_Vol'], 
            np.nan
        )
        
        # 3. 신고가 (Breakout 감지)
        df['Highest_20'] = df['High'].rolling(window=20).max().shift(1)
        
        # 4. ATR (변동성)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        whale_price = curr_row.get('Whale_Price')
        current_price = curr_row['Close']
        highest_20 = curr_row.get('Highest_20')
        
        if pd.isna(whale_price) or whale_price == 0:
            return 'HOLD'

        # === [매수 로직: 비싸도 산다] ===
        if not has_position:
            # 수급 필수: 오늘 세력이 사고 있는가?
            is_smart_buying = curr_row['Smart_Vol'] > 0
            
            # 조건 A: 신고가 돌파 (Breakout)
            # 전일 기준 20일 고가를 현재가가 넘어섰거나 같음
            if pd.isna(highest_20): highest_20 = current_price * 1.5 # 데이터 부족 방어
            is_breakout = current_price >= highest_20
            
            # 조건 B: 세력 평단가 지지 (Support)
            # 평단가 대비 98% 이상 (너무 멀어지지 않은 지지 구간)
            is_support = current_price >= whale_price * 0.98
            
            # 세력 평단가보다는 위에 있어야 함 (정배열 전제)
            is_above_whale = current_price > whale_price
            
            if is_smart_buying and is_above_whale and (is_breakout or is_support):
                return 'BUY'

        # === [매도 로직: 끝까지 버틴다] ===
        if has_position:
            # 익절 로직 없음 (Let profits run)
            
            # 손절/청산 로직: 추세 붕괴
            # 세력 평단가 대비 3~5% 이탈 시 매도
            # 여기서는 5% 이탈로 설정 (넉넉하게)
            if current_price < whale_price * 0.95:
                return 'SELL'
            
            # 추가 안전장치: 스마트 머니 대량 이탈? (옵션)
            # 일단은 가격 추세 추종에 집중
            
        return 'HOLD'
