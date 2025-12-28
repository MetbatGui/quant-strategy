import pandas as pd
import numpy as np

class SmartMoneyCostStrategy:
    """
    [세력 평단가 추적 전략]
    - 원리: 최근 N일간 외국인/기관이 매수한 금액을 바탕으로 '그들의 평단가'를 역산.
    - 매수: 현재가가 '세력 평단가' 근처에 왔을 때 (세력이 방어해 줄 거라는 믿음)
    - 매도: 현재가가 '세력 평단가'보다 15~20% 이상 비싸지면 (세력의 차익실현 구간)
    """
    
    def __init__(self, window=60):
        # 60일(약 3개월) 동안의 수급을 분석
        self.window = window

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 데이터 유효성 체크
        if 'Foreigner' not in df.columns or 'Institution' not in df.columns:
            return df

        # 1. 스마트 머니 순매수량 (수량)
        df['Smart_Vol'] = df['Foreigner'] + df['Institution']
        
        # 2. 스마트 머니 순매수 금액 (추정치 = 종가 * 순매수량)
        # 더 정확히 하려면 (Open+High+Low+Close)/4 * Volume 등을 써야 하지만 종가로 근사
        df['Smart_Amt'] = df['Close'] * df['Smart_Vol']
        
        # 3. 세력 평단가 계산 (핵심 로직)
        # 윈도우 기간 내에 '순매수(양수)'인 날들의 데이터만 합산
        # (매도한 날은 평단가 희석을 막기 위해 제외하거나, 단순히 누적합으로 계산하기도 함)
        
        # 여기서는 단순하게 기간 내 누적 금액 / 누적 수량으로 계산
        # Rolling Sum 이용
        # 주의: 순매수/순매도가 섞여있어서 단순히 더하면 'Net' 평단가가 됨.
        # User Logic: (기간 내 순매수 금액의 합) / (기간 내 순매수 수량의 합)
        # 단, 순매수가 '+'인 날만 계산에 포함하는 것이 더 정확하다고 했음.
        
        # 양수 필터링 (매수한 날만)
        buy_mask = df['Smart_Vol'] > 0
        df['Buy_Vol'] = np.where(buy_mask, df['Smart_Vol'], 0)
        df['Buy_Amt'] = np.where(buy_mask, df['Smart_Amt'], 0)
        
        df['Cum_Amt'] = df['Buy_Amt'].rolling(window=self.window).sum()
        df['Cum_Vol'] = df['Buy_Vol'].rolling(window=self.window).sum()
        
        # 평단가 = 누적 금액 / 누적 수량
        # 수량이 0이거나 음수면 계산 불가 (무한대 방지)
        df['Whale_Price'] = np.where(
            df['Cum_Vol'] > 0, 
            df['Cum_Amt'] / df['Cum_Vol'], 
            np.nan
        )
        
        # ATR 계산 (BacktestService에서 사용할 수도 있으므로 추가)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)

        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        whale_price = curr_row.get('Whale_Price')
        current_price = curr_row['Close']
        
        if pd.isna(whale_price) or whale_price == 0:
            return 'HOLD'

        # [수익률 괴리율]
        # 세력 평단가 대비 현재가가 어디에 있는가?
        ratio = current_price / whale_price
        
        # === [매수 로직] ===
        # 조건 1: 세력이 매집 중이어야 함 (누적 수량 > 0)
        # 조건 2: 현재가가 세력 평단가 근처임 (0.98 ~ 1.05) 
        # -> 세력도 본전 근처라 주가를 방어할 확률 높음 (눌림목)
        if not has_position:
            is_accumulating = curr_row['Cum_Vol'] > 0
            is_near_cost = 0.98 <= ratio <= 1.05 
            
            if is_accumulating and is_near_cost:
                return 'BUY'

        # === [매도 로직] ===
        if has_position:
            # 1. 차익 실현: 세력 평단가보다 20% 이상 오르면 세력도 팔고 싶어함
            if ratio >= 1.20:
                return 'SELL'
            
            # 2. 손절: 세력 평단가가 깨짐 (세력도 물림 -> 투매 나올 수 있음)
            # 평단가 대비 -5% 이탈 시
            if ratio < 0.95:
                # 여기서 바로 SELL을 리턴하면 BacktestService에서 처리가 됨.
                # 단, BacktestService의 로직이 'Stop Loss'와 'Techncial Sell'을 구분하지 않을 수 있음.
                return 'SELL'
            
        return 'HOLD'
