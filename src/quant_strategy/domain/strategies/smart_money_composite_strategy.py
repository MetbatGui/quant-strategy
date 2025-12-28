import pandas as pd
import numpy as np

class SmartMoneyCompositeStrategy:
    """
    [스마트 머니 복합 전략 (Smart Money Composite)]
    - 목적: 수급(Smart Money), 변동성(ATR), 추세(MA)를 '동적으로' 결합하여 신뢰도 향상.
    - 핵심: 
      1. 스마트 머니 유입 시 '샹들리에 청산(Chandelier Exit)'을 느슨하게 풀어주어 수익 극대화 (Let profits run).
      2. 스마트 머니 이탈 시 '샹들리에 청산'을 타이트하게 조여 이익 보전 (Tight Stop).
    """
    
    def __init__(self, window=20):
        self.window = window

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 이동평균선 (Trend)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # 2. ATR (Volatility)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)
        
        # 3. 스마트 머니 흐름 (Smart Money Flow)
        if 'Foreigner' not in df.columns: df['Foreigner'] = 0
        if 'Institution' not in df.columns: df['Institution'] = 0
        df['Smart_Vol'] = df['Foreigner'] + df['Institution']
        
        # 20일 누적 순매수 (단기~중기 수급 추세)
        df['Smart_Cum_20'] = df['Smart_Vol'].rolling(window=20).sum()
        
        # 세력 평단가 (보조 지지선)
        buy_mask = df['Smart_Vol'] > 0
        df['Buy_Amt'] = np.where(buy_mask, df['Close'] * df['Smart_Vol'], 0)
        df['Buy_Vol'] = np.where(buy_mask, df['Smart_Vol'], 0)
        
        # 60일 기준 VWAP
        cum_amt = df['Buy_Amt'].rolling(window=60).sum()
        cum_vol = df['Buy_Vol'].rolling(window=60).sum()
        df['Whale_Price'] = np.where(cum_vol > 0, cum_amt / cum_vol, np.nan)

        # 4. 동적 샹들리에 청산 (Dynamic Chandelier Exit)
        # 공식: 최근 N일 최고가 - (Multiplier * ATR)
        period_high = df['High'].rolling(window=20).max()
        
        # 수급이 좋으면(양수) 4 ATR (여유), 나쁘면 2 ATR (타이트)
        multiplier = np.where(df['Smart_Cum_20'] > 0, 4.0, 2.0)
        
        df['Chandelier_Exit'] = period_high - (multiplier * df['ATR'])
        
        # 60일선이 지지선 역할
        # 최종 매도 라인: MAX(샹들리에, MA60) -> 
        # 아니면 둘 중 하나라도 깨지면 매도? -> 샹들리에가 더 민감하므로 샹들리에 기준.
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        current_price = curr_row['Close']
        ma5 = curr_row.get('MA5', 0) # 엔진에서 계산해서 넣어줌
        ma20 = curr_row.get('MA20', 0)
        ma60 = curr_row.get('MA60', 0)
        smart_flow = curr_row.get('Smart_Cum_20', 0)
        chandelier = curr_row.get('Chandelier_Exit', 0)
        
        if pd.isna(chandelier) or chandelier == 0: return 'HOLD'

        # === [매수 로직 (Entry) - 완화] ===
        if not has_position:
            # 1. 골든 크로스 (가장 강력한 추세 신호)
            # 5일선 > 20일선 (단기 정배열)
            if float(ma5) == 0: # MA5가 없는 경우 대비 (엔진에서 계산하지만 안전장치)
                 ma5 = current_price # 임시
            
            is_golden = ma5 > ma20
            
            # 2. 스마트 머니 (보조)
            # 수급이 좋으면 더 좋지만, 수급이 없어도 차트가 좋으면 진입해야 함 (기회비용 방지)
            
            if is_golden and current_price > ma60: # 정배열 & 대세 상승
                return 'BUY'

        # === [매도 로직 (Exit)] ===
        if has_position:
            # 1. 샹들리에 청산
            # 여기서 스마트 머니의 역할이 중요함
            # 수급이 좋으면 Chandelier가 4ATR로 널널해서 안 털림
            # 수급이 나쁘면 Chandelier가 2ATR로 타이트해서 털림
            if current_price < chandelier:
                return 'SELL'
            
            # 2. 추세 붕괴 (60일선)
            if not pd.isna(ma60) and current_price < ma60:
                return 'SELL'
                
        return 'HOLD'
