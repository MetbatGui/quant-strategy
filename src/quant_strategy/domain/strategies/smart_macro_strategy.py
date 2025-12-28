import pandas as pd
import numpy as np
from quant_strategy.domain.strategies.macro_hybrid_strategy import MacroHybridStrategy

class SmartMacroStrategy(MacroHybridStrategy):
    """
    [스마트 매크로 전략 (Smart Macro Strategy)]
    - 부제: Whale Shield (세력 방패)
    - 상속: MacroHybridStrategy (글로벌 매크로 필터 계승)
    
    1. 개선점
       - Whale Shield: 매도 신호가 나와도 수급(외국인/기관)이 매수 중이면 '개미털기'로 보고 HOLD.
       - Chandelier Exit: MA 이탈 대신 고점 대비 변동성(ATR) 이탈로 청산하여 급등주 추세 향유.
    
    2. Logic
       - Entry: (기존) Global Bull + Golden Cross
       - Exit:
         (1) Primary: Chandelier Exit 이탈
         (2) Shield: 
             IF (Primary Exit Triggered) OR (Global/Local Sell Signal):
                 IF (Smart Money 5일 합계 > 0): HOLD (방어 발동)
                 ELSE: SELL
         (3) Hard Stop: Price < MA60 (System Defense - 절대 방어선)
    """
    
    def __init__(self, window=60):
        super().__init__(window)
        
    def add_indicators(self, df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        # 부모 클래스의 지표 계산 (MA, ATR, Macro 등)
        df = super().add_indicators(df, macro_df)
        
        # === [추가 지표 계산] ===
        
        # 1. 스마트 머니 계산 (수급 합계)
        # Foreigner, Institution 컬럼이 있다고 가정 (BacktestService/DataLoader에서 병합됨)
        if 'Foreigner' in df.columns:
            # 일별 합계 (단위: 원) -> 숫자가 크므로 정규화보다는 부호 위주
            df['Smart_Daily'] = df['Foreigner'] + df['Institution']
            # 5일 누적 수급 (단기 세력 의도 파악)
            df['Smart_Sum_5'] = df['Smart_Daily'].rolling(window=5).sum()
            # 20일 누적 수급 (중기 매집 확인용 - 샹들리에 계수 조절)
            df['Smart_Sum_20'] = df['Smart_Daily'].rolling(window=20).sum()
        else:
            df['Smart_Sum_5'] = 0
            df['Smart_Sum_20'] = 0
            
        # 2. 샹들리에 청산 (Chandelier Exit)
        # 공식: Highest High(20) - k * ATR(20)
        # k값: 기본 3.0, 수급 좋으면 5.0 (더 널널하게)
        
        # (1) 20일 신고가
        df['High_20'] = df['High'].rolling(window=20).max()
        
        # (2) 동적 k값 계산 (벡터 연산 불가하므로 apply 혹은 numpy where)
        # 수급이 좋으면(20일 누적 양수) k=5, 아니면 k=3
        k_factor = np.where(df['Smart_Sum_20'] > 0, 5.0, 3.0)
        
        # (3) 샹들리에 컷 계산
        if 'ATR' in df.columns:
            df['Chandelier_Cut'] = df['High_20'] - (k_factor * df['ATR'])
        else:
            df['Chandelier_Cut'] = df['High_20'] * 0.95 # Fallback
            
        # 전일 기준 컷을 써야 당일 장중 대응 가능 (보수적 접근: 당일 종가 vs 전일 컷은 아님, 
        # 보통 백테스트에선 당일 종가 vs 당일 계산된 컷 or 전일 컷 비교.
        # 여기선 '당일 종가'로 판단하므로 당일 컷을 계산하되, High_20은 "현재 봉 포함"이므로
        # 주가가 떨어지면 High도 유지됨. Chandelier는 "Trailing High" 개념.
        # 정확히는 "진입 이후 최고가" 기준이지만, 단순화해서 "최근 20일 고가" 사용.
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        # 1. 부모 클래스의 매수/매도 로직 먼저 확인?
        # -> 부모 로직이 'SELL'일 때 Shield를 쳐야 함.
        # -> 다만 부모 check_signals는 내부 변수를 안 뱉으므로, 여기서 다시 구현하는 게 깔끔함.
        
        current_price = curr_row['Close']
        ma5 = curr_row.get('MA5', 0)
        ma20 = curr_row.get('MA20', 0)
        ma60 = curr_row.get('MA60', 0)
        macro_bull = curr_row.get('Macro_Bull', 1)
        
        smart_sum_5 = curr_row.get('Smart_Sum_5', 0)
        chandelier_cut = curr_row.get('Chandelier_Cut', 0)
        
        # === [매수 로직 (Entry)] ===
        if not has_position:
            # 부모와 동일 (Global Bull + Golden Cross)
            if macro_bull == 1:
                if ma5 > ma20:
                    return 'BUY'

        # === [매도 로직 (Exit)] ===
        if has_position:
            signal = 'HOLD'
            
            # [조건 1] 샹들리에 컷 이탈 (이익 보전)
            if current_price < chandelier_cut:
                signal = 'SELL'
            
            # [조건 2] 글로벌/개별 악재 (부모 로직 계승)
            if macro_bull == 0 and current_price < ma20:
                signal = 'SELL'
            elif current_price < ma60: # 절대 방어선
                # MA60 깨지면 세력이고 뭐고 튐 (하락장 진입)
                return 'SELL' 
                
            # === [🛡️ Whale Shield (세력 방패)] ===
            if signal == 'SELL':
                # 절대 방어선(MA60) 위라면, 수급 확인 후 구조
                if current_price >= ma60:
                     # 최근 5일간 세력이 사고 있다면? -> 개미털기다. 버텨라.
                     if smart_sum_5 > 0:
                         return 'HOLD'
            
            return signal
                
        return 'HOLD'
