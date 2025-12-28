import pandas as pd
import numpy as np

class HybridStrategy:
    """
    [하이브리드 전략: 추세 + 수급 + 변동성]
    - 목적: 불장에서는 '존버' 수익률을 따라잡고, 하락장에서는 '방어'한다.
    - 혼합 요소:
      1. MA (이동평균): 추세 판단 (5일, 20일, 60일)
      2. Smart Money (수급): 외국인/기관 매집 여부 확인
      3. ATR (변동성): 자금 관리
    """
    
    def __init__(self, short_win=5, mid_win=20, long_win=60):
        self.short_win = short_win # 진입용
        self.mid_win = mid_win     # 단기 추세용
        self.long_win = long_win   # 대세 하락 판단용 (생명선)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 이동평균선 (MA)
        df['MA5'] = df['Close'].rolling(window=self.short_win).mean()
        df['MA20'] = df['Close'].rolling(window=self.mid_win).mean()
        df['MA60'] = df['Close'].rolling(window=self.long_win).mean()
        
        # 2. 스마트 머니 (수급)
        # 데이터가 없으면 0으로 처리
        if 'Foreigner' not in df.columns: df['Foreigner'] = 0
        if 'Institution' not in df.columns: df['Institution'] = 0
        
        df['Smart_Vol'] = df['Foreigner'] + df['Institution']
        # 최근 5일간 수급이 들어왔는가? (양수면 매집 중)
        df['Smart_Trend'] = df['Smart_Vol'].rolling(window=5).sum().shift(1)
        
        # 3. ATR (변동성)
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool) -> str:
        # 데이터 부족 시
        if pd.isna(curr_row['MA60']): return 'HOLD'
        
        current_price = curr_row['Close']
        ma5 = curr_row['MA5']
        ma20 = curr_row['MA20']
        ma60 = curr_row['MA60']
        
        # === [상태 정의] ===
        # 1. 불장(Bull Regime): 주가가 60일선 위에 있음 (대세 상승)
        is_bull_regime = current_price > ma60
        
        # 2. 수급 긍정(Smart Buy): 최근 5일간 메이저가 매집 중
        is_smart_buying = curr_row['Smart_Trend'] > 0
        
        # 3. 골든크로스: 5일선이 20일선 위에 있음 (단기 상승)
        is_golden = ma5 > ma20

        # === [매수 로직 (Entry)] ===
        if not has_position:
            # 조건: (대세 상승장 OR 수급 매집) AND (단기 정배열)
            # 수급이 좋으면 60일선 아래라도 선취매 가능 (Smart Money 효과)
            if (is_bull_regime or is_smart_buying) and is_golden:
                return 'BUY'

        # === [매도 로직 (Exit) - 핵심!] ===
        if has_position:
            # 🔥 [하이브리드 모드]
            
            # CASE A: 대세 상승장(60일선 위) + 수급 좋음
            # -> "존버 모드": 자잘한 데드크로스 무시. 60일선 깨질 때까지 안 팖.
            if is_bull_regime and is_smart_buying:
                if current_price < ma60: # 생명선 붕괴 시에만 매도
                    return 'SELL'
                else:
                    return 'HOLD' # 5일선 깨져도 버팀 (휩소 방지)
            
            # CASE B: 힘이 약한 장 (60일선 아래 or 수급 이탈)
            # -> "칼손절 모드": 20일선만 깨져도 바로 도망감.
            else:
                if current_price < ma20: # 단기 추세 이탈 시 매도
                    return 'SELL'
                
                # 혹은 스마트 머니가 대량 매도 시 탈출
                if curr_row['Smart_Trend'] < 0 and current_price < ma5:
                    return 'SELL'

        return 'HOLD'
