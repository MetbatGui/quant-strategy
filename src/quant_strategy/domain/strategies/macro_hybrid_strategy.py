import pandas as pd
import numpy as np

class MacroHybridStrategy:
    """
    [매크로 하이브리드 전략 (Macro Hybrid)]
    - 목적: "세계 증시가 망하면 한국 증시는 무조건 망한다"는 대전제에 기반.
    - 핵심: 나스닥(Nasdaq)과 환율(Exchange Rate)을 필터로 사용하여 '시스템 리스크'를 회피.
    
    1. Regime Logic:
       - Global Bull: 나스닥 > 60일선 (안전 구간)
       - Global Bear: 나스닥 < 60일선 (위험 구간)
    
    2. Trading Logic:
       - Entry: Global Bull일 때만 '하이브리드/모멘텀' 진입 (Bear일 땐 현금 보유).
       - Exit: Global Bear 전환 시 즉시 비중 축소 or 청산.
    """
    
    def __init__(self, window=60):
        self.window = window

    def add_indicators(self, df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 개별 종목 지표
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # ATR
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)

        # 2. 매크로 지표 병합
        if macro_df is not None and not macro_df.empty:
            # 매크로 지표 계산 (병합 전에 수행해야 전체 기간 MA가 나옴)
            macro_calc = macro_df.copy()
            if '^NDX' in macro_calc.columns:
                macro_calc['NDX_MA60'] = macro_calc['^NDX'].rolling(window=60).mean()
                macro_calc['Macro_Bull'] = np.where(macro_calc['^NDX'] > macro_calc['NDX_MA60'], 1, 0)
                # 앞쪽 NaN은 0으로 처리 (보수적)
                macro_calc['Macro_Bull'] = macro_calc['Macro_Bull'].fillna(0)
            else:
                macro_calc['Macro_Bull'] = 1
                
            # Timezone 정리 (Normalize to match dates)
            # df.index와 macro_calc.index를 모두 normalize
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            
            if macro_calc.index.tz is not None:
                macro_calc.index = macro_calc.index.tz_localize(None)
            macro_calc.index = macro_calc.index.normalize()

            # 필요한 컬럼만 선택해서 병합
            cols_to_merge = ['Macro_Bull']
            if 'KRW=X' in macro_calc.columns:
                cols_to_merge.append('KRW=X')
                
            # 병합 (Left Join)
            # 중복 컬럼 방지
            macro_subset = macro_calc[cols_to_merge]
            df = df.join(macro_subset, how='left')
            
            # 결측치 처리 (휴장일 등 매칭 안 되는 날)
            # 전일 값 사용 (ffill) -> 그래도 없으면 1 (Bull)
            df['Macro_Bull'] = df['Macro_Bull'].ffill().fillna(1)
            
        else:
             df['Macro_Bull'] = 1 # Fallback

        return df

    def check_signals(self, curr_row, prev_row, has_position: bool, entry_price: float = 0) -> str:
        current_price = curr_row['Close']
        ma5 = curr_row.get('MA5', 0)
        ma20 = curr_row.get('MA20', 0)
        ma60 = curr_row.get('MA60', 0)
        macro_bull = curr_row.get('Macro_Bull', 1) # 기본값 Bull
        
        # === [매수 로직 (Entry)] ===
        if not has_position:
            # 대전제: 글로벌 불장이어야 함 (나스닥 > 60일선)
            if macro_bull == 1:
                # 개별 종목 조건: 5일 > 20일 (골든크로스) OR 60일선 위
                # 더 공격적으로 잡기: 그냥 골든크로스면 진입
                if ma5 > ma20:
                    return 'BUY'

        # === [매도 로직 (Exit)] ===
        if has_position:
            # 1. 글로벌 악재 (System Risk)
            # 나스닥이 꺾였다고 무조건 파는 게 아니라, 내 종목도 흔들릴 때(20일선 이탈) 매도
            # 즉, '불장'일 땐 60일선까지 버티지만, '시스템 위기'땐 20일선만 깨져도 도망감.
            if macro_bull == 0:
                if current_price < ma20:
                     return 'SELL'
            
            # 2. 개별 종목 악재 (60일선 붕괴)
            # 평상시(글로벌 불장)에는 60일선이 생명선
            elif current_price < ma60:
                return 'SELL'
            
            # 참고: 글로벌 불장(Macro=1)이고, 개별 종목이 20일선 깨졌을 땐?
            # -> Hybrid 전략에 따르면 '존버' 모드이므로 60일선까지 버팀. (여기서 별도 처리 불필요)
                
        return 'HOLD'
