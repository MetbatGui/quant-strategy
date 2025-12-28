import pandas as pd
import numpy as np

class SmartWhaleStrategy:
    """
    [Smart Whale Strategy] (Target 150%)
    - "세력과 운명을 함께한다 (Absolute Shield)"
    - Smart Optimized에서 MA60 방어선마저 해제.
    - 10-bagger 종목(펩트론, 알테오젠)은 개미털기가 심해서 MA60도 깬다.
    - 오직 'Smart_Sum_20 > 0' (월간 수급) 하나만 믿고 버틴다.
    
    [Inherited Logic Consolidted]
    - from MacroHybrid: NDX > 60MA (Macro Bull Filter).
    - from SmartMacro: ATR, MA, Smart_Sum calc.
    """
    
    def __init__(self, window=60):
        self.window = window
        
    def add_indicators(self, df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()
        
        # 1. 개별 종목 지표 (MA, ATR)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # ATR 계산
        df['TR'] = np.maximum(df['High'] - df['Low'], 
                              np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                         abs(df['Low'] - df['Close'].shift(1))))
        df['ATR'] = df['TR'].rolling(window=20).mean().shift(1)

        # 2. 매크로 지표 병합 (Nasdaq Filter)
        if macro_df is not None and not macro_df.empty:
            macro_calc = macro_df.copy()
            if '^NDX' in macro_calc.columns:
                macro_calc['NDX_MA60'] = macro_calc['^NDX'].rolling(window=60).mean()
                macro_calc['Macro_Bull'] = np.where(macro_calc['^NDX'] > macro_calc['NDX_MA60'], 1, 0)
                macro_calc['Macro_Bull'] = macro_calc['Macro_Bull'].fillna(0)
            else:
                macro_calc['Macro_Bull'] = 1
                
            # Timezone Normalize
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            if macro_calc.index.tz is not None: macro_calc.index = macro_calc.index.tz_localize(None)
            macro_calc.index = macro_calc.index.normalize()

            # Merge
            cols_to_merge = ['Macro_Bull']
            if 'KRW=X' in macro_calc.columns: cols_to_merge.append('KRW=X')
            
            # Left Join & Fillna
            macro_subset = macro_calc[cols_to_merge]
            df = df.join(macro_subset, how='left')
            df['Macro_Bull'] = df['Macro_Bull'].ffill().fillna(1)
        else:
            df['Macro_Bull'] = 1

        # 3. 스마트 머니 계산 (Smart_Sum)
        if 'Foreigner' in df.columns:
            df['Smart_Daily'] = df['Foreigner'] + df['Institution']
            # 20일 세력 합계 (Whale Shield 핵심)
            df['Smart_Sum_20'] = df['Smart_Daily'].rolling(window=20).sum()
        else:
            df['Smart_Sum_20'] = 0
            
        # 4. 샹들리에 컷 (k=3.0)
        # Whale Shield가 있으므로 빡빡하게 k=3.0 적용
        df['High_20'] = df['High'].rolling(window=20).max()
        if 'ATR' in df.columns:
             df['Chandelier_Cut'] = df['High_20'] - (3.0 * df['ATR'])
        else:
             df['Chandelier_Cut'] = df['High_20'] * 0.95
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool, entry_price: float = 0) -> str:
        current_price = curr_row['Close']
        ma5 = curr_row.get('MA5', 0)
        ma20 = curr_row.get('MA20', 0)
        ma60 = curr_row.get('MA60', 0)
        macro_bull = curr_row.get('Macro_Bull', 1)
        
        smart_sum_20 = curr_row.get('Smart_Sum_20', 0)
        chandelier_cut = curr_row.get('Chandelier_Cut', 0)
        
        # === [매수 로직 (Entry)] ===
        if not has_position:
            if macro_bull == 1:
                # Classic Golden Cross
                if ma5 > ma20:
                    return 'BUY'

        # === [매도 로직 (Exit)] ===
        if has_position:
            signal = 'HOLD'
            
            # [조건 1] 샹들리에 컷 이탈 (k=3.0 Strict)
            if current_price < chandelier_cut:
                signal = 'SELL'
            
            # [조건 2] 글로벌 악재 (Macro Bear + MA20 Break)
            if macro_bull == 0:
                 if smart_sum_20 <= 0: # 세력도 없으면 탈출
                     if current_price < ma20:
                        signal = 'SELL'
            
            # [조건 3] 절대 방어선 (MA60)
            if current_price < ma60:
                signal = 'SELL'

            # === [🐋 Absolute Whale Shield] ===
            # 매도 신호가 떴지만, "세력이 한 달(20일) 동안 매집 중"이라면?
            # -> 개미털기다. 기술적 신호 무시하고 무조건 버틴다.
            if signal == 'SELL':
                if smart_sum_20 > 0:
                    return 'HOLD'
            
            return signal
            
        return 'HOLD'
