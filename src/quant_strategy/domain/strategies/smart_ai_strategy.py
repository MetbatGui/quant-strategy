import pandas as pd
import numpy as np
import joblib
import os
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

class SmartAiStrategy(SmartWhaleStrategy):
    """
    [Smart AI Strategy] (Target 200%)
    - "세력의 등에 타고, AI의 눈으로 검증한다."
    - Base: SmartWhaleStrategy (Rule-based)
    - Filter: XGBoost Model (AI)
    
    Logic:
    1. Smart Whale이 'BUY' 신호를 보냄 (Golden Cross + Macro Bull)
    2. AI에게 물어봄: "이거 진짜 5일 뒤에 오를까?"
       - Input: 이격도, 변동성, 수급 강도, RSI 등
    3. AI가 OK(1) 하면 진입, NO(0) 하면 Pass.
    """
    
    def __init__(self, window=60, model_path='models/xgb_whale_v3_deep.pkl'):
        super().__init__(window)
        self.model = None
        
        # Load AI Model
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"🤖 AI Model Loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ AI Model Load Failed: {e}")
        else:
            print(f"⚠️ AI Model not found at {model_path}. Running in Classic Mode.")

        self.current_score = 0.5 # Default Neutral confidence

    def add_indicators(self, df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
        # 1. Base Indicators
        df = super().add_indicators(df, macro_df)
        
        # 2. AI Extra Features
        # Smart Z
        roll_mean = df['Smart_Sum_20'].rolling(60).mean()
        roll_std = df['Smart_Sum_20'].rolling(60).std()
        df['Smart_Z'] = (df['Smart_Sum_20'] - roll_mean) / (roll_std + 1e-9)
        
        # Macro Columns 
        if macro_df is not None and not macro_df.empty:
            if df.index.tz is not None and macro_df.index.tz is None:
                 macro_df.index = macro_df.index.tz_localize(df.index.tz)
            elif df.index.tz is None and macro_df.index.tz is not None:
                 macro_df.index = macro_df.index.tz_localize(None)

            m_re = macro_df.reindex(df.index, method='ffill')
            if 'CL=F' in m_re.columns:
                df['Oil_Chg'] = m_re['CL=F'].pct_change().fillna(0)
                df['Gold_Chg'] = m_re['GC=F'].pct_change().fillna(0)
                df['Bond_Yield'] = m_re['^TNX'] / 100.0
                df['USD_KRW'] = m_re['KRW=X'].pct_change().fillna(0)
            else:
                 df['Oil_Chg'] = 0; df['Gold_Chg'] = 0; df['Bond_Yield'] = 0; df['USD_KRW'] = 0
        else:
            df['Oil_Chg'] = 0; df['Gold_Chg'] = 0; df['Bond_Yield'] = 0; df['USD_KRW'] = 0

        # OBV Chg
        if 'OBV' in df.columns:
            df['OBV_Chg'] = df['OBV'].pct_change().fillna(0)
        
        return df

    def check_signals(self, curr_row, prev_row, has_position: bool, entry_price: float = 0) -> str:
        # 1. Whale Signal (Rules)
        signal = super().check_signals(curr_row, prev_row, has_position, entry_price)
        
        # If model is missing, fallback to Rules
        if self.model is None:
            return signal
            
        # 2. AI Calculation
        try:
            # Features V3 (14 cols)
            log_ret = np.log(curr_row['Close'] / prev_row['Close']) if prev_row['Close'] > 0 else 0
            dist_ma20 = curr_row['Close'] / curr_row['MA20'] - 1 if curr_row['MA20'] > 0 else 0
            dist_ma60 = curr_row['Close'] / curr_row['MA60'] - 1 if curr_row['MA60'] > 0 else 0
            
            vol_avg = curr_row.get('Vol_20_Avg', 1)
            vol_roll = curr_row['Volume'] / (vol_avg + 1)
            
            rsi_norm = curr_row.get('RSI', 50) / 100.0
            stoch_k_norm = curr_row.get('Stoch_K', 50) / 100.0
            bb_width = curr_row.get('BB_Width', 0.1)
            macd_hist = curr_row.get('MACD', 0) - curr_row.get('MACD_Signal', 0)
            
            obv_chg = curr_row.get('OBV_Chg', 0)
            smart_z = curr_row.get('Smart_Z', 0)
            
            oil_chg = curr_row.get('Oil_Chg', 0)
            gold_chg = curr_row.get('Gold_Chg', 0)
            bond_yield = curr_row.get('Bond_Yield', 0)
            usd_krw = curr_row.get('USD_KRW', 0)
            
            X_pred = np.array([[
                log_ret, dist_ma20, dist_ma60, vol_roll, 
                rsi_norm, stoch_k_norm, bb_width, macd_hist, obv_chg,
                smart_z, 
                oil_chg, gold_chg, bond_yield, usd_krw
            ]])
            
            probs = self.model.predict_proba(X_pred)[0]
            ai_score = probs[1]
            self.current_score = ai_score # [NEW] Save for Dynamic Sizing
            
            # === [Alpha Max Logic] ===
            
            if signal == 'BUY':
                # Entry Filter: Crisis Stopper (Threshold 0.2)
                # "폭락장만 아니면 사라"
                if ai_score >= 0.20:
                    return 'BUY'
                else:
                    return 'HOLD'
            
            elif signal == 'SELL':
                # Exit Filter: Bull Market Guard (Threshold 0.6)
                # "AI가 강력하다고 하면(>60%), 룰이 팔라고 해도 버텨라"
                # 단, MA60(생명선)이 깨지면 무조건 탈출.
                
                ma60 = curr_row.get('MA60', 0)
                current_price = curr_row['Close']
                
                if current_price > ma60:
                    if ai_score > 0.60:
                        return 'HOLD' # "AI가 보증하는 강세장. 흔들리지 마라."
                
                return 'SELL'

        except Exception as e:
            # print(f"AI Check Error: {e}")
            return signal # Fallback to Rule
            
        return signal

