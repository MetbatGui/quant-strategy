import pandas as pd
import numpy as np
from quant_strategy.infrastructure.data_loader import MarketDataLoader

class BacktestService:
    def __init__(self, strategy, initial_capital=100_000_000, risk_pct=0.05):
        """
        :param strategy: 사용할 전략 인스턴스 (Duck Typing)
        :param initial_capital: 초기 자본금
        :param risk_pct: 리스크 허용 비율 (기본 5%)
        """
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct 
        self.data_loader = MarketDataLoader()
        self.strategy = strategy


    def run(self, ticker: str, start_date: str, end_date: str):
        # 1. 데이터 준비
        raw_df = self.data_loader.fetch_data(ticker)
        if raw_df.empty:
            print(f"⚠️ 데이터 없음: {ticker}")
            return pd.DataFrame()
        
        # 매크로 데이터 가져오기 (나스닥, 환율)
        # 충분한 윈도우(MA60) 확보를 위해 start_date보다 미리 가져오면 좋지만,
        # 편의상 전체를 가져와서 내부 Join으로 처리
        macro_df = self.data_loader.fetch_macro_data()

        raw_df = raw_df[(raw_df.index >= pd.to_datetime(start_date)) & (raw_df.index <= pd.to_datetime(end_date))]
        if raw_df.empty:
            return pd.DataFrame()
        
        # 지표 추가 (MA, ATR, Macro 등)
        # 지표 추가 (MA, ATR, Macro 등)
        df = self.strategy.add_indicators(raw_df, macro_df=macro_df)
        df['Action'] = None 
        
        # 5일 이동평균선 확인 (피닉스 룰용)
        if 'MA5' not in df.columns:
            df['MA5'] = df['Close'].rolling(window=5).mean()

        # 2. 시뮬레이션 변수
        cash = self.initial_capital
        shares = 0
        
        units = 0            
        last_entry_price = 0 
        self.avg_price = 0 # 평단가 (수익률 계산용)
        # stop_loss_price는 이제 Strategy의 Chandelier Exit이 담당하므로 보조적 역할(안전망)만 수행
        stop_loss_price = 0
        last_exit_idx = -1 # 마지막 매도 시점 (피닉스 룰용)
        
        equity_curve = []

        # 대조군 계산
        if not df.empty:
            hold_shares = self.initial_capital / df['Open'].iloc[0]
            df['Equity_Hold'] = hold_shares * df['Close']

        # 3. 루프 실행
        for i in range(len(df)):
            curr_row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else curr_row
            
            current_price = curr_row['Close']
            atr = curr_row.get('ATR', 0)
            if pd.isna(atr) or atr == 0: 
                atr = current_price * 0.02 
            
            # [전략 로직 위임]
            # 엔진에서는 복잡한 손절 계산을 하지 않고, 전략이 'SELL'을 주면 판다.
            # 단, 안전망(Safety Net)으로 진입가 대비 -10% 등 강제 손절 기능을 둘 순 있으나,
            # 여기서는 전략의 Chandelier Exit을 신뢰.

            current_equity = cash + (shares * current_price)
            
            # ------------------------------------
            # [Step 0] 피닉스 룰 (즉시 재진입)
            # ------------------------------------
            is_phoenix_entry = False
            # 매크로 전략에서는 '시스템 리스크(나스닥 하락)' 시 진입 금지이므로,
            # 피닉스 룰도 나스닥 불장일 때만 작동해야 함.
            macro_bull = curr_row.get('Macro_Bull', 1)
            
            if macro_bull == 1 and shares == 0 and last_exit_idx != -1:
                days_since_exit = (df.index[i] - df.index[last_exit_idx]).days
                ma5 = curr_row.get('MA5', 0)
                
                if days_since_exit <= 3 and current_price > ma5:
                    is_phoenix_entry = True

            # ------------------------------------
            # [Step 1] 전략 신호 확인 (매도 포함)
            # ------------------------------------
            # 5. 매매 신호 확인
            # entry_price(평단가)를 전달하여 수익률 기반 로직 가능하게 함
            has_position = (shares > 0)
            signal = self.strategy.check_signals(curr_row, prev_row, has_position, entry_price=self.avg_price)
            
            # 6. 매수/매도 실행
            # === [A. 매도 청산] ===
            # 나스닥 폭락 or 개별 종목 악재
            if shares > 0 and signal == 'SELL':
                cash += shares * current_price
                shares = 0
                units = 0
                self.avg_price = 0 # 매도 시 평균 매수 단가 초기화
                last_exit_idx = i
                df.at[curr_row.name, 'Action'] = 'SELL'
            
            # === [B. 매수 진입] ===
            # 피닉스 룰 발동 OR 일반 매수 신호
            elif shares == 0 and (signal == 'BUY' or is_phoenix_entry):
                risk_amount = current_equity * self.risk_pct
                # 공격적 사이징: 2 ATR Risk
                risk_per_share = 2 * atr 
                
                unit_size = int(risk_amount / risk_per_share)
                max_buyable = int(cash / current_price)
                buy_amount = min(unit_size, max_buyable)

                if buy_amount > 0:
                    shares = buy_amount
                    cash -= shares * current_price
                    self.avg_price = current_price # 첫 진입 평단가
                    
                    units = 1
                    last_entry_price = current_price
                    # stop_loss_price: 전략이 알아서 SELL 신호 주므로 여기선 명시적 관리 안 함 (단, 안전망 필요시 추가)
                    
                    df.at[curr_row.name, 'Action'] = 'BUY'

            # === [C. 불타기 (Pyramiding)] ===
            # 불도 상황 봐가며 (Max 6 Units)
            elif shares > 0 and units < 6:
                # 0.3 ATR 상승 시
                threshold = 0.3 * atr
                if current_price > last_entry_price + threshold: 
                    risk_amount = current_equity * self.risk_pct
                    risk_per_share = 2 * atr
                    unit_size = int(risk_amount / risk_per_share)
                    
                    max_buyable = int(cash / current_price)
                    buy_amount = min(unit_size, max_buyable)
                    
                    if buy_amount > 0:
                        # 평단가 갱신 (가중 평균)
                        total_cost = (shares * self.avg_price) + (buy_amount * current_price)
                        shares += buy_amount
                        cash -= buy_amount * current_price
                        self.avg_price = total_cost / shares
                        
                        last_entry_price = current_price
                        units += 1
                        
                        df.at[curr_row.name, 'Action'] = 'BUY'

            total_asset = cash + (shares * current_price)
            equity_curve.append(total_asset)

        df['Equity_Strategy'] = equity_curve
        return df