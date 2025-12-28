import pandas as pd
import numpy as np
# 🔥 전략 교체: 스마트 머니(SmartMoney) -> 세력 평단가(SmartMoneyCost)
from quant_strategy.domain.strategies.smart_money_cost_strategy import SmartMoneyCostStrategy
from quant_strategy.infrastructure.data_loader import MarketDataLoader

class BacktestService:
    def __init__(self, initial_capital=100_000_000, risk_pct=0.05):
        """
        :param initial_capital: 초기 자본금
        :param risk_pct: 리스크 허용 비율 (기본 5%)
        """
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct 
        self.data_loader = MarketDataLoader()
        
        # 🔥 전략 인스턴스화
        self.strategy = SmartMoneyCostStrategy(window=60)

    def run(self, ticker: str, start_date: str, end_date: str):
        # 1. 데이터 준비
        raw_df = self.data_loader.fetch_data(ticker)
        if raw_df.empty:
            print(f"⚠️ 데이터 없음: {ticker}")
            return pd.DataFrame()

        raw_df = raw_df[(raw_df.index >= pd.to_datetime(start_date)) & (raw_df.index <= pd.to_datetime(end_date))]
        if raw_df.empty:
            return pd.DataFrame()
        
        # 지표 추가 (MA, ATR 등)
        df = self.strategy.add_indicators(raw_df)
        df['Action'] = None 

        # 2. 시뮬레이션 변수
        cash = self.initial_capital
        shares = 0
        
        units = 0            
        last_entry_price = 0 
        stop_loss_price = 0  
        
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

            current_equity = cash + (shares * current_price)
            
            # ------------------------------------
            # [Step 1] 강제 손절 체크 (Stop Loss)
            # 골든크로스라 해도 급락은 피해야 하므로 2N 손절 유지
            # ------------------------------------
            executed_stop_loss = False
            """
            if shares > 0:
                if curr_row['Low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    if curr_row['Open'] < stop_loss_price:
                        exit_price = curr_row['Open'] 
                    
                    cash += shares * exit_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    executed_stop_loss = True
                    df.at[curr_row.name, 'Action'] = 'SELL'"""
            # ------------------------------------
            # [Step 2] 전략 신호 확인
            # ------------------------------------
            if not executed_stop_loss:
                signal = self.strategy.check_signals(curr_row, prev_row, has_position=(shares > 0))

                # === [A. 매수 진입] ===
                # 골든크로스 발생 or 정배열 상태
                if signal == 'BUY' and shares == 0:
                    risk_amount = current_equity * self.risk_pct
                    risk_per_share = 2 * atr 
                    
                    unit_size = int(risk_amount / risk_per_share)
                    max_buyable = int(cash / current_price)
                    buy_amount = min(unit_size, max_buyable)

                    if buy_amount > 0:
                        shares = buy_amount
                        cash -= shares * current_price
                        
                        units = 1
                        last_entry_price = current_price
                        stop_loss_price = current_price - (2 * atr)
                        
                        df.at[curr_row.name, 'Action'] = 'BUY'

                # === [B. 불타기 (Pyramiding)] ===
                # 골든크로스 전략에서도 추세 강화 시 추가 매수 유효
                elif shares > 0 and units < 4:
                    if current_price > last_entry_price + (0.5 * atr): 
                        risk_amount = current_equity * self.risk_pct
                        risk_per_share = 2 * atr
                        unit_size = int(risk_amount / risk_per_share)
                        
                        max_buyable = int(cash / current_price)
                        buy_amount = min(unit_size, max_buyable)
                        
                        if buy_amount > 0:
                            shares += buy_amount
                            cash -= buy_amount * current_price
                            
                            last_entry_price = current_price
                            units += 1
                            
                            new_stop_loss = current_price - (2 * atr)
                            if new_stop_loss > stop_loss_price:
                                stop_loss_price = new_stop_loss
                            
                            df.at[curr_row.name, 'Action'] = 'BUY'

                # === [C. 매도 청산] ===
                # 데드크로스 발생 시 전량 매도
                elif signal == 'SELL' and shares > 0:
                    cash += shares * current_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    df.at[curr_row.name, 'Action'] = 'SELL'

            total_asset = cash + (shares * current_price)
            equity_curve.append(total_asset)

        df['Equity_Strategy'] = equity_curve
        return df