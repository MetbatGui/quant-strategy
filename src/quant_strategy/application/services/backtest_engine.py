import pandas as pd
import numpy as np
# 🔥 전략 교체: 스마트 머니 모멘텀(SmartMoneyMomentum) -> 하이브리드(Hybrid)
from quant_strategy.domain.strategies.hybrid_strategy import HybridStrategy
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
        self.strategy = HybridStrategy()

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

        # 5일 이동평균선 확인 (피닉스 룰용)
        if 'MA5' not in df.columns:
            df['MA5'] = df['Close'].rolling(window=5).mean()
        
        # 2. 시뮬레이션 변수
        cash = self.initial_capital
        shares = 0
        
        units = 0            
        last_entry_price = 0 
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
            
            # [가변 손절 로직]
            # 60일선 위에 있으면 '존버 모드' -> 손절폭 5 ATR (아주 넓게)
            # 60일선 아래에 있으면 '방어 모드' -> 손절폭 2 ATR (타이트하게)
            ma60 = curr_row.get('MA60', 0)
            if shares > 0:
                if current_price > ma60:
                     # 불장: 흔들기에 털리지 않도록 5배 ATR 적용
                    dynamic_stop_width = 5 * atr
                else:
                    # 약세장: 칼손절
                    dynamic_stop_width = 2 * atr
                
                # 진입가 기준 손절 or 최고가 기준 트레일링 스탑?
                # 여기서는 '진입가 기준' 손절 라인을 동적으로 조정하거나,
                # 단순히 현재가가 (진입가 - width)보다 낮으면 파는 식도 가능.
                # 편의상 기존 stop_loss_price 로직을 유지하되, 업데이트는 신중하게.
                # (이미 설정된 stop_loss_price가 있고, 이걸상황에 따라 늘려주는 건 위험할 수 있음)
                # 따라서, "현재 시점의 적정 손절가"를 매번 계산해서 비교하는 방식 적용.
                
                # 다만, 손절가가 위로 따라올라가는 건(Trailing) 좋지만 내려가는 건 안됨.
                pass 

            current_equity = cash + (shares * current_price)
            
            # ------------------------------------
            # [Step 0] 피닉스 룰 (즉시 재진입)
            # ------------------------------------
            # 최근 3일 이내에 손절/매도했고 + 현재가가 5일선 위에 있으면 즉시 재매수
            # 불장의 V자 반등 놓치지 않기 위함
            is_phoenix_entry = False
            if shares == 0 and last_exit_idx != -1:
                # 인덱스가 날짜라고 가정
                days_since_exit = (df.index[i] - df.index[last_exit_idx]).days
                ma5 = curr_row.get('MA5', 0)
                
                if days_since_exit <= 3 and current_price > ma5:
                    is_phoenix_entry = True

            # ------------------------------------
            # [Step 1] 강제 손절 체크 (Dynamic Stop Loss)
            # ------------------------------------
            executed_stop_loss = False
            # 피닉스 진입이 아닐 때만 손절 체크 (재진입 당일 손절 방지 등)
            if shares > 0 and not is_phoenix_entry:
                # 동적 손절가격 계산 (진입가 기준이 아니라 '현재가'가 이 가격 밑이면 위험하다는 판정)
                # 여기서는 트레일링 스탑 개념을 적용: '최근 고점' 대비 하락폭?
                # 아니면 간단하게 '진입가 대비'로 하되 폭만 조절? -> 진입가 대비로 하면 이미 수익 났을 때 토해내는 문제.
                
                # 심플하게: "매수 후 최고가"를 추적해야 함 (변수 추가 필요하지만 복잡해짐).
                # 대안: 기존 stop_loss_price 변수를 업데이트하는 로직을 강화.
                
                # 여기서는 기존 stop_loss_price를 존중하되, 
                # '방어 모드'일 때 stop_loss_price가 너무 낮으면(널널하면) 끌어올리는 로직? (복잡)
                
                # 가장 확실한 방법: 여기서 직접 stop loss 체크
                # (기존 stop_loss_price 변수보다, 전략적 손절 라인이 우선)
                
                check_price = stop_loss_price 
                
                # 만약 '방어 모드(MA60 아래)'인데 손절선이 너무 멀다면? (진입 시점엔 불장이어서 5ATR이었을 수 있음)
                # -> 즉시 2ATR로 조이고 싶지만, 그러면 바로 손절 나갈 수 있음.
                # -> 일단은 진입 시 정해진 손절폭을 유지하되, 추가 매수 시 업데이트 로직에 맡김.
                
                if curr_row['Low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    if curr_row['Open'] < stop_loss_price:
                        exit_price = curr_row['Open'] 
                    
                    cash += shares * exit_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    executed_stop_loss = True
                    last_exit_idx = i # 매도 시점 기록
                    df.at[curr_row.name, 'Action'] = 'SELL'

            # ------------------------------------
            # [Step 2] 전략 신호 확인
            # ------------------------------------
            if not executed_stop_loss:
                signal = self.strategy.check_signals(curr_row, prev_row, has_position=(shares > 0))

                # === [A. 매수 진입] ===
                # 피닉스 룰 발동 OR 일반 매수 신호
                if (signal == 'BUY' or is_phoenix_entry) and shares == 0:
                    risk_amount = current_equity * self.risk_pct
                    # 레버리지 투입: 손절은 4N이지만, 유닛 사이징은 2N 기준으로 계산 (공격적)
                    # 즉, 손절 시 -10% 손실 감수 (기존 -5%)
                    risk_per_share = 2 * atr 
                    
                    unit_size = int(risk_amount / risk_per_share)
                    max_buyable = int(cash / current_price)
                    buy_amount = min(unit_size, max_buyable)

                    if buy_amount > 0:
                        shares = buy_amount
                        cash -= shares * current_price
                        
                        units = 1
                        last_entry_price = current_price
                        
                        # [초기 손절가 설정]
                        # 60일선 위(불장) -> 5 ATR (존버)
                        # 60일선 아래(약세) -> 2 ATR (칼손절)
                        ma60 = curr_row.get('MA60', 0)
                        if current_price > ma60:
                            stop_loss_price = current_price - (5 * atr) 
                        else:
                            stop_loss_price = current_price - (2 * atr)
                        
                        df.at[curr_row.name, 'Action'] = 'BUY'

                # === [B. 불타기 (Pyramiding)] ===
                # 불타기도 상황 봐가며
                elif shares > 0 and units < 6:
                    # 불장에서는 좀 더 공격적으로? (0.3 ATR)
                    threshold = 0.3 * atr
                    if current_price > last_entry_price + threshold: 
                        risk_amount = current_equity * self.risk_pct
                        risk_per_share = 2 * atr # 공격적 사이징 유지
                        unit_size = int(risk_amount / risk_per_share)
                        
                        max_buyable = int(cash / current_price)
                        buy_amount = min(unit_size, max_buyable)
                        
                        if buy_amount > 0:
                            shares += buy_amount
                            cash -= buy_amount * current_price
                            
                            last_entry_price = current_price
                            units += 1
                            
                            # 손절 라인 업데이트 (Trailing Stop)
                            ma60 = curr_row.get('MA60', 0)
                            if current_price > ma60:
                                stop_width = 5 * atr
                            else:
                                stop_width = 2 * atr
                                
                            new_stop_loss = current_price - stop_width
                            if new_stop_loss > stop_loss_price:
                                stop_loss_price = new_stop_loss
                            
                            df.at[curr_row.name, 'Action'] = 'BUY'

                # === [C. 매도 청산] ===
                elif signal == 'SELL' and shares > 0:
                    cash += shares * current_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    last_exit_idx = i # 매도 시점 기록
                    df.at[curr_row.name, 'Action'] = 'SELL'

            total_asset = cash + (shares * current_price)
            equity_curve.append(total_asset)

        df['Equity_Strategy'] = equity_curve
        return df