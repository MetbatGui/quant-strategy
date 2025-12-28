import pandas as pd
import numpy as np
from quant_strategy.domain.strategies.turtle_strategy import TurtleStrategy
from quant_strategy.infrastructure.data_loader import MarketDataLoader

class BacktestService:
    def __init__(self, initial_capital=100_000_000, risk_pct=0.05):
        """
        :param initial_capital: 초기 자본금
        :param risk_pct: 1회 진입(1 Unit) 당 허용할 리스크 비율 (기본 5%)
                         0.01(1%) ~ 0.10(10%) 사이 권장
        """
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct 
        self.data_loader = MarketDataLoader()
        self.strategy = TurtleStrategy()

    def run(self, ticker: str, start_date: str, end_date: str):
        # 1. 데이터 준비
        raw_df = self.data_loader.fetch_data(ticker)
        # 날짜 필터링
        raw_df = raw_df[(raw_df.index >= pd.to_datetime(start_date)) & (raw_df.index <= pd.to_datetime(end_date))]
        
        # 전략 지표 계산 (ATR, Donchian Channel 등)
        df = self.strategy.add_indicators(raw_df)

        # 차트 마킹용 컬럼
        df['Action'] = None 

        # 2. 변수 초기화
        cash = self.initial_capital
        shares = 0
        
        # 피라미딩 및 리스크 관리 변수
        units = 0            # 현재 보유 유닛 개수 (Max 4)
        last_entry_price = 0 # 마지막 진입 가격 (피라미딩 기준점)
        stop_loss_price = 0  # 현재 손절가 (2N)
        
        equity_curve = []

        # 대조군(단순 보유) 계산
        if not df.empty:
            hold_shares = self.initial_capital / df['Open'].iloc[0]
            df['Equity_Hold'] = hold_shares * df['Close']

        # 3. 타임라인 루프 실행
        for i in range(len(df)):
            curr_row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else curr_row
            
            # 현재가 및 ATR 확보
            current_price = curr_row['Close']
            atr = curr_row.get('ATR', 0)
            # ATR이 계산 안 된 초기 구간은 임의로 2% 설정 (에러 방지)
            if pd.isna(atr) or atr == 0: 
                atr = current_price * 0.02 

            current_equity = cash + (shares * current_price)
            
            # ------------------------------------
            # [Step 1] 강제 손절 체크 (Stop Loss)
            # ------------------------------------
            executed_stop_loss = False
            if shares > 0:
                # 보수적 체결가 계산: 저가가 손절가 이하면 손절 발동
                if curr_row['Low'] <= stop_loss_price:
                    exit_price = stop_loss_price
                    # 시가부터 갭락으로 손절가 밑에서 시작하면 시가에 체결
                    if curr_row['Open'] < stop_loss_price:
                        exit_price = curr_row['Open'] 
                    
                    # 전량 청산
                    cash += shares * exit_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    executed_stop_loss = True
                    
                    # 마킹
                    df.at[curr_row.name, 'Action'] = 'SELL'

            # ------------------------------------
            # [Step 2] 전략 신호 확인
            # ------------------------------------
            if not executed_stop_loss:
                signal = self.strategy.check_signals(curr_row, prev_row, has_position=(shares > 0))

                # === [A. 신규 진입] (포지션 없을 때) ===
                if signal == 'BUY' and shares == 0:
                    # [자금 관리] 설정된 risk_pct 사용
                    risk_amount = current_equity * self.risk_pct
                    risk_per_share = 2 * atr # 1주당 감내할 손실 (2N)
                    
                    # 매수 유닛 크기 계산
                    unit_size = int(risk_amount / risk_per_share)
                    
                    # 현금 한도 체크
                    max_buyable = int(cash / current_price)
                    buy_amount = min(unit_size, max_buyable)

                    if buy_amount > 0:
                        shares = buy_amount
                        cash -= shares * current_price
                        
                        # 상태 업데이트
                        units = 1
                        last_entry_price = current_price
                        stop_loss_price = current_price - (2 * atr) # 초기 손절가: 진입가 - 2N
                        
                        df.at[curr_row.name, 'Action'] = 'BUY'

                # === [B. 피라미딩 (불타기)] (보유 중 & 상승 추세) ===
                # 조건: 가격이 마지막 진입가보다 0.5N(0.5 ATR) 이상 올랐고 & 유닛이 4개 미만일 때
                elif shares > 0 and units < 4:
                    if current_price > last_entry_price + (0.5 * atr): 
                        # 추가 매수 유닛 계산 (현재 자산 기준)
                        risk_amount = current_equity * self.risk_pct
                        risk_per_share = 2 * atr
                        unit_size = int(risk_amount / risk_per_share)
                        
                        max_buyable = int(cash / current_price)
                        buy_amount = min(unit_size, max_buyable)
                        
                        if buy_amount > 0:
                            shares += buy_amount
                            cash -= buy_amount * current_price
                            
                            # 기준가 갱신 및 유닛 추가
                            last_entry_price = current_price
                            units += 1
                            
                            # [핵심] 트레일링 스탑 (손절 라인 위로 끌어올리기)
                            # 새로운 손절가 = (현재 진입가 - 2N)
                            new_stop_loss = current_price - (2 * atr)
                            if new_stop_loss > stop_loss_price:
                                stop_loss_price = new_stop_loss
                            
                            df.at[curr_row.name, 'Action'] = 'BUY'

                # === [C. 청산 매도] (전략적 매도 신호) ===
                # 20일 신저가 이탈 or 60일 이평선 이탈 등
                elif signal == 'SELL' and shares > 0:
                    cash += shares * current_price
                    shares = 0
                    units = 0
                    stop_loss_price = 0
                    
                    df.at[curr_row.name, 'Action'] = 'SELL'

            # 자산 기록
            total_asset = cash + (shares * current_price)
            equity_curve.append(total_asset)

        # 결과 저장
        df['Equity_Strategy'] = equity_curve
        return df