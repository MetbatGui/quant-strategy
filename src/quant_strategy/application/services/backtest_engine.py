"""
ETF 품질 점수 전략 백테스트 엔진
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List
from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy
from quant_strategy.domain.entities.trade import Trade
from quant_strategy.domain.entities.portfolio import Portfolio
from quant_strategy.domain.entities.backtest_result import BacktestResult


class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, strategy: EtfQualityStrategy, initial_capital: float = 10_000_000, top_n: int = 1):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.initial_capital = initial_capital
        self.top_n = top_n
    
    def load_etf_data(self, start_date: str, end_date: str, train_start: str = None) -> Dict[str, pd.DataFrame]:
        """
        전체 ETF 데이터 로드
        
        Args:
            start_date: 백테스트 시뮬레이션 시작일
            end_date: 종료일
            train_start: 모델 학습 시작일 (없으면 start_date - 180일)
        
        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        etf_data = {}
        
        # 모델 학습을 위한 시작일 설정
        if train_start:
            # 명시된 학습 시작일 사용
            adjusted_start = train_start
            print(f"📥 ETF 데이터 로딩 (학습용): {adjusted_start} ~ {end_date}")
        else:
            # 기본값: 시작일 430일 전 (사용자 최적화 기간: 24.11.01 기준)
            start_dt = pd.to_datetime(start_date)
            adjusted_start = (start_dt - timedelta(days=430)).strftime('%Y-%m-%d')
            print(f"📥 ETF 데이터 로딩 (기본 430일): {adjusted_start} ~ {end_date}")
        
        for ticker, name in self.strategy.ETF_POOL.items():
            try:
                # yfinance download end date is exclusive, so add 1 day to include end_date
                download_end = (pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                df_raw = yf.download(ticker, start=adjusted_start, end=download_end, progress=False)
                
                # MultiIndex 제거
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
                
                if not df_raw.empty:
                    df = pd.DataFrame()
                    df['Open'] = df_raw['Open']
                    df['High'] = df_raw['High']
                    df['Low'] = df_raw['Low']
                    df['Close'] = df_raw['Close']
                    if 'Volume' in df_raw.columns:
                        df['Volume'] = df_raw['Volume']
                    
                    etf_data[ticker] = df
                    print(f"  ✅ {name}: {len(df)} 일")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        return etf_data
    
    def run(self, start_date: str, end_date: str, train_start: str = None, trade_on_last_day: bool = True) -> BacktestResult:
        """
        백테스트 실행
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
            train_start: 모델 학습 시작일 (Optional)
        
        Returns:
            BacktestResult 객체
        """
        print("\n" + "="*80)
        print(f"🚀 백테스트 시작: {start_date} ~ {end_date}")
        if train_start:
            print(f"📊 모델 학습 기간: {train_start} ~ {end_date}")
        print("="*80 + "\n")
        
        # 1. 데이터 로드
        etf_data = self.load_etf_data(start_date, end_date, train_start)
        
        if not etf_data:
            print("❌ 데이터가 없습니다.")
            return BacktestResult(
                strategy_name="EtfQualityStrategy",
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.portfolio.capital,
                total_return=0.0
            )
        
        # 2. 모델 학습 (Strict Separation)
        print("\n🤖 XGBoost 모델 학습 중... (Strict Separation applied)")
        
        # 학습용 데이터 슬라이싱 (Start Date 이전 데이터만 사용)
        start_dt = pd.to_datetime(start_date)
        train_data = {}
        for ticker, df in etf_data.items():
            # 미래 데이터(Start Date 이후)는 학습에서 제외
            train_df = df[df.index < start_dt].copy()
            if not train_df.empty:
                train_data[ticker] = train_df
        
        if not train_data:
            print("❌ 학습할 과거 데이터가 부족합니다.")
            return BacktestResult(
                strategy_name="EtfQualityStrategy",
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.portfolio.capital,
                total_return=0.0
            )
            
        self.strategy.train_models(train_data)
        print(f"  ✅ {len(self.strategy.models)}개 모델 학습 완료 (학습 종료일: {max([df.index[-1] for df in train_data.values()]).strftime('%Y-%m-%d')})")
        
        # 3. 거래일 생성 (첫 번째 ETF 기준)
        first_ticker = list(etf_data.keys())[0]
        all_dates = etf_data[first_ticker].index
        if trade_on_last_day:
            backtest_dates = all_dates[(all_dates >= pd.to_datetime(start_date)) & (all_dates <= pd.to_datetime(end_date))]
        else:
            # If not trading on last day, we still need to include end_date to allow morning exit
            # But entry logic will be skipped inside the loop
            backtest_dates = all_dates[(all_dates >= pd.to_datetime(start_date)) & (all_dates <= pd.to_datetime(end_date))]

        
        print(f"\n📅 백테스트 기간: {len(backtest_dates)} 거래일")
        print("\n" + "="*80)
        print("거래 시뮬레이션 시작")
        print("="*80 + "\n")
        
        # 4. 백테스트 루프
        for i, current_date in enumerate(backtest_dates):
            # 1. 기존 포지션 관리 (청산)
            just_closed_at_open = False
            
            if self.portfolio.current_position is not None:
                # 당일 진입한 포지션은 청산 대상 아님 (Overnight 전략)
                if self.portfolio.current_position.entry_date < current_date:
                    df = etf_data[self.portfolio.current_position.ticker]
                    
                    # 청산 시그널 확인
                    should_exit, exit_timing = self.strategy.check_exit_signal(
                        df,
                        self.portfolio.current_position.entry_date,
                        current_date
                    )
                    
                    if should_exit:
                        if exit_timing == 'OPEN':
                            # 시가 청산 -> 당일 재진입 기회 있음
                            exit_price = df.loc[current_date, 'Open']
                            self.portfolio.close_position(exit_price, current_date)
                            pos_return = self.portfolio.closed_trades[-1].position_return
                            
                            # --- [벤치마크: 최적의 대안 찾기] ---
                            best_ticker = None
                            best_return = -999.0
                            closed_trade = self.portfolio.closed_trades[-1]
                            entry_dt = closed_trade.entry_date
                            
                            for other_ticker in self.strategy.ETF_POOL.keys():
                                if other_ticker == closed_trade.ticker: continue
                                try:
                                    odf = etf_data[other_ticker]
                                    e_price = self.strategy.calculate_entry_price(odf, entry_dt)
                                    if e_price and self.strategy.check_entry_signal(odf, entry_dt, e_price):
                                        if current_date in odf.index:
                                            x_price = odf.loc[current_date, 'Open']
                                            
                                            # 수수료 반영 (0.015%)
                                            fee = 0.00015
                                            buy_v = e_price * (1+fee)
                                            sell_v = x_price * (1-fee)
                                            ret = (sell_v/buy_v - 1)*100
                                            
                                            if ret > best_return:
                                                best_return = ret
                                                best_ticker = other_ticker
                                except: pass
                                
                            if best_ticker:
                                closed_trade.best_alternative_ticker = best_ticker
                                closed_trade.best_alternative_name = self.strategy.ETF_POOL[best_ticker]
                                closed_trade.best_alternative_return = best_return
                            # -------------------------------

                            print(f"🔴 청산(OPEN): {current_date.strftime('%Y-%m-%d')} | {exit_price:,.0f}원 | {pos_return:+.2f}%")
                            just_closed_at_open = True
                            
                        elif exit_timing == 'CLOSE':
                            # 종가 청산 -> 당일 거래 종료
                            exit_price = df.loc[current_date, 'Close']
                            self.portfolio.close_position(exit_price, current_date)
                            pos_return = self.portfolio.closed_trades[-1].position_return
                            print(f"🔴 청산(CLOSE): {current_date.strftime('%Y-%m-%d')} | {exit_price:,.0f}원 | {pos_return:+.2f}%")
                            continue # 하루 종료
                    else:
                        # 홀딩 -> 당일 거래 종료
                        continue

            # 2. 신규 진입 (포지션이 없거나, 방금 시가 청산한 경우)
            # 마지막 거래일이고 trade_on_last_day가 False면 진입 생략
            if not trade_on_last_day and current_date == backtest_dates[-1]:
                continue

            if self.portfolio.current_position is None:
                # 4-1. 품질 점수 계산
                # IMPORTANT: 점수는 '전일' 데이터 기준으로 계산해야 함 (Lookahead Bias 방지)
                if i > 0:
                    prev_date = backtest_dates[i-1]
                else:
                    curr_idx = all_dates.get_loc(current_date)
                    if curr_idx > 0:
                        prev_date = all_dates[curr_idx - 1]
                    else:
                        continue 

                scores = {}
                score_details_map = {}
                
                for ticker, df in etf_data.items():
                    if prev_date in df.index:
                        score, details = self.strategy.calculate_quality_score(ticker, df, prev_date)
                        scores[ticker] = score
                        score_details_map[ticker] = details
                    else:
                        scores[ticker] = 0
                        score_details_map[ticker] = {'Error': 'No Data'}
                
                # 4-2. 상위 N개 ETF 선택
                top_tickers = self.strategy.select_top_etfs(scores, n=self.top_n)
                
                # 순차적으로 진입 시그널 확인
                for ticker in top_tickers:
                    score = scores[ticker]
                    name = self.strategy.ETF_POOL[ticker]
                    
                    # 4-3. 진입가 계산
                    df = etf_data[ticker]
                    entry_price = self.strategy.calculate_entry_price(df, current_date)
                    
                    if entry_price:
                        # 4-4. 진입 시그널 확인 (고가 돌파)
                        entered = self.strategy.check_entry_signal(df, current_date, entry_price)
                        
                        if entered:
                            # 거래 생성
                            trade = Trade(
                                ticker=ticker,
                                ticker_name=name,
                                entry_date=current_date,
                                entry_price=entry_price,
                                quality_score=score,
                                score_details=score_details_map[ticker]
                            )
                            self.portfolio.open_position(trade)
                            
                            re_entry_tag = " (재진입)" if just_closed_at_open else ""
                            print(f"🔵 진입{re_entry_tag}: {current_date.strftime('%Y-%m-%d')} | {name} | {entry_price:,.0f}원 | 점수: {score}")
                            break  # 1일 1종목 원칙

        # 루프 종료 후 남은 포지션 강제 청산
        if self.portfolio.current_position is not None:
             final_date = backtest_dates[-1]
             df = etf_data[self.portfolio.current_position.ticker]
             close_price = df.loc[final_date, 'Close']
             self.portfolio.close_position(close_price, final_date)
             print(f"🔴 청산(종료): {final_date.strftime('%Y-%m-%d')} | {close_price:,.0f}원 | 강제 청산")
        
        # 5. 결과 분석
        metrics = self.portfolio.get_metrics()
        
        print("\n" + "="*80)
        print("📊 백테스트 결과")
        print("="*80)
        print(f"초기 자본: {self.initial_capital:,.0f}원")
        print(f"최종 자본: {metrics['final_capital']:,.0f}원")
        print(f"총 수익률: {metrics['total_return']:.2f}%")
        print(f"거래 횟수: {metrics['num_trades']}회")
        print(f"승률: {metrics['win_rate']:.1f}%")
        print(f"평균 수익률: {metrics['average_return']:.2f}%")
        print(f"최대 낙폭(MDD): {metrics['max_drawdown']:.2f}%")
        print("="*80 + "\n")
        
        # 6. ETF별 성과 분석
        self._analyze_by_etf()
        
        return BacktestResult(
            strategy_name=self.strategy.__class__.__name__,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=metrics['final_capital'],
            total_return=metrics['total_return'],
            win_rate=metrics['win_rate'],
            num_trades=metrics['num_trades'],
            max_drawdown=metrics['max_drawdown'],
            average_return=metrics['average_return'],
            trades=self.portfolio.closed_trades
        )
    
    def _analyze_by_etf(self):
        """ETF별 성과 분석"""
        if not self.portfolio.closed_trades:
            return
        
        etf_stats = {}
        for trade in self.portfolio.closed_trades:
            ticker = trade.ticker
            if ticker not in etf_stats:
                etf_stats[ticker] = {
                    'name': trade.ticker_name,
                    'count': 0,
                    'wins': 0,
                    'returns': []
                }
            
            etf_stats[ticker]['count'] += 1
            if trade.is_winning:
                etf_stats[ticker]['wins'] += 1
            etf_stats[ticker]['returns'].append(trade.position_return)
        
        print("="*80)
        print("📈 ETF별 성과")
        print("="*80)
        
        for ticker, stats in sorted(etf_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            win_rate = stats['wins'] / stats['count'] * 100
            avg_return = sum(stats['returns']) / len(stats['returns'])
            print(f"{stats['name']:20s} | 거래: {stats['count']:2d}회 | 승률: {win_rate:5.1f}% | 평균: {avg_return:+6.2f}%")
        
        print("="*80 + "\n")