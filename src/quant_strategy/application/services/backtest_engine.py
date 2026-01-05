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


class BacktestEngine:
    """백테스트 엔진"""
    
    def __init__(self, strategy: EtfQualityStrategy, initial_capital: float = 10_000_000):
        self.strategy = strategy
        self.portfolio = Portfolio(initial_capital)
        self.initial_capital = initial_capital
    
    def load_etf_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        전체 ETF 데이터 로드
        
        Args:
            start_date: 시작일
            end_date: 종료일
        
        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        etf_data = {}
        
        # 충분한 과거 데이터 확보를 위해 시작일 3개월 전부터 로드
        start_dt = pd.to_datetime(start_date)
        adjusted_start = (start_dt - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"📥 ETF 데이터 로딩 중... ({adjusted_start} ~ {end_date})")
        
        for ticker, name in self.strategy.ETF_POOL.items():
            try:
                df_raw = yf.download(ticker, start=adjusted_start, end=end_date, progress=False)
                
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
    
    def run(self, start_date: str, end_date: str) -> Dict:
        """
        백테스트 실행
        
        Args:
            start_date: 백테스트 시작일
            end_date: 백테스트 종료일
        
        Returns:
            백테스트 결과 딕셔너리
        """
        print("\n" + "="*80)
        print(f"🚀 백테스트 시작: {start_date} ~ {end_date}")
        print("="*80 + "\n")
        
        # 1. 데이터 로드
        etf_data = self.load_etf_data(start_date, end_date)
        
        if not etf_data:
            print("❌ 데이터가 없습니다.")
            return {}
        
        # 2. 모델 학습
        print("\n🤖 XGBoost 모델 학습 중...")
        self.strategy.train_models(etf_data)
        print(f"  ✅ {len(self.strategy.models)}개 모델 학습 완료")
        
        # 3. 거래일 생성 (첫 번째 ETF 기준)
        first_ticker = list(etf_data.keys())[0]
        all_dates = etf_data[first_ticker].index
        backtest_dates = all_dates[all_dates >= pd.to_datetime(start_date)]
        
        print(f"\n📅 백테스트 기간: {len(backtest_dates)} 거래일")
        print("\n" + "="*80)
        print("거래 시뮬레이션 시작")
        print("="*80 + "\n")
        
        # 4. 백테스트 루프
        i = 0
        while i < len(backtest_dates):
            current_date = backtest_dates[i]
            
            # 포지션이 없을 때만 새로운 진입 시도
            if self.portfolio.current_position is None:
                # 4-1. 품질 점수 계산
                scores = {}
                score_details_map = {}
                
                for ticker, df in etf_data.items():
                    score, details = self.strategy.calculate_quality_score(ticker, df, current_date)
                    scores[ticker] = score
                    score_details_map[ticker] = details
                
                # 4-2. 최적 ETF 선택
                best_ticker = self.strategy.select_best_etf(scores)
                
                if best_ticker:
                    best_score = scores[best_ticker]
                    best_name = self.strategy.ETF_POOL[best_ticker]
                    
                    # 4-3. 진입가 계산
                    df = etf_data[best_ticker]
                    entry_price = self.strategy.calculate_entry_price(df, current_date)
                    
                    if entry_price:
                        # 4-4. 진입 시그널 확인 (고가 돌파)
                        entered = self.strategy.check_entry_signal(df, current_date, entry_price)
                        
                        if entered:
                            # 거래 생성
                            trade = Trade(
                                ticker=best_ticker,
                                ticker_name=best_name,
                                entry_date=current_date,
                                entry_price=entry_price,
                                quality_score=best_score,
                                score_details=score_details_map[best_ticker]
                            )
                            self.portfolio.open_position(trade)
                            
                            print(f"🔵 진입: {current_date.strftime('%Y-%m-%d')} | {best_name} | {entry_price:,.0f}원 | 점수: {best_score}")
            
            # 포지션이 열려있으면 청산 확인
            if self.portfolio.current_position is not None:
                # 익일로 이동
                i += 1
                if i >= len(backtest_dates):
                    # 백테스트 종료일에 강제 청산
                    exit_date = backtest_dates[i-1]
                    df = etf_data[self.portfolio.current_position.ticker]
                    exit_price = df.loc[exit_date, 'Close']
                    self.portfolio.close_position(exit_price, exit_date)
                    
                    pos_return = self.portfolio.closed_trades[-1].position_return
                    print(f"🔴 청산(종료): {exit_date.strftime('%Y-%m-%d')} | {exit_price:,.0f}원 | {pos_return:+.2f}%")
                    break
                
                exit_date = backtest_dates[i]
                df = etf_data[self.portfolio.current_position.ticker]
                
                # 청산 시그널 확인
                should_exit, exit_timing = self.strategy.check_exit_signal(
                    df,
                    self.portfolio.current_position.entry_date,
                    exit_date
                )
                
                if should_exit:
                    exit_price = df.loc[exit_date, exit_timing.capitalize()]
                    self.portfolio.close_position(exit_price, exit_date)
                    
                    pos_return = self.portfolio.closed_trades[-1].position_return
                    print(f"🔴 청산({exit_timing}): {exit_date.strftime('%Y-%m-%d')} | {exit_price:,.0f}원 | {pos_return:+.2f}%")
            
            i += 1
        
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
        
        return metrics
    
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