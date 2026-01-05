"""
일일 거래 신호 생성 서비스
predict_monday.py의 로직을 서비스화
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy


class SignalService:
    """일일 거래 신호 생성"""
    
    def __init__(self, strategy: EtfQualityStrategy):
        self.strategy = strategy
    
    def generate_daily_signal(self, target_date: Optional[str] = None) -> Dict:
        """
        특정 날짜의 거래 신호 생성
        
        Args:
            target_date: 목표 날짜 (None이면 오늘)
        
        Returns:
            신호 정보 딕셔너리
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        target_dt = pd.to_datetime(target_date)
        
        print("=" * 80)
        print(f"📅 일일 거래 신호 생성: {target_date}")
        print("=" * 80)
        print()
        
        # 1. 데이터 로드 (충분한 기간)
        start_date = (target_dt - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
        
        etf_data = {}
        
        print("📥 데이터 로드 및 모델 학습 중...")
        for ticker, name in self.strategy.ETF_POOL.items():
            try:
                df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
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
                    print(f"  ✅ {name}")
            except Exception as e:
                print(f"  ❌ {name}: {e}")
        
        if not etf_data:
            print("❌ 데이터가 없습니다.")
            return {}
        
        # 2. 모델 학습
        self.strategy.train_models(etf_data)
        
        # 3. 최신 거래일 찾기
        first_ticker = list(etf_data.keys())[0]
        all_dates = etf_data[first_ticker].index
        available_dates = all_dates[all_dates <= target_dt]
        
        if len(available_dates) == 0:
            print(f"❌ {target_date} 이전에 거래일이 없습니다.")
            return {}
        
        latest_date = available_dates[-1]
        
        print()
        print("=" * 80)
        print(f"🔍 최신 거래일({latest_date.strftime('%Y-%m-%d')}) 기준 품질 점수 분석")
        print("=" * 80)
        print()
        
        # 4. 품질 점수 계산
        scores = {}
        score_details_map = {}
        latest_prices = {}
        
        for ticker, name in self.strategy.ETF_POOL.items():
            if ticker in etf_data:
                df = etf_data[ticker]
                
                if latest_date in df.index:
                    score, details = self.strategy.calculate_quality_score(ticker, df, latest_date)
                    scores[ticker] = score
                    score_details_map[ticker] = details
                    latest_prices[ticker] = df.loc[latest_date]
                    
                    print(f"📊 {name} ({ticker})")
                    print(f"   총점: {score}점")
                    for key, value in details.items():
                        print(f"   - {key}: {value}")
                    print()
        
        # 5. 최적 ETF 선택 (Top 1)
        top_tickers = self.strategy.select_top_etfs(scores, n=1)
        
        print("=" * 80)
        print("🎯 거래 신호 (All-In Strategy)")
        print("   오늘의 1순위 대장주 하나만 공략합니다.")
        print("=" * 80)
        print()
        
        if not top_tickers:
            print("❌ 분석 가능한 데이터가 없거나 60점 이상인 종목이 없습니다.")
            print()
            return {
                'signal': 'HOLD',
                'reason': 'Low quality score'
            }
        
        signals = []
        
        for rank, ticker in enumerate(top_tickers, 1):
            score = scores[ticker]
            name = self.strategy.ETF_POOL[ticker]
            latest_price_data = latest_prices[ticker]
            
            # 전일 변동폭 계산
            df = etf_data[ticker]
            latest_idx = df.index.get_loc(latest_date)
            
            if latest_idx > 0:
                prev_data = df.iloc[latest_idx - 1]
                prev_range = prev_data['High'] - prev_data['Low']
            else:
                prev_range = latest_price_data['High'] - latest_price_data['Low']
            
            # 예상 진입가
            reference_entry = latest_price_data['Close'] + (prev_range * self.strategy.k)
            reference_entry = round(reference_entry / 5) * 5
            
            print(f"[{rank}순위] {name} ({ticker})")
            print(f"   품질 점수: {score}점")
            print(f"   기준 종가: {latest_price_data['Close']:,.0f}원")
            print(f"   전일 변동폭: {prev_range:,.0f}원")
            print(f"   예상 진입가: {reference_entry:,.0f}원 (시가 + {prev_range * self.strategy.k:,.0f}원)")
            print("-" * 40)
            
            signals.append({
                'rank': rank,
                'ticker': ticker,
                'name': name,
                'score': score,
                'reference_entry': reference_entry,
                'prev_range': prev_range,
                'score_details': score_details_map[ticker]
            })
            
        print()
        print(f"📋 실전 가이드:")
        print(f"   1. 장 시작 시 1순위({self.strategy.ETF_POOL[top_tickers[0]]}) 자동감시주문 설정")
        print(f"   2. 진입가 도달 시 자동 체결")
        print(f"   3. 나머지는 무시 (집중 투자)")
        print(f"   4. 청산 규칙은 기존과 동일 (갭상승 홀딩, 갭하락 시가청산)")
        print()
        print("=" * 80)
        
        return {
            'signal': 'BUY',
            'targets': signals,
            'latest_date': latest_date.strftime('%Y-%m-%d')
        }
