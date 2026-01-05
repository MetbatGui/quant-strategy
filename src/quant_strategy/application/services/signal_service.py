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
        
        # 5. 최적 ETF 선택
        best_ticker = self.strategy.select_best_etf(scores)
        
        print("=" * 80)
        print("🎯 거래 신호")
        print("=" * 80)
        print()
        
        if best_ticker is None:
            max_ticker = max(scores, key=scores.get) if scores else None
            if max_ticker:
                print(f"⚠️  최고 점수 종목: {self.strategy.ETF_POOL[max_ticker]} ({scores[max_ticker]}점)")
                print(f"   품질 점수가 {self.strategy.QUALITY_THRESHOLD}점 미만이므로 거래 보류 권장")
            else:
                print("❌ 분석 가능한 데이터가 없습니다.")
            print()
            return {
                'signal': 'HOLD',
                'reason': 'Low quality score'
            }
        
        best_score = scores[best_ticker]
        best_name = self.strategy.ETF_POOL[best_ticker]
        latest_price_data = latest_prices[best_ticker]
        
        # 6. 진입가 계산
        df = etf_data[best_ticker]
        
        # 다음 거래일 예상 (최신일 다음)
        future_dates = all_dates[all_dates > latest_date]
        if len(future_dates) == 0:
            next_date = latest_date + timedelta(days=1)
        else:
            next_date = future_dates[0]
        
        entry_price = self.strategy.calculate_entry_price(df, latest_date)
        
        if entry_price is None:
            # 최신일이 첫날인 경우 간단 계산
            estimated_open = latest_price_data['Close']
            range_val = latest_price_data['High'] - latest_price_data['Low']
            entry_price = estimated_open + (range_val * self.strategy.ENTRY_K)
            entry_price = round(entry_price / 5) * 5
        
        print(f"✅ 선정 종목: {best_name}")
        print(f"   티커: {best_ticker}")
        print(f"   품질 점수: {best_score}점")
        print()
        print(f"📈 최신 거래일({latest_date.strftime('%Y-%m-%d')}) 종가: {latest_price_data['Close']:,.0f}원")
        print()
        print(f"🎯 다음 거래일 진입 전략:")
        print(f"   예상 날짜: {next_date.strftime('%Y-%m-%d')}")
        print(f"   진입가(시가+변동폭3%): {entry_price:,.0f}원")
        print()
        print(f"💡 실전 가이드:")
        print(f"   1. 다음 거래일 시가 확인")
        print(f"   2. 당일 고가가 {entry_price:,.0f}원 돌파 시 매수")
        print(f"   3. 익일 갭 상승 시 익일 종가에 청산")
        print(f"   4. 익일 갭 하락 시 익일 시가에 청산")
        print()
        print("=" * 80)
        
        return {
            'signal': 'BUY',
            'selected_etf': best_ticker,
            'etf_name': best_name,
            'quality_score': best_score,
            'entry_price': entry_price,
            'latest_close': latest_price_data['Close'],
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'next_date': next_date.strftime('%Y-%m-%d'),
            'score_details': score_details_map[best_ticker]
        }
