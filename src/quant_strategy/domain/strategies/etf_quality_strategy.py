"""
ETF 품질 점수 기반 전략
- 7개 일반 ETF 대상
- 품질 점수 ≥ 60점 필터링
- 돌파 매매 + 갭 청산
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from quant_strategy.domain.indicators.technical import (
    calculate_rsi,
    calculate_volume_ratio
)
import tomllib
from pathlib import Path
from quant_strategy.domain.ml.predictor import (
    train_xgboost_model,
    predict_trade_return
)


class EtfQualityStrategy:
    """ETF Quality Score Strategy (Refactored to use TOML config)"""
    
    def __init__(self, config_path: str = "config/strategy.toml"):
        self.models = {}
        self.features = {}
        
        # Load Configuration
        self.config = self._load_config(config_path)
        
        # Apply Config
        self.ETF_POOL = self.config['etf_pool']
        self.QUALITY_THRESHOLD = self.config['strategy']['quality_threshold']
        self.k = self.config['strategy']['k']
        self.exit_strategy = self.config['strategy']['exit_strategy']

    def _load_config(self, path: str) -> Dict:
        """Load configuration from TOML file"""
        try:
            # Try absolute path first, then relative to project root
            file_path = Path(path)
            if not file_path.exists():
                # Fallback: assume running from project root
                file_path = Path.cwd() / path
                
            if not file_path.exists():
                # Fallback 2: hardcoded defaults if file missing (optional, but good for safety)
                print(f"Warning: Config file not found at {path}. Using internal defaults.")
                return {
                    'strategy': {'quality_threshold': 60, 'k': 0.03, 'exit_strategy': 'always_open'},
                    'etf_pool': {
                        '069500.KS': 'KODEX 200', '091160.KS': 'KODEX 반도체', '140700.KS': 'KODEX 금융',
                        '244580.KS': 'KODEX 2차전지K-뉴딜', '091180.KS': 'KODEX 자동차', '117680.KS': 'KODEX 철강',
                        '266390.KS': 'KODEX 미디어&엔터', '132030.KS': 'KODEX 골드선물(H)', '152380.KS': 'KODEX 국채선물10년'
                    }
                }
                
            with open(file_path, "rb") as f:
                return tomllib.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            raise e
    
    def calculate_quality_score(
        self,
        ticker: str,
        df: pd.DataFrame,
        date: pd.Timestamp
    ) -> Tuple[int, Dict[str, str]]:
        """
        ETF 품질 점수 계산 (100점 만점)
        
        Args:
            ticker: ETF 티커
            df: OHLCV 데이터
            date: 평가 날짜
        
        Returns:
            (점수, 상세정보)
        """
        try:
            score = 0
            details = {}
            
            if date not in df.index:
                return 0, {'Error': 'Date not in index'}
            
            # 1. XGBoost 예측 (최대 30점)
            if ticker in self.models:
                pred = predict_trade_return(df, date, self.models[ticker], self.features[ticker])
                if pred >= 1.0:
                    score += 30
                    details['XGBoost'] = f"30점 (예측 {pred:.2f}%)"
                elif pred >= 0.7:
                    score += 20
                    details['XGBoost'] = f"20점 (예측 {pred:.2f}%)"
                elif pred >= 0.4:
                    score += 10
                    details['XGBoost'] = f"10점 (예측 {pred:.2f}%)"
                else:
                    details['XGBoost'] = f"0점 (예측 {pred:.2f}%)"
            else:
                details['XGBoost'] = "0점 (모델 없음)"
            
            # 2. 추세 (최대 20점)
            ma5 = df['Close'].rolling(5).mean()
            ma20 = df['Close'].rolling(20).mean()
            
            if date in ma5.index and date in ma20.index:
                price = df.loc[date, 'Close']
                m5 = ma5.loc[date]
                m20 = ma20.loc[date]
                
                if pd.notna(m5) and pd.notna(m20):
                    if price > m5 > m20:
                        score += 20
                        details['추세'] = "20점 (강세)"
                    elif price > m5:
                        score += 10
                        details['추세'] = "10점 (약세상승)"
                    else:
                        details['추세'] = "0점 (약세)"
                else:
                    details['추세'] = "0점 (데이터 부족)"
            
            # 3. 모멘텀 (최대 20점)
            if date in df.index:
                idx = df.index.get_loc(date)
                if idx >= 5:
                    momentum_5d = (df['Close'].iloc[idx] / df['Close'].iloc[idx-5] - 1) * 100
                    if momentum_5d > 3:
                        score += 20
                        details['모멘텀'] = f"20점 (5일 {momentum_5d:.2f}%)"
                    elif momentum_5d > 1:
                        score += 10
                        details['모멘텀'] = f"10점 (5일 {momentum_5d:.2f}%)"
                    elif momentum_5d > 0:
                        score += 5
                        details['모멘텀'] = f"5점 (5일 {momentum_5d:.2f}%)"
                    else:
                        details['모멘텀'] = f"0점 (5일 {momentum_5d:.2f}%)"
                else:
                    details['모멘텀'] = "0점 (데이터 부족)"
            
            # 4. 거래량 (최대 15점)
            vol_ratio = calculate_volume_ratio(df, 20)
            if date in vol_ratio.index:
                vr = vol_ratio.loc[date]
                if pd.notna(vr):
                    if vr >= 1.5:
                        score += 15
                        details['거래량'] = f"15점 (평균 대비 {vr:.2f}배)"
                    elif vr >= 1.2:
                        score += 10
                        details['거래량'] = f"10점 (평균 대비 {vr:.2f}배)"
                    elif vr >= 1.0:
                        score += 5
                        details['거래량'] = f"5점 (평균 대비 {vr:.2f}배)"
                    else:
                        details['거래량'] = f"0점 (평균 대비 {vr:.2f}배)"
                else:
                    details['거래량'] = "0점 (데이터 부족)"
            
            # 5. RSI (최대 15점)
            rsi = calculate_rsi(df['Close'], 14)
            if date in rsi.index:
                r = rsi.loc[date]
                if pd.notna(r):
                    if 40 < r < 70:
                        score += 15
                        details['RSI'] = f"15점 (RSI {r:.1f})"
                    elif 30 < r < 80:
                        score += 10
                        details['RSI'] = f"10점 (RSI {r:.1f})"
                    else:
                        details['RSI'] = f"0점 (RSI {r:.1f})"
                else:
                    details['RSI'] = "0점 (데이터 부족)"
            
            return score, details
            
        except Exception as e:
            return 0, {'Error': str(e)}
    
    def select_top_etfs(self, etf_scores: Dict[str, int], n: int = 3) -> List[str]:
        """
        상위 N개 ETF 선택
        
        Args:
            etf_scores: {ticker: score} 딕셔너리
            n: 선택할 개수
        
        Returns:
            점수 내림차순 정렬된 티커 리스트
        """
        if not etf_scores:
            return []
        
        # 60점 이상만 필터링
        qualified = {k: v for k, v in etf_scores.items() if v >= self.QUALITY_THRESHOLD}
        
        if not qualified:
            return []
        
        # 점수 내림차순 정렬
        sorted_tickers = sorted(qualified.keys(), key=lambda k: qualified[k], reverse=True)
        return sorted_tickers[:n]

    def select_best_etf(self, etf_scores: Dict[str, int]) -> Optional[str]:
        """
        최고 점수 ETF 선택 (하위 호환성 유지)
        """
        top = self.select_top_etfs(etf_scores, n=1)
        return top[0] if top else None
    
    def calculate_entry_price(self, df: pd.DataFrame, date: pd.Timestamp) -> Optional[float]:
        """
        진입가 계산: 당일 시가 + 전일 변동폭 * 0.03
        
        Args:
            df: OHLCV 데이터
            date: 거래 날짜
        
        Returns:
            진입가 또는 None
        """
        if date not in df.index:
            return None
        
        idx = df.index.get_loc(date)
        if idx == 0:
            return None
        
        # 전일 데이터
        prev_data = df.iloc[idx - 1]
        prev_range = prev_data['High'] - prev_data['Low']
        
        # 당일 시가
        current_open = df.loc[date, 'Open']
        
        # 진입가 계산
        entry_price = current_open + (prev_range * self.k)
        
        # 5원 단위로 반올림
        entry_price = round(entry_price / 5) * 5
        
        return entry_price
    
    def check_entry_signal(
        self,
        df: pd.DataFrame,
        date: pd.Timestamp,
        entry_price: float
    ) -> bool:
        """
        고가 돌파 확인
        
        Args:
            df: OHLCV 데이터
            date: 거래 날짜
            entry_price: 진입가
        
        Returns:
            진입 여부
        """
        if date not in df.index:
            return False
        
        high = df.loc[date, 'High']
        return high >= entry_price
    
    def check_exit_signal(
        self,
        df: pd.DataFrame,
        entry_date: pd.Timestamp,
        exit_date: pd.Timestamp
    ) -> Tuple[bool, Optional[str]]:
        """
        갭 청산 로직
        
        Args:
            df: OHLCV 데이터
            entry_date: 진입 날짜
            exit_date: 청산 날짜 (익일)
        
        Returns:
            (청산 여부, 청산 시점: 'OPEN' or 'CLOSE')
        """
        if exit_date not in df.index or entry_date not in df.index:
            return False, None
        
        entry_close = df.loc[entry_date, 'Close']
        exit_open = df.loc[exit_date, 'Open']
        
        # 갭 계산
        gap = (exit_open / entry_close) - 1
        
        # 청산 전략에 따른 분기
        if self.exit_strategy == 'always_open':
            return True, 'OPEN'
        else:
            # dynamic (기존 로직)
            if gap > 0:
                # 갭 상승 -> 종가에 청산
                return True, 'CLOSE'
            else:
                # 갭 하락 -> 시가에 청산
                return True, 'OPEN'
    
    def train_models(self, etf_data: Dict[str, pd.DataFrame]):
        """
        전체 ETF에 대해 XGBoost 모델 학습
        
        Args:
            etf_data: {ticker: DataFrame} 딕셔너리
        """
        for ticker, df in etf_data.items():
            try:
                model, features, metrics = train_xgboost_model(df, test_size=0.3)
                self.models[ticker] = model
                self.features[ticker] = features
            except Exception as e:
                print(f"Warning: Failed to train model for {ticker}: {e}")
