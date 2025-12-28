import pandas as pd
import numpy as np
import yfinance as yf

class MarketDataLoader:
    def __init__(self, start_date="2023-01-01"):
        self.start_date = start_date

    def fetch_data(self, ticker: str) -> pd.DataFrame:
        """
        데이터를 가져오고 보조지표(ATR 등)를 계산하여 반환
        """
        # 1. 야후 파이낸스 데이터 다운로드
        ticker_symbol = f"{ticker}.KS" if not ticker.endswith(".KS") else ticker
        df = yf.download(ticker_symbol, start=self.start_date, progress=False)
        
        if df.empty:
            print(f"⚠️ 데이터 없음: {ticker}")
            return pd.DataFrame()

        # 컬럼 정리 (MultiIndex 문제 방지)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        # yfinance 최신 버전은 컬럼이 튜플일 수 있어 단순화 처리
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # 2. 지표 추가
        df = self._add_atr(df)
        
        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        [ATR 계산 로직]
        - TR = Max(|High-Low|, |High-PrevClose|, |Low-PrevClose|)
        - ATR = TR의 20일 이동평균 (Simple or Exponential)
        * 오리지널 터틀은 20일 사용
        """
        prev_close = df['Close'].shift(1)
        
        # TR 계산 (세 가지 중 최댓값)
        # 1. 고가 - 저가
        tr1 = df['High'] - df['Low']
        # 2. |고가 - 전일종가|
        tr2 = abs(df['High'] - prev_close)
        # 3. |저가 - 전일종가|
        tr3 = abs(df['Low'] - prev_close)
        
        # TR 컬럼 생성
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR(N) 계산
        # 방법 A: 단순 이동평균 (SMA) - 가장 일반적
        df['ATR'] = df['TR'].rolling(window=period).mean()
        
        # 방법 B: Wilder's Smoothing (오리지널 RSI/ATR 창시자 방식) - 더 매끄러움
        # df['ATR'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
        
        # 첫 부분 NaN 제거
        df.dropna(inplace=True)
        
        return df

# ==========================================
# 🧪 테스트 실행
# ==========================================
if __name__ == "__main__":
    loader = MarketDataLoader()
    # 한화에어로스페이스 테스트
    df = loader.fetch_data("012450") 
    
    print(f"📊 데이터 로드 완료: {len(df)} rows")
    print(df[['Close', 'TR', 'ATR']].tail(5))