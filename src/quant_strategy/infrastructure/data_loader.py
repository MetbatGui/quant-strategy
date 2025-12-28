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
        # 티커 처리: 이미 .KS나 .KQ가 붙어있으면 그대로 사용, 없으면 .KS 붙임
        if not (ticker.endswith('.KS') or ticker.endswith('.KQ')):
            ticker_symbol = f"{ticker}.KS"
        else:
            ticker_symbol = ticker
            
        df = yf.download(ticker_symbol, start=self.start_date, progress=False)
        
        if df.empty:
            print(f"⚠️ 데이터 없음: {ticker}")
            return pd.DataFrame()

        # 컬럼 정리 (MultiIndex 문제 방지)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        # yfinance 최신 버전은 컬럼이 튜플일 수 있어 단순화 처리
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        # ----------------------------------------------------
        # [NEW] 수급 데이터 병합 (Foreigner, Institution)
        # ----------------------------------------------------
        try:
            df = self._merge_investor_data(df, ticker)
        except Exception as e:
            print(f"⚠️ 수급 데이터 병합 실패 ({ticker}): {e}")
            df['Foreigner'] = 0
            df['Institution'] = 0

        # 2. 지표 추가
        # (1) 이동평균선
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        # (2) ATR (기존 메소드 대신 직접 계산)
        df = self._add_atr(df)

        # (3) RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # (4) MFI (Money Flow Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        pos_flow = pd.Series(0.0, index=df.index)
        neg_flow = pd.Series(0.0, index=df.index)
        
        price_diff = typical_price.diff()
        pos_flow[price_diff > 0] = money_flow[price_diff > 0]
        neg_flow[price_diff < 0] = money_flow[price_diff < 0]
        
        pos_mf_sum = pos_flow.rolling(window=14).sum()
        neg_mf_sum = neg_flow.rolling(window=14).sum()
        
        mfi_ratio = pos_mf_sum / neg_mf_sum
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # (5) MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # (6) Bollinger Bands
        bb_period = 20
        bb_std = df['Close'].rolling(window=bb_period).std()
        df['BB_Mid'] = df['Close'].rolling(window=bb_period).mean()
        df['BB_Upper'] = df['BB_Mid'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (bb_std * 2)
        
        # NaN 제거
        df = df.dropna()
        
        return df

    def _merge_investor_data(self, ohlcv_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        from pykrx import stock
        
        if ohlcv_df.empty: return ohlcv_df
        
        # 날짜 범위 설정
        start_date = ohlcv_df.index[0].strftime("%Y%m%d")
        end_date = ohlcv_df.index[-1].strftime("%Y%m%d")
        
        # pykrx 티커 변환 (yfinance 티커에서 숫자만 추출)
        code = ''.join(filter(str.isdigit, ticker))
        
        try:
            # 일별 거래실적 (기관/외국인)
            # 주의: pykrx의 날짜는 'YYYYMMDD' 문자열이 아니라 datetime 인덱스로 나옴
            investor_df = stock.get_market_trading_value_by_date(start_date, end_date, code)
            
            # 컬럼 매핑 (pykrx 버전에 따라 컬럼명이 다를 수 있음, 안전하게 확인)
            # 보통: '외국인합계', '기관합계', '기타법인', '개인', '전체'
            # 영문 환경일 경우: 'Foreigner', 'Institution' 등
            
            # 한국어 컬럼명을 영문으로 변경
            rename_map = {
                '외국인합계': 'Foreigner',
                '기관합계': 'Institution',
                '외국인': 'Foreigner', # 버전에 따라 다를 수 있음
                '기관': 'Institution'
            }
            investor_df = investor_df.rename(columns=rename_map)
            
            # 필요한 컬럼만 추출 (없으면 생성)
            if 'Foreigner' not in investor_df.columns: investor_df['Foreigner'] = 0
            if 'Institution' not in investor_df.columns: investor_df['Institution'] = 0
            
            investor_df = investor_df[['Foreigner', 'Institution']]
            
            # 인덱스 타임존 처리 (yfinance는 tz-aware일 수 있음)
            # ohlcv_df.index가 tz-aware인지 확인
            if ohlcv_df.index.tz is not None:
                # investor_df 인덱스를 tz-aware로 변환 (UTC+9 등 적절히, 여기서는 단순하게 ohlcv와 맞춤)
                if investor_df.index.tz is None:
                    investor_df.index = investor_df.index.tz_localize(ohlcv_df.index.tz)
            else:
                if investor_df.index.tz is not None:
                    investor_df.index = investor_df.index.tz_localize(None)

            # 병합 (Left Join)
            merged = ohlcv_df.join(investor_df, how='left').fillna(0)
            return merged

        except Exception as e:
            print(f"⚠️ Investor data fetch failed: {e}")
            # 실패 시 0으로 채워서 반환
            ohlcv_df['Foreigner'] = 0
            ohlcv_df['Institution'] = 0
            return ohlcv_df
    
    def fetch_macro_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        나스닥(^NDX) 및 환율(KRW=X) 데이터를 가져옴
        """
        try:
            tickers = ['^NDX', 'KRW=X'] # 나스닥100, 원달러환율
            macro_df = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # MultiIndex 컬럼 처리
            if isinstance(macro_df.columns, pd.MultiIndex):
                # 'Close' 레벨만 가져오기
                close_df = macro_df['Close'].copy()
            else:
                close_df = macro_df.copy()
            
            # 컬럼명 정리
            # 다운로드된 컬럼이 symbol 이름으로 되어있음
            # 예: KRW=X, ^NDX
            
            # 날짜 정렬 및 결측치 처리 (ffill)
            close_df = close_df.ffill()
            
            return close_df
        except Exception as e:
            print(f"⚠️ Macro data fetch failed: {e}")
            return pd.DataFrame()

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