import pandas as pd
from datetime import datetime, timedelta
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.turtle_strategy import TurtleStrategy

class MarketScanner:
    def __init__(self):
        self.data_loader = MarketDataLoader()
        self.strategy = TurtleStrategy()

    def scan(self, tickers: dict):
        """
        :param tickers: { "종목코드": "종목명" } 형태의 딕셔너리
        """
        results = []
        print(f"📡 총 {len(tickers)}개 종목 스캔 시작...")

        # 스캔 속도를 위해 최근 100일 데이터만 가져옴
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")

        for code, name in tickers.items():
            try:
                # 1. 데이터 가져오기
                df = self.data_loader.fetch_data(code)
                if df.empty: continue
                
                # 최근 데이터만 슬라이싱 (속도 최적화)
                df = df[df.index >= start_date]
                if len(df) < 60: continue # 데이터 부족하면 패스

                # 2. 지표 계산
                df = self.strategy.add_indicators(df)
                
                # 3. 현재 상태 분석 (마지막 날 기준)
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                
                current_price = last_row['Close']
                breakout_price = last_row['Donchian_High'] # 55일 고가 (돌파 기준)
                ma_filter = last_row.get('MA_Filter', 0)   # 60일 이평선
                
                # 신호 확인
                signal = self.strategy.check_signals(last_row, prev_row, has_position=False)
                
                # 돌파까지 남은 거리 (%)
                dist_to_breakout = 0
                if breakout_price > 0:
                    dist_to_breakout = (breakout_price - current_price) / current_price * 100

                status = ""
                # [조건 1] 이미 보유 중이거나 매수 신호 발생
                if signal == 'BUY':
                    status = "🔥 매수 포착"
                # [조건 2] 돌파 임박 (3% 이내)
                elif 0 < dist_to_breakout <= 3.0:
                    status = "👀 관망 (돌파 임박)"
                # [조건 3] 추세는 좋은데 아직 멂
                elif current_price > ma_filter:
                    status = "📈 상승 추세"
                else:
                    status = "📉 하락/횡보"

                results.append({
                    "종목명": name,
                    "종목코드": code,
                    "현재가": current_price,
                    "돌파기준가(55일고가)": f"{breakout_price:,.0f}",
                    "이격도(%)": round(dist_to_breakout, 2),
                    "추세(60일선)": "위" if current_price > ma_filter else "아래",
                    "상태": status
                })
                print(f"[{status}] {name}...", end="\r")

            except Exception as e:
                print(f"에러 ({name}): {e}")
                continue
        
        print("\n✅ 스캔 완료!")
        return pd.DataFrame(results)