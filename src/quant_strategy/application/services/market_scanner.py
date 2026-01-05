import pandas as pd
from datetime import datetime, timedelta
from quant_strategy.infrastructure.data_loader import MarketDataLoader
from quant_strategy.domain.strategies.smart_ai_strategy import SmartAiStrategy

class MarketScanner:
    def __init__(self):
        self.data_loader = MarketDataLoader()
        self.strategy = SmartAiStrategy() # Defaults to XGB V3

    def scan(self, tickers: dict):
        """
        :param tickers: { "종목코드": "종목명" } 형태의 딕셔너리
        """
        results = []
        print(f"📡 총 {len(tickers)}개 종목 스캔 시작 (AI Hybrid V3)...")

        # 스캔 속도를 위해 최근 1년 데이터만 가져옴
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # 0. 매크로 데이터 가져오기 (필수)
        print("🌍 거시경제 지표 수집 중...")
        try:
             macro_df = self.data_loader.fetch_macro_data((datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d"), end_date)
        except:
             macro_df = pd.DataFrame()

        for code, name in tickers.items():
            try:
                # 1. 데이터 가져오기
                df = self.data_loader.fetch_data(code)
                if df.empty: continue
                
                # 최근 데이터만 슬라이싱
                if len(df) > 365:
                    df = df[df.index >= start_date]
                
                if len(df) < 100: continue # 데이터 부족하면 패스

                # 2. 지표 계산 (매크로 포함)
                try:
                    df = self.strategy.add_indicators(df, macro_df)
                except:
                    df = self.strategy.add_indicators(df)
                
                # 3. 현재 상태 분석 (마지막 날 기준)
                if len(df) < 2: continue
                last_row = df.iloc[-1]
                prev_row = df.iloc[-2]
                
                current_price = last_row['Close']
                
                # 신호 확인 (Smart Whale Rule + AI Filter)
                # Position is assumed False for scanning "New Entries"
                try:
                    signal = self.strategy.check_signals(last_row, prev_row, has_position=False)
                except:
                    signal = 'HOLD'
                
                status = ""
                # AI Score logic is hidden, but result implies it.
                if signal == 'BUY':
                    status = "🔥 강력 매수 (AI 승인)"
                else:
                    # 보조 분석: Rule이 떴는데 AI가 막았는지 확인
                    ma20 = last_row.get('MA20', 0)
                    ma60 = last_row.get('MA60', 0)
                    prev_ma20 = prev_row.get('MA20', 0)
                    prev_ma60 = prev_row.get('MA60', 0)
                    
                    is_golden = (prev_ma20 <= prev_ma60) and (ma20 > ma60)
                    
                    if is_golden:
                        status = "⚠️ Rule 매수 -> AI 거절"
                    elif ma20 > ma60:
                        status = "📈 상승 추세"
                    else:
                        status = "📉 하락/횡보"

                results.append({
                    "종목명": name,
                    "종목코드": code,
                    "현재가": current_price,
                    "추세": "상승" if ma20 > ma60 else "하락",
                    "상태": status,
                    "Date": last_row.name.strftime("%Y-%m-%d")
                })
                print(f"[{status}] {name}...", end="\r")

            except Exception as e:
                # print(f"에러 ({name}): {e}")
                continue
        
        print("\n✅ 스캔 완료!")
        return pd.DataFrame(results)