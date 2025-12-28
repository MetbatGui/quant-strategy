from quant_strategy.presentation.visualizer import Visualizer
from quant_strategy.application.services.backtest_engine import BacktestService
from quant_strategy.domain.strategies.turtle_strategy import TurtleStrategy
from quant_strategy.domain.strategies.smart_whale_strategy import SmartWhaleStrategy

def main():
    print("🚀 Project Darwin: 스마트 웨일(광역 검증) vs 터틀 vs 존버...")
    print("🧪 검증 대상: 우량주/주도주 30종목 (반도체, 2차전지, 바이오, 방산, 로봇, 엔터 등)")
    
    # 전략 준비
    smart_strategy = SmartWhaleStrategy(window=60)
    turtle_strategy = TurtleStrategy(buy_period=55, sell_period=20)
    
    # 엔진 생성
    engine_smart = BacktestService(strategy=smart_strategy)
    engine_turtle = BacktestService(strategy=turtle_strategy)
    
    # === [광역 검증 유니버스 30] ===
    tickers = [
        # 1. 반도체 (Semiconductor)
        "005930.KS", # 삼성전자
        "000660.KS", # SK하이닉스
        "042700.KS", # 한미반도체 (HBM 대장)
        "005830.KS", # DB하이텍
        "084370.KQ", # 유진테크 (소부장)

        # 2. 2차전지 (Battery) - 변동성 극심
        "373220.KS", # LG에너지솔루션
        "006400.KS", # 삼성SDI
        "051910.KS", # LG화학
        "003670.KQ", # 포스코퓨처엠
        "247540.KQ", # 에코프로비엠 (코스닥 대장)
        "086520.KQ", # 에코프로 (광기 종목)
        "402340.KS", # SK스퀘어 (투자)

        # 3. 자동차/모빌리티 (Auto)
        "005380.KS", # 현대차
        "000270.KS", # 기아
        "012330.KS", # 현대모비스

        # 4. 바이오/헬스케어 (Bio)
        "207940.KS", # 삼성바이오로직스
        "068270.KS", # 셀트리온
        "028300.KQ", # HLB (FDA 이슈)
        "196170.KQ", # 알테오젠 (플랫폼 기술수출)
        "087010.KQ", # 펩트론 (비만치료제)

        # 5. 인터넷/게임/엔터 (Platform/Ent)
        "035420.KS", # NAVER
        "035720.KS", # 카카오
        "352820.KS", # 하이브 (BTS)
        "251270.KQ", # 넷마블

        # 6. 방산/중공업/원전 (Heavy Industry)
        "012450.KS", # 한화에어로스페이스 (방산)
        "047810.KS", # 한국항공우주 (KAI)
        "010120.KS", # LSELECTRIC (전력설비)
        "034020.KS", # 두산에너빌리티 (원전)

        # 7. 로봇/AI (Future)
        "277810.KQ", # 레인보우로보틱스
        "403870.KQ", # HPSP (반도체 고압수소)
    ]
    
    start_date = "2020-01-01"
    end_date = "2024-12-25"
    
    # 결과 저장
    total_capital = 1_000_000_000 
    
    total_smart_final = 0
    total_turtle_final = 0
    total_hold_final = 0
    
    ticker_names = {
        "005930.KS": "삼성전자", "000660.KS": "SK하이닉스", "042700.KS": "한미반도체", "005830.KS": "DB하이텍", "084370.KQ": "유진테크",
        "373220.KS": "LG엔솔", "006400.KS": "삼성SDI", "051910.KS": "LG화학", "003670.KQ": "포스코퓨처", "247540.KQ": "에코프로BM", "086520.KQ": "에코프로", "402340.KS": "SK스퀘어",
        "005380.KS": "현대차", "000270.KS": "기아", "012330.KS": "현대모비스",
        "207940.KS": "삼바", "068270.KS": "셀트리온", "028300.KQ": "HLB", "196170.KQ": "알테오젠", "087010.KQ": "펩트론",
        "035420.KS": "NAVER", "035720.KS": "카카오", "352820.KS": "하이브", "251270.KQ": "넷마블",
        "012450.KS": "한화에어로", "047810.KS": "KAI", "010120.KS": "LSELEC", "034020.KS": "두산에너빌",
        "277810.KQ": "레인보우", "403870.KQ": "HPSP"
    }

    print("-" * 100)
    print(f"{'종목명':<10} | {'🚀웨일':>10} | {'🐢터틀':>10} | {'🗿존버':>10} | {'승자':<8}")
    print("-" * 100)
    
    for ticker in tickers:
        try:
            # 전략 실행
            df_smart = engine_smart.run(ticker, start_date, end_date)
            df_turtle = engine_turtle.run(ticker, start_date, end_date)
            
            if df_smart.empty: 
                print(f"{ticker_names.get(ticker, ticker)}: No Data")
                continue
                
            # 존버 데이터는 공통
            final_hold = df_smart['Equity_Hold'].iloc[-1]
            final_smart = df_smart['Equity_Strategy'].iloc[-1]
            final_turtle = df_turtle['Equity_Strategy'].iloc[-1]
            
            initial = engine_smart.initial_capital
            
            # 수익률 계산
            ret_smart = (final_smart - initial) / initial * 100
            ret_turtle = (final_turtle - initial) / initial * 100
            ret_hold = (final_hold - initial) / initial * 100
            
            # 합산
            total_smart_final += final_smart
            total_turtle_final += final_turtle
            total_hold_final += final_hold
            
            # 승자 판정
            targets = {'🚀웨일': ret_smart, '🐢터틀': ret_turtle, '🗿존버': ret_hold}
            winner = max(targets, key=targets.get)
            
            name = ticker_names.get(ticker, ticker)
            print(f"{name:<10} | {ret_smart:>9.2f}% | {ret_turtle:>9.2f}% | {ret_hold:>9.2f}% | {winner}")
            
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("=" * 100)
    print("📊 [광역 검증 최종 결과 (30종목)]")
    # 원금은 종목 수만큼 비례해서 가정 (단순 합산 수익률 비교를 위함이므로)
    total_capital = 1_000_000_000 * len(tickers) # 10억씩 30종목 투자 가정
    
    # 실제로는 위 루프에서 total_capital을 누적하지 않았음. 
    # BacktestService는 매번 10억으로 시작함.
    # 따라서 총 자산은 (30 * 10억) + 총 이익
    
    net_profit_smart = total_smart_final - total_capital
    net_profit_turtle = total_turtle_final - total_capital
    net_profit_hold = total_hold_final - total_capital
    
    ret_total_smart = (net_profit_smart) / total_capital * 100
    ret_total_turtle = (net_profit_turtle) / total_capital * 100
    ret_total_hold = (net_profit_hold) / total_capital * 100
    
    print(f"💰 총 포트폴리오 규모 : {total_capital/100000000:,.0f} 억 원")
    print("-" * 60)
    print(f"🗿 존버 총 수익률    : {ret_total_hold:.2f}%")
    print(f"🚀 웨일 총 수익률    : {ret_total_smart:.2f}%")
    print(f"🐢 터틀 총 수익률    : {ret_total_turtle:.2f}%")
    print("-" * 60)
    
    # 최종 결론
    best_ret = max(ret_total_smart, ret_total_turtle, ret_total_hold)
    
    print("📋 [종합 평가]")
    if best_ret == ret_total_smart:
        print("🏆 Winner: Smart Whale! (시장 전체를 이겼습니다)")
    elif best_ret == ret_total_hold:
        print("🏆 Winner: Buy & Hold (아직도 존버가 강한가?)")
        if ret_total_smart > ret_total_hold * 0.8:
            print("👉 But Smart Whale is close! (리스크 대비 훌륭함)")
    else:
        print("🏆 Winner: Turtle (의외의 결과)")

if __name__ == "__main__":
    main()