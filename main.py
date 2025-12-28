from quant_strategy.presentation.visualizer import Visualizer
from quant_strategy.application.services.backtest_engine import BacktestService
from quant_strategy.domain.strategies.macro_hybrid_strategy import MacroHybridStrategy
from quant_strategy.domain.strategies.turtle_strategy import TurtleStrategy
from quant_strategy.domain.strategies.smart_macro_strategy import SmartMacroStrategy

def main():
    print("🚀 Project Darwin: 스마트 매크로(세력방패) vs 매크로 하이브리드 vs 터틀 vs 존버 대결 시작...")
    
    # 전략 준비
    smart_strategy = SmartMacroStrategy(window=60)
    macro_strategy = MacroHybridStrategy(window=60)
    turtle_strategy = TurtleStrategy(buy_period=55, sell_period=20)
    
    # 엔진 생성
    engine_smart = BacktestService(strategy=smart_strategy)
    engine_macro = BacktestService(strategy=macro_strategy)
    engine_turtle = BacktestService(strategy=turtle_strategy)
    
    tickers = [
        "005930.KS", # 삼성전자
        "000660.KS", # SK하이닉스
        "005935.KS", # 삼성전자우
        "042700.KS", # 한미반도체
        "402340.KS", # SK스퀘어
        
        "200710.KS", # 에이디테크놀로지
        "012450.KS", # 한화에어로스페이스
        "196170.KQ", # 알테오젠
        "347850.KQ", # 디앤디파마텍
        "298380.KQ", # 에이비엘바이오
        "087010.KQ", # 펩트론
        "226950.KQ", # 올릭스
    ]
    
    start_date = "2020-01-01"
    end_date = "2024-12-25"
    
    results = []
    
    ticker_names = {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스",
        "005935.KS": "삼성전자우",
        "042700.KS": "한미반도체",
        "402340.KS": "SK스퀘어",
        "200710.KS": "에이디테크놀로지",
        "012450.KS": "한화에어로스페이스",
        "196170.KQ": "알테오젠",
        "347850.KQ": "디앤디파마텍",
        "298380.KQ": "에이비엘바이오",
        "087010.KQ": "펩트론",
        "226950.KQ": "올릭스"
    }

    print("-" * 120)
    print(f"{'종목명':<10} | {'💰스마트':>10} | {'🛡️매크로':>10} | {'🐢터틀':>10} | {'🗿존버':>10} | {'승자':<8}")
    print("-" * 120)
    
    total_capital = 1_200_000_000 
    
    total_smart_final = 0
    total_macro_final = 0
    total_turtle_final = 0
    total_hold_final = 0
    
    for ticker in tickers:
        try:
            # 전략 실행
            df_smart = engine_smart.run(ticker, start_date, end_date)
            df_macro = engine_macro.run(ticker, start_date, end_date)
            df_turtle = engine_turtle.run(ticker, start_date, end_date)
            
            if df_macro.empty or df_turtle.empty or df_smart.empty: 
                continue
                
            final_smart = df_smart['Equity_Strategy'].iloc[-1]
            final_macro = df_macro['Equity_Strategy'].iloc[-1]
            final_turtle = df_turtle['Equity_Strategy'].iloc[-1]
            final_hold = df_macro['Equity_Hold'].iloc[-1]
            
            initial = engine_smart.initial_capital
            
            # 수익률 계산
            ret_smart = (final_smart - initial) / initial * 100
            ret_macro = (final_macro - initial) / initial * 100
            ret_turtle = (final_turtle - initial) / initial * 100
            ret_hold = (final_hold - initial) / initial * 100
            
            # 합산
            total_smart_final += final_smart
            total_macro_final += final_macro
            total_turtle_final += final_turtle
            total_hold_final += final_hold
            
            # 승자 판정
            targets = {'💰스마트': ret_smart, '🛡️매크로': ret_macro, '🐢터틀': ret_turtle, '🗿존버': ret_hold}
            winner = max(targets, key=targets.get)
            
            name = ticker_names.get(ticker, ticker)
            print(f"{name:<10} | {ret_smart:>9.2f}% | {ret_macro:>9.2f}% | {ret_turtle:>9.2f}% | {ret_hold:>9.2f}% | {winner}")
            
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("=" * 120)
    print("📊 [최종 종합 결과 (4파전)]")
    print(f"💰 총 투자 원금 : {total_capital:,.0f} 원")
    print("-" * 60)
    
    ret_total_smart = (total_smart_final - total_capital) / total_capital * 100
    ret_total_macro = (total_macro_final - total_capital) / total_capital * 100
    ret_total_turtle = (total_turtle_final - total_capital) / total_capital * 100
    ret_total_hold = (total_hold_final - total_capital) / total_capital * 100
    
    print(f"🗿 존버 총 자산      : {total_hold_final:,.0f} 원 (수익률: {ret_total_hold:.2f}%)")
    print(f"💰 스마트(방패) 자산 : {total_smart_final:,.0f} 원 (수익률: {ret_total_smart:.2f}%)")
    print(f"🛡️ 매크로 자산      : {total_macro_final:,.0f} 원 (수익률: {ret_total_macro:.2f}%)")
    print(f"🐢 터틀 자산        : {total_turtle_final:,.0f} 원 (수익률: {ret_total_turtle:.2f}%)")
    print("-" * 60)
    
    # 최종 결론
    best_ret = max(ret_total_smart, ret_total_macro, ret_total_turtle, ret_total_hold)
    
    if best_ret == ret_total_hold:
        print("💤 결과: 여전히 '존버'가 최강... (바이오 텐배거의 위엄)")
    elif best_ret == ret_total_smart:
        print("🚀 결과: '스마트 매크로(Whale Shield)' 전략이 승리했습니다! 존버를 이겼습니다!")
    elif best_ret == ret_total_macro:
        print("🏆 결과: '매크로 하이브리드'가 가장 좋습니다.")
    else:
        print("🐢 결과: '터틀'이 이겼습니다.")

if __name__ == "__main__":
    main()