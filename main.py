import pandas as pd
from quant_strategy.presentation.visualizer import Visualizer
from quant_strategy.application.services.backtest_engine import BacktestService
from quant_strategy.domain.strategies.macro_hybrid_strategy import MacroHybridStrategy
from quant_strategy.domain.strategies.turtle_strategy import TurtleStrategy

def main():
    print("🚀 Project Darwin: 매크로 하이브리드 vs 터틀 vs 존버 대결 시작...")
    
    # 두 전략 준비
    macro_strategy = MacroHybridStrategy(window=60)
    # 터틀: 진입 55일, 청산 20일 (Donchian) - window 인자 아님
    turtle_strategy = TurtleStrategy(buy_period=55, sell_period=20)
    
    # 엔진 2개 생성 (각 전략용)
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

    print("-" * 100)
    print(f"{'종목명':<10} | {'매크로 수익':>10} | {'터틀 수익':>10} | {'존버 수익':>10} | {'승자':<8}")
    print("-" * 100)
    
    total_capital = 1_200_000_000 # 12억
    per_ticker_capital = total_capital / len(tickers) # 1억씩
    
    total_macro_final = 0
    total_turtle_final = 0
    total_hold_final = 0
    
    for ticker in tickers:
        try:
            # 1. 매크로 전략 실행
            df_macro = engine_macro.run(ticker, start_date, end_date)
            # 2. 터틀 전략 실행
            df_turtle = engine_turtle.run(ticker, start_date, end_date)
            
            if df_macro.empty or df_turtle.empty: 
                continue
                
            final_macro = df_macro['Equity_Strategy'].iloc[-1]
            final_turtle = df_turtle['Equity_Strategy'].iloc[-1]
            final_hold = df_macro['Equity_Hold'].iloc[-1]
            
            initial = engine_macro.initial_capital # 1억
            
            # 수익률 계산
            ret_macro = (final_macro - initial) / initial * 100
            ret_turtle = (final_turtle - initial) / initial * 100
            ret_hold = (final_hold - initial) / initial * 100
            
            # 최종 자산 합산
            total_macro_final += final_macro
            total_turtle_final += final_turtle
            total_hold_final += final_hold
            
            winner = "🗿존버"
            if ret_macro > ret_hold and ret_macro > ret_turtle:
                winner = "🛡️매크로"
            elif ret_turtle > ret_hold and ret_turtle > ret_macro:
                winner = "🐢터틀"
                
            name = ticker_names.get(ticker, ticker)
            print(f"{name:<10} | {ret_macro:>9.2f}% | {ret_turtle:>9.2f}% | {ret_hold:>9.2f}% | {winner}")
            
        except Exception as e:
            print(f"Error {ticker}: {e}")

    print("=" * 100)
    print("📊 [최종 종합 결과]")
    print(f"💰 총 투자 원금 : {total_capital:,.0f} 원")
    print("-" * 50)
    
    ret_total_macro = (total_macro_final - total_capital) / total_capital * 100
    ret_total_turtle = (total_turtle_final - total_capital) / total_capital * 100
    ret_total_hold = (total_hold_final - total_capital) / total_capital * 100
    
    print(f"🗿 존버 총 자산   : {total_hold_final:,.0f} 원 (수익률: {ret_total_hold:.2f}%)")
    print(f"🛡️ 매크로 총 자산 : {total_macro_final:,.0f} 원 (수익률: {ret_total_macro:.2f}%)")
    print(f"🐢 터틀 총 자산   : {total_turtle_final:,.0f} 원 (수익률: {ret_total_turtle:.2f}%)")
    print("-" * 50)
    
    # 최종 결론
    best = max(ret_total_macro, ret_total_turtle, ret_total_hold)
    if best == ret_total_hold:
        print("💤 결과: 아직은 '존버'가 최강입니다.")
    elif best == ret_total_macro:
        print("🏆 결과: '매크로 하이브리드' 전략이 승리했습니다! (안정성+수익성)")
    else:
        print("🐢 결과: '터틀' 전략이 승리했습니다! (추세 추종의 승리)")

if __name__ == "__main__":
    main()