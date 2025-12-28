import pandas as pd
from quant_strategy.application.services.backtest_engine import BacktestService

def main():
    # 1. 설정
    start_date = "20250101"
    end_date = "20251227"
    initial_capital_per_stock = 100_000_000 
    risk_pct = 0.02

    # 2. 분석 대상 종목 (12선)
    target_tickers = {
        "005930": "삼성전자", "000660": "SK하이닉스", "005935": "삼성전자우",
        "042700": "한미반도체", "402340": "SK스퀘어", "200710": "에이디테크놀로지",
        "012450": "한화에어로스페이스",
        "196170": "알테오젠", "347850": "디앤디파마텍", "298380": "에이비엘바이오",
        "087010": "펩트론", "226950": "올릭스"
    }

    backtest_service = BacktestService(initial_capital=initial_capital_per_stock, risk_pct=risk_pct)
    
    # 누적 변수
    total_invested = 0
    total_strat_equity = 0 # 터틀 전략 총 자산
    total_hold_equity = 0  # 단순 보유 총 자산

    print(f"🚀 Project Darwin: 전략 vs 존버 대결 시작...")
    print("-" * 85)
    print(f"{'종목명':<10} | {'터틀 수익':>9} | {'존버 수익':>9} | {'승자':<6} | {'비고'}")
    print("-" * 85)

    for code, name in target_tickers.items():
        try:
            df = backtest_service.run(code, start_date, end_date)
            if df.empty: continue

            # 1. 터틀 결과
            final_strat = df['Equity_Strategy'].iloc[-1]
            ror_strat = (final_strat - initial_capital_per_stock) / initial_capital_per_stock * 100
            
            # 2. 단순 보유(존버) 결과
            final_hold = df['Equity_Hold'].iloc[-1]
            ror_hold = (final_hold - initial_capital_per_stock) / initial_capital_per_stock * 100

            # 3. 합산
            total_invested += initial_capital_per_stock
            total_strat_equity += final_strat
            total_hold_equity += final_hold

            # 4. 비교 출력
            winner = "🐢터틀" if ror_strat > ror_hold else "🗿존버"
            diff = ror_strat - ror_hold # 알파(초과수익)
            
            # 비고란에 차이 표시
            note = f"(+{diff:>5.1f}%)" if diff > 0 else f"({diff:>5.1f}%)"
            
            print(f"{name:<10} | {ror_strat:>8.2f}% | {ror_hold:>8.2f}% | {winner:<6} | {note}")

        except Exception as e:
            print(f"⚠️ 에러 ({name}): {e}")

    # 최종 결과 비교
    total_strat_ror = (total_strat_equity - total_invested) / total_invested * 100
    total_hold_ror = (total_hold_equity - total_invested) / total_invested * 100

    print("=" * 85)
    print("📊 [최종 승자 판정]")
    print(f"💰 총 투자 원금 : {total_invested:,.0f} 원")
    print("-" * 40)
    print(f"🗿 단순 보유 총 자산 : {total_hold_equity:,.0f} 원 (수익률: {total_hold_ror:.2f}%)")
    print(f"🐢 터틀 전략 총 자산 : {total_strat_equity:,.0f} 원 (수익률: {total_strat_ror:.2f}%)")
    print("-" * 40)
    
    alpha = total_strat_equity - total_hold_equity
    if alpha > 0:
        print(f"🏆 결과: 터틀 전략이 '{alpha:,.0f}원' 더 벌었습니다! (Alpha: +{total_strat_ror - total_hold_ror:.2f}%)")
    else:
        print(f"💤 결과: 그냥 들고 있는 게 '{abs(alpha):,.0f}원' 더 나았습니다. (채찍효과 비용 발생)")

if __name__ == "__main__":
    main()