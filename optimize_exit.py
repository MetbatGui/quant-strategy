
import sys
import os
import pandas as pd

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy
from quant_strategy.application.services.backtest_engine import BacktestEngine

def run_experiment():
    print("="*60)
    print("🧪 Exit Strategy Optimization Experiment")
    print("="*60)
    
    modes = ['dynamic', 'always_open']
    results = []
    
    start_date = "2025-07-01"
    end_date = "2026-01-05"
    
    for mode in modes:
        print(f"\nrunning backtest with exit_strategy='{mode}'...")
        
        # 전략 초기화
        strategy = EtfQualityStrategy(exit_strategy=mode)
        
        # 엔진 초기화
        engine = BacktestEngine(strategy, initial_capital=10_000_000)
        
        # 실행 (로그는 최소화하기 위해 stdout 임시 차단 가능하지만, 여기선 그냥 둠)
        metrics = engine.run(start_date, end_date)
        
        results.append({
            "Mode": mode,
            "Return": metrics['total_return'],
            "MDD": metrics['max_drawdown'],
            "WinRate": metrics['win_rate'],
            "Trades": metrics['num_trades']
        })

    print("\n" + "="*60)
    print(f"{'Exit Mode':<15} | {'Return':>8} | {'MDD':>8} | {'WinRate':>8} | {'Trades':>6}")
    print("-" * 60)
    
    for res in results:
        print(f"{res['Mode']:<15} | {res['Return']:>7.2f}% | {res['MDD']:>7.2f}% | {res['WinRate']:>7.1f}% | {res['Trades']:>6}")
    print("="*60)

if __name__ == "__main__":
    run_experiment()
