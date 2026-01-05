
import sys
import os
import pandas as pd
from typing import Dict

# Add src to path
sys.path.append(os.path.abspath("src"))

from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy
from quant_strategy.application.services.backtest_engine import BacktestEngine

def run_experiment(k_value: float, etf_pool: Dict[str, str], description: str):
    print(f"\n[{description}] k={k_value}, Pool Size={len(etf_pool)}")
    
    # Setup Strategy
    strategy = EtfQualityStrategy()
    strategy.ENTRY_K = k_value
    strategy.ETF_POOL = etf_pool
    
    # Setup Engine
    engine = BacktestEngine(strategy, initial_capital=10_000_000)
    
    # Run Backtest (Quietly if possible, but our engine prints a lot)
    # We will capture the returned metrics
    metrics = engine.run("2025-11-01", "2026-01-05")
    return metrics

def main():
    # Base Optimized Pool (No Bio, No Kosdaq150)
    base_pool = {
        "069500.KS": "KODEX 200",
        "091160.KS": "KODEX 반도체",
        "091170.KS": "KODEX 금융",
        "091180.KS": "KODEX 자동차",
        "305720.KS": "KODEX 2차전지K-뉴딜",
    }
    
    results = []
    
    # 1. Parameter Sweep for K
    print(">>> EXPERIMENT 1: K-Value Sweep (0.01 ~ 0.05)")
    k_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    for k in k_values:
        metrics = run_experiment(k, base_pool.copy(), f"K={k}")
        results.append({
            "Type": "K-Sweep",
            "Param": k,
            "Return": metrics['total_return'],
            "MDD": metrics['max_drawdown'],
            "WinRate": metrics['win_rate'],
            "Trades": metrics['num_trades']
        })

    # Find best K
    best_k_run = max(results[:5], key=lambda x: x['Return'])
    best_k = best_k_run['Param']
    print(f"\n*** Best K found: {best_k} (Return: {best_k_run['Return']:.2f}%) ***")
    
    # 2. ETF Pool Expansion (using Best K)
    print(f"\n>>> EXPERIMENT 2: Pool Expansion (using K={best_k})")
    
    # Candidates
    pool_steel = base_pool.copy()
    pool_steel["117680.KS"] = "KODEX 철강"
    
    pool_media = base_pool.copy()
    pool_media["266390.KS"] = "KODEX 미디어&엔터"
    
    pool_both = base_pool.copy()
    pool_both["117680.KS"] = "KODEX 철강"
    pool_both["266390.KS"] = "KODEX 미디어&엔터"
    
    experiments = [
        ("Add Steel", pool_steel),
        ("Add Media", pool_media),
        ("Add Both", pool_both)
    ]
    
    for name, pool in experiments:
        metrics = run_experiment(best_k, pool, name)
        results.append({
            "Type": "Pool-Exp",
            "Param": name,
            "Return": metrics['total_return'],
            "MDD": metrics['max_drawdown'],
            "WinRate": metrics['win_rate'],
            "Trades": metrics['num_trades']
        })
        
    # Simplify Output
    print("\n" + "="*60)
    print(f"{'Experiment':<20} | {'Return':>8} | {'MDD':>8} | {'WinRate':>8} | {'Trades':>6}")
    print("-" * 60)
    for res in results:
        label = f"{res['Type']}: {res['Param']}"
        print(f"{label:<20} | {res['Return']:>7.2f}% | {res['MDD']:>7.2f}% | {res['WinRate']:>7.1f}% | {res['Trades']:>6}")
    print("="*60)

if __name__ == "__main__":
    main()
