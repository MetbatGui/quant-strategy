"""
퀀트 전략 CLI
"""

def backtest_command(start: str = "2024-01-01", end: str = "2025-01-01", train_start: str = None):
    """백테스트 실행"""
    from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy
    from quant_strategy.application.services.backtest_engine import BacktestEngine
    
    strategy = EtfQualityStrategy()
    engine = BacktestEngine(strategy, initial_capital=10_000_000)
    
    result = engine.run(start, end, train_start)
    
    return result


def signal_command(date: str = None):
    """일일 거래 신호 생성"""
    from quant_strategy.domain.strategies.etf_quality_strategy import EtfQualityStrategy
    from quant_strategy.application.services.signal_service import SignalService
    
    strategy = EtfQualityStrategy()
    service = SignalService(strategy)
    
    signal = service.generate_daily_signal(date)
    
    return signal


def main():
    """CLI 메인 엔트리포인트"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: quant [command] [options]")
        print()
        print("Commands:")
        print("  backtest [--start YYYY-MM-DD] [--end YYYY-MM-DD]  백테스트 실행")
        print("  signal [--date YYYY-MM-DD]                        일일 신호 생성")
        print()
        print("Examples:")
        print("  quant backtest --start 2025-11-01 --end 2026-01-05")
        print("  quant signal --date 2026-01-06")
        print("  quant signal")
        return
    
    command = sys.argv[1]
    
    if command == "backtest":
        start = "2024-01-01"
        end = "2025-01-01"
        
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--start" and i + 3 < len(sys.argv):
                start = sys.argv[i + 3]
            elif arg == "--end" and i + 3 < len(sys.argv):
                end = sys.argv[i + 3]
        
        backtest_command(start, end)
    
    elif command == "signal":
        date = None
        
        for i, arg in enumerate(sys.argv[2:]):
            if arg == "--date" and i + 3 < len(sys.argv):
                date = sys.argv[i + 3]
        
        signal_command(date)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'quant' without arguments to see available commands.")


if __name__ == "__main__":
    main()
