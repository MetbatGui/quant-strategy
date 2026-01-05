
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from quant_strategy.presentation.cli.main import backtest_command

if __name__ == "__main__":
    print("Running backtest for 2025-11-01 to 2026-01-05")
    try:
        backtest_command(start="2025-11-01", end="2026-01-05")
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
