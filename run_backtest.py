import sys
import os
import argparse
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath("src"))

from quant_strategy.presentation.cli.main import backtest_command

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run backtest with custom date range.')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)', default=None)
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)', default=None)
    parser.add_argument('--train-start', type=str, help='Training start date (YYYY-MM-DD)', default=None)
    
    args = parser.parse_args()
    
    # Default to recent 2 months if not specified
    if not args.end:
        args.end = datetime.now().strftime('%Y-%m-%d')
    if not args.start:
        # Default to 2 months ago
        end_dt = datetime.strptime(args.end, '%Y-%m-%d')
        args.start = (end_dt - timedelta(days=60)).strftime('%Y-%m-%d')

    print(f"Running backtest for {args.start} to {args.end}")
    
    try:
        backtest_command(start=args.start, end=args.end, train_start=args.train_start)
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
