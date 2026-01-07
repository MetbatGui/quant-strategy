"""
Backtest Result Domain Entity
"""

import json
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from quant_strategy.domain.entities.trade import Trade

@dataclass
class BacktestResult:
    """백테스트 결과 컨테이너"""
    
    # Metadata
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    
    # Metrics
    total_return: float
    cagr: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    max_drawdown: float = 0.0
    average_return: float = 0.0
    
    # Data
    trades: List[Trade] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self):
        return (
            f"=== Backtest Result ===\n"
            f"Period: {self.start_date} ~ {self.end_date}\n"
            f"Return: {self.total_return:.2f}%\n"
            f"Win Rate: {self.win_rate:.1f}% ({self.num_trades} trades)\n"
            f"MDD: {self.max_drawdown:.2f}%\n"
            f"Final Capital: {self.final_capital:,.0f}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, output_dir: str = "reports"):
        """Save result to files"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Summary JSON
        summary_file = path / f"backtest_summary_{timestamp}.json"
        
        # Datetime serialization helper
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            raise TypeError (f"Type {type(obj)} not serializable")

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, default=json_serial, indent=2, ensure_ascii=False)
            
        # 2. Trades CSV
        trades_file = path / f"backtest_trades_{timestamp}.csv"
        
        if self.trades:
            with open(trades_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                # Header
                header = [
                    'Entry Date', 'Ticker', 'Name', 'Action', 'Entry Price', 
                    'Exit Date', 'Exit Price', 'Return(%)', 'Score', 
                    'Best Alt. Name', 'Best Alt. Return(%)'
                ]
                writer.writerow(header)
                
                for t in self.trades:
                    writer.writerow([
                        t.entry_date.strftime('%Y-%m-%d'),
                        t.ticker,
                        t.ticker_name,
                        'LONG',
                        t.entry_price,
                        t.exit_date.strftime('%Y-%m-%d') if t.exit_date else '',
                        t.exit_price if t.exit_price else '',
                        f"{t.position_return:.2f}" if t.position_return is not None else '',
                        t.quality_score,
                        t.best_alternative_name if t.best_alternative_name else '',
                        f"{t.best_alternative_return:.2f}" if t.best_alternative_return is not None else ''
                    ])
                    
        # 3. HTML Report (Plotly)
        html_path = path / f"backtest_report_{timestamp}.html"
        try:
            self._generate_html_report(html_path)
        except Exception as e:
            print(f"❌ Failed to generate HTML report: {e}")
            html_path = None
            
        return {
            'summary': summary_file,
            'trades': trades_file,
            'html': html_path
        }

    def _generate_html_report(self, output_path: Path):
        """Generate interactive HTML report using Plotly"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Prepare Data
        dates = []
        equity = [self.initial_capital]
        current_equity = self.initial_capital
        
        # Reconstruct daily equity curve (approximation based on closed trades)
        # For a more accurate curve, we would need daily value history from BacktestEngine.
        # Here we map trade exits to dates for visualization.
        
        trade_data = []
        for t in self.trades:
            d = t.__dict__.copy()
            d['position_return'] = t.position_return
            trade_data.append(d)
            
        trade_df = pd.DataFrame(trade_data)
        if trade_df.empty:
            return

        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
        trade_df = trade_df.sort_values('exit_date')
        
        cumulative_return = 0
        equity_curve = []
        
        # Simplified Equity Curve: Step function based on trade close
        # (Ideal would be daily mark-to-market, but this suffices for trade analysis)
        
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            subplot_titles=('Cumulative Return (%)', 'Drawdown (%)', 'Trade Return (%)'),
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Calculate cumulative returns properly
        running_equity = self.initial_capital
        eq_dates = [pd.to_datetime(self.start_date)]
        eq_values = [0.0] # 0% start
        
        drawdowns = [0.0]
        peak = self.initial_capital
        
        # Aggregate trades by date
        daily_pnl = trade_df.groupby('exit_date')['position_return'].sum()
        
        current_ret = 0
        for date, pnl in daily_pnl.items():
            # Approximation: Adding percentages (Simple Interest)
            # For Compound: current_equity *= (1 + pnl/100)
            # Let's use the actual capital growth if available, but here we use simple sum for returns graph
            current_ret += pnl
            
            eq_dates.append(date)
            eq_values.append(current_ret)
            
            # Drawdown Calculation (approximate)
            running_equity *= (1 + pnl/100)
            if running_equity > peak:
                peak = running_equity
            dd = (running_equity - peak) / peak * 100
            drawdowns.append(dd)

        # 1. Equity Curve
        fig.add_trace(go.Scatter(
            x=eq_dates, y=eq_values, 
            mode='lines', name='Strategy Return',
            line=dict(color='blue', width=2),
            fill='tozeroy', fillcolor='rgba(0,0,255,0.1)'
        ), row=1, col=1)
        
        # 2. Drawdown
        fig.add_trace(go.Scatter(
            x=eq_dates, y=drawdowns,
            mode='lines', name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy', fillcolor='rgba(255,0,0,0.2)'
        ), row=2, col=1)
        
        # 3. Trade Returns Bar Chart
        colors = ['green' if val >= 0 else 'red' for val in trade_df['position_return']]
        
        fig.add_trace(go.Bar(
            x=trade_df['exit_date'], 
            y=trade_df['position_return'],
            name='Trade PnL',
            marker_color=colors,
            customdata=trade_df[['ticker_name', 'quality_score']].values,
            hovertemplate='%{x}<br>%{customdata[0]}<br>Return: %{y:.2f}%<br>Score: %{customdata[1]}<extra></extra>'
        ), row=3, col=1)
        
        # Update Layout
        fig.update_layout(
            title=f"<b>Strategy Performance Report: {self.strategy_name}</b><br>" + 
                  f"Total Return: {self.total_return:.2f}% | Win Rate: {self.win_rate:.1f}% | MDD: {self.max_drawdown:.2f}%",
            template="plotly_white",
            height=1000,
            showlegend=True
        )
        
        fig.write_html(str(output_path))
        print(f"📊 HTML Chart saved to {output_path}")
