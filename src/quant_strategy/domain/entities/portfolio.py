"""
포트폴리오 엔티티
"""

from typing import Optional, List
from datetime import datetime
import numpy as np
from quant_strategy.domain.entities.trade import Trade


class Portfolio:
    """포트폴리오 관리"""
    
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.current_position: Optional[Trade] = None
        self.closed_trades: List[Trade] = []
    
    def open_position(self, trade: Trade):
        """포지션 열기"""
        if self.current_position is not None and self.current_position.is_open:
            raise ValueError("Cannot open position: existing position is still open")
        
        self.current_position = trade
    
    def close_position(self, exit_price: float, exit_date: datetime):
        """현재 포지션 청산"""
        if self.current_position is None:
            raise ValueError("No position to close")
        
        self.current_position.close(exit_price, exit_date)
        
        # 자본 업데이트
        position_return = self.current_position.position_return
        self.capital *= (1 + position_return / 100)
        
        # 포지션을 히스토리로 이동
        self.closed_trades.append(self.current_position)
        self.current_position = None
    
    @property
    def total_return(self) -> float:
        """총 수익률 (%)"""
        return (self.capital / self.initial_capital - 1) * 100
    
    @property
    def num_trades(self) -> int:
        """총 거래 횟수"""
        return len(self.closed_trades)
    
    @property
    def win_rate(self) -> float:
        """승률 (%)"""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.is_winning)
        return wins / len(self.closed_trades) * 100
    
    @property
    def average_return(self) -> float:
        """평균 수익률 (%)"""
        if not self.closed_trades:
            return 0.0
        returns = [t.position_return for t in self.closed_trades]
        return np.mean(returns)
    
    @property
    def max_drawdown(self) -> float:
        """최대 낙폭 (MDD, %)"""
        if not self.closed_trades:
            return 0.0
        
        capital_history = [self.initial_capital]
        for trade in self.closed_trades:
            capital_history.append(capital_history[-1] * (1 + trade.position_return / 100))
        
        peak = capital_history[0]
        max_dd = 0
        
        for value in capital_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def get_metrics(self) -> dict:
        """성과 지표 반환"""
        return {
            'total_return': self.total_return,
            'final_capital': self.capital,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'average_return': self.average_return,
            'max_drawdown': self.max_drawdown
        }
