"""
거래(Trade) 엔티티
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Trade:
    """단일 거래 정보"""
    ticker: str
    ticker_name: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quality_score: int = 0
    score_details: dict = None
    
    def __post_init__(self):
        if self.score_details is None:
            self.score_details = {}
    
    @property
    def position_return(self) -> Optional[float]:
        """포지션 수익률 (%)"""
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1) * 100
    
    @property
    def is_open(self) -> bool:
        """포지션이 열려있는지"""
        return self.exit_date is None
    
    @property
    def is_winning(self) -> Optional[bool]:
        """수익 거래인지"""
        if self.position_return is None:
            return None
        return self.position_return > 0
    
    def close(self, exit_price: float, exit_date: datetime):
        """포지션 종료"""
        self.exit_price = exit_price
        self.exit_date = exit_date
