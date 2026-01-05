"""
전략 기본 인터페이스
"""

from typing import Protocol
import pandas as pd
from datetime import datetime


class TradingStrategy(Protocol):
    """모든 전략이 구현해야 하는 인터페이스"""
    
    def check_entry_signal(self, df: pd.DataFrame, date: datetime) -> bool:
        """진입 신호 확인"""
        ...
    
    def check_exit_signal(self, df: pd.DataFrame, entry_date: datetime, current_date: datetime) -> bool:
        """청산 신호 확인"""
        ...
