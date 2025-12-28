import matplotlib.pyplot as plt
import platform
import pandas as pd

class Visualizer:
    def __init__(self):
        system_name = platform.system()
        if system_name == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        elif system_name == 'Darwin':
            plt.rc('font', family='AppleGothic')
        else:
            plt.rc('font', family='NanumBarunGothic')
        plt.rcParams['axes.unicode_minus'] = False

    def plot_result(self, df: pd.DataFrame, ticker_name: str, initial_capital: float):
        hold_return = (df['Equity_Hold'].iloc[-1] / initial_capital - 1) * 100
        strat_return = (df['Equity_Strategy'].iloc[-1] / initial_capital - 1) * 100
        
        print(f"\n📊 [{ticker_name} - 터틀 트레이딩 결과]")
        print(f"1. 단순 보유: {hold_return:.2f}%")
        print(f"2. 터틀 전략: {strat_return:.2f}%")
        
# 캔버스 설정
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
        
        # [1] 자산 곡선 (상단 차트)
        axes[0].plot(df.index, df['Equity_Hold'], label='단순 보유', color='gray', linestyle='--')
        axes[0].plot(df.index, df['Equity_Strategy'], label='터틀 전략', color='red', linewidth=2)
        axes[0].set_title(f'{ticker_name} 수익률 비교')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # [2] 매매 타점 차트 (하단 차트)
        # 주가
        axes[1].plot(df.index, df['Close'], label='주가', color='black', alpha=0.6)
        
        # 채널 라인
        if 'Donchian_High' in df.columns:
            axes[1].plot(df.index, df['Donchian_High'], color='blue', linestyle=':', alpha=0.3)
            axes[1].plot(df.index, df['Donchian_Low'], color='green', linestyle=':', alpha=0.3)
            # 이평선 필터가 있다면 표시
            if 'MA_Filter' in df.columns:
                 axes[1].plot(df.index, df['MA_Filter'], label='추세필터(MA)', color='orange', linestyle='--', alpha=0.5)

        # ==========================================
        # [NEW] 매수/매도 포인트 시각화 (Scatter Plot)
        # ==========================================
        # 1. 매수 지점 (BUY) -> 빨간색 삼각형 (^)
        buy_idx = df[df['Action'] == 'BUY'].index
        buy_price = df.loc[buy_idx, 'Close']
        axes[1].scatter(buy_idx, buy_price, marker='^', color='red', s=100, label='매수', zorder=5)

        # 2. 매도 지점 (SELL) -> 파란색 역삼각형 (v)
        sell_idx = df[df['Action'] == 'SELL'].index
        sell_price = df.loc[sell_idx, 'Close']
        axes[1].scatter(sell_idx, sell_price, marker='v', color='blue', s=100, label='매도', zorder=5)

        axes[1].set_title('매매 타점 (▲매수 / ▼매도)', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()