import pandas as pd
import matplotlib.pyplot as plt
from pykrx import stock
import platform

# ==========================================
# 1. 설정 (파마리서치)
# ==========================================
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

ticker = "214450"  # 파마리서치
start_date = "20240101"
end_date = "20251227"

print(f"📥 파마리서치({ticker}) 심폐소생 중... ({start_date} ~ {end_date})")

# 데이터 수집 및 병합
df_price = stock.get_market_ohlcv(start_date, end_date, ticker)
df_investor = stock.get_market_trading_value_by_date(start_date, end_date, ticker)
df = pd.concat([df_price[['시가', '종가']], df_investor[['외국인합계', '기관합계']]], axis=1)
df.columns = ['Open', 'Close', 'Foreigner', 'Institution']
df.dropna(inplace=True)

# ==========================================
# 2. 지표 계산
# ==========================================
def calculate_indicators(df, rsi_period=14, signal_period=9, ma_period=20):
    # 수급 지표
    smart_money = df['Foreigner'] + df['Institution']
    U = smart_money.where(smart_money > 0, 0)
    D = -smart_money.where(smart_money < 0, 0)
    AU = U.rolling(window=rsi_period).mean()
    AD = D.rolling(window=rsi_period).mean()
    RS = AU / (AD + 1e-9)
    df['Smart_RSI'] = 100 - (100 / (1 + RS))
    df['Smart_Signal'] = df['Smart_RSI'].rolling(window=signal_period).mean()

    # 추세 지표
    df['MA20'] = df['Close'].rolling(window=ma_period).mean()
    return df

df = calculate_indicators(df)

# ==========================================
# 3. 시뮬레이션 (재진입 로직 추가)
# ==========================================
initial_capital = 100_000_000
hold_shares = initial_capital / df['Open'].iloc[0]
df['Equity_Hold'] = hold_shares * df['Close']

cash = initial_capital
shares = 0
equity_curve = []

start_idx = 20
for _ in range(start_idx): equity_curve.append(initial_capital)

for i in range(start_idx, len(df)):
    curr_close = df['Close'].iloc[i]
    curr_ma20 = df['MA20'].iloc[i]
    prev_close = df['Close'].iloc[i-1]
    prev_ma20 = df['MA20'].iloc[i-1]
    
    # 수급 신호
    prev_rsi = df['Smart_RSI'].iloc[i-1]
    prev_sig = df['Smart_Signal'].iloc[i-1]
    curr_rsi = df['Smart_RSI'].iloc[i]
    curr_sig = df['Smart_Signal'].iloc[i]
    
    # --- [매수 로직 개선] ---
    # 1. 수급이 들어오거나 (Smart Money Buy)
    cond1 = (prev_rsi < prev_sig) and (curr_rsi > curr_sig)
    
    # 2. ★추세가 복귀되거나 (Trend Recovery) -> "재진입" 핵심
    # 어제는 20일선 밑이었는데, 오늘 다시 뚫고 올라옴
    cond2 = (prev_close < prev_ma20) and (curr_close > curr_ma20)
    
    if (cond1 or cond2) and shares == 0:
        shares = cash / curr_close
        cash = 0
    
    # --- [매도 로직 유지] ---
    # 추세 이탈 시 (20일선 붕괴)
    if (curr_close < curr_ma20) and shares > 0:
        cash = shares * curr_close
        shares = 0
        
    equity_curve.append(cash + (shares * curr_close))

# 결과 정리
if len(equity_curve) < len(df):
    equity_curve = [initial_capital] * (len(df) - len(equity_curve)) + equity_curve
df['Equity_Hybrid'] = equity_curve[-len(df):]

hold_return = (df['Equity_Hold'].iloc[-1] / initial_capital - 1) * 100
hybrid_return = (df['Equity_Hybrid'].iloc[-1] / initial_capital - 1) * 100

print(f"\n📊 [파마리서치 최종 수정 성과]")
print(f"1. 단순 보유: {hold_return:.2f}%")
print(f"2. 수정 전략: {hybrid_return:.2f}% (재진입 로직 추가)")
print(f"👉 전략 차이: {hybrid_return - hold_return:.2f}%p")

# 그래프
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Equity_Hold'], label=f'단순 보유 ({hold_return:.0f}%)', color='gray', linestyle='--', alpha=0.6)
plt.plot(df.index, df['Equity_Hybrid'], label=f'수정 전략 ({hybrid_return:.0f}%)', color='red', linewidth=2.5)
plt.fill_between(df.index, df['Equity_Hybrid'], df['Equity_Hold'], where=(df['Equity_Hybrid'] >= df['Equity_Hold']), color='red', alpha=0.1)
plt.title('파마리서치: 추세 복귀 시 즉시 재탑승 (Whipsaw 극복)', fontsize=16)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
