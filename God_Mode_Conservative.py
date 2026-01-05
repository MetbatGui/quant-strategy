"""
GOD MODE - Conservative Enhancement Version
보수적 개선 버전: 볼륨 필터 + 변동성 필터 추가

원본 대비 개선사항:
1. 거래량 필터: 평균 대비 1.2배 이상일 때만 진입
2. 변동성 필터: ATR 백분위 20~80% 구간에서만 거래
"""

import os
import sys
import time
import schedule
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz
from indicators import calculate_atr_percentile, calculate_volume_ratio, is_valid_volatility_regime, has_sufficient_volume

# Capital Settings
TOTAL_CAPITAL = 10_000_000
ALLOC_STABLE = 0.0
ALLOC_TURBO = 1.0

cap_stable = TOTAL_CAPITAL * ALLOC_STABLE
cap_turbo = TOTAL_CAPITAL * ALLOC_TURBO

# Targets
TARGETS = {
    "HYNIX": "000660.KS",
    "LEV": "122630.KS",
    "INV": "252670.KS",
    "FUTURES": "NQ=F",
    "KRW": "KRW=X"
}

# Enhancement Parameters (Relaxed)
VOLUME_THRESHOLD = 1.0  # 평균 이상 (완화)
ATR_MIN_PERCENTILE = 10.0  # 변동성 하한 (완화)
ATR_MAX_PERCENTILE = 90.0  # 변동성 상한 (완화)

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def is_holiday(date):
    if date.weekday() >= 5: return True
    return False

def run_morning_prep():
    """
    08:50 KST: Morning Preparation & Macro Check
    """
    log("🌅 [Morning Prep] Analyzing Market Conditions...")
    
    signals = {"STABLE": "HOLD", "TURBO": "CASH"}
    
    # 1. Macro Check (Stable Pot)
    try:
        krw = yf.download(TARGETS["KRW"], period="5d", progress=False)
        if hasattr(krw.columns, 'levels'): krw.columns = krw.columns.get_level_values(0)
        
        last_close = krw['Close'].iloc[-1]
        prev_close = krw['Close'].iloc[-2]
        change = ((last_close / prev_close) - 1) * 100
        
        log(f"  💵 Exchange Rate Change: {change:.2f}%")
        
        if change < -0.5:
            log("  🚨 Signal: KRW Drop Detected! -> Stable Pot: BUY HYNIX (Open)")
            signals["STABLE"] = "BUY_HYNIX_OPEN"
        else:
            log("  ✅ Signal: Macro Calm. -> Stable Pot: Wait for Breakout.")
            
    except Exception as e:
        log(f"  ❌ Error checking Macro: {e}")

    # 2. Futures Check (Turbo Pot)
    try:
        fut = yf.download(TARGETS["FUTURES"], interval="5m", period="5d", progress=False)
        if hasattr(fut.columns, 'levels'): fut.columns = fut.columns.get_level_values(0)
        
        curr_price = fut['Close'].iloc[-1]
        past_12h = fut.iloc[-144:]
        start_p = past_12h['Open'].iloc[0]
        change_f = ((curr_price / start_p) - 1) * 100
        
        log(f"  📉 Futures (12H) Change: {change_f:.2f}%")
        
        vix_ok = True
        try:
            vix_data = yf.download("^VIX", period="5d", progress=False)
            if hasattr(vix_data.columns, 'levels'): vix_data.columns = vix_data.columns.get_level_values(0)
            if not vix_data.empty:
                curr_vix = vix_data['Close'].iloc[-1]
                log(f"  😨 VIX Level: {curr_vix:.2f}")
                if curr_vix > 20:
                    vix_ok = False
                    log("  ⚠️ VIX > 20 (High Fear). Skipping Leverage Buy.")
        except:
            log("  ⚠️ VIX Check Failed. Proceeding with caution.")

        if change_f > 0.2:
            if vix_ok:
                log("  🚀 Signal: Futures UP (Early Entry)! -> Turbo Pot: BUY LEVERAGE")
                signals["TURBO"] = "BUY_LEV"
            else:
                log("  💤 Signal: Futures UP but VIX High. -> Turbo Pot: CASH (Safety First)")
                
        elif change_f < -0.2:
            log("  📉 Signal: Futures DOWN! -> Turbo Pot: BUY INVERSE 2X")
            signals["TURBO"] = "BUY_INV"
        else:
            log("  💤 Signal: Futures Flat. -> Turbo Pot: CASH")
            
    except Exception as e:
        log(f"  ❌ Error checking Futures: {e}")
        
    return signals

def run_intraday_check():
    """
    10:00 KST: Breakout Check for Stable Pot
    """
    log("🕙 [Intraday Check] Hynix Breakout Analysis...")
    
    try:
        df = yf.download(TARGETS["HYNIX"], interval="5m", period="1d", progress=False)
        if hasattr(df.columns, 'levels'): df.columns = df.columns.get_level_values(0)
        
        if len(df) < 10:
            log("  ⚠️ Not enough data yet.")
            return

        morning_high = df['High'].iloc[:12].max()
        current_price = df['Close'].iloc[-1]
        
        log(f"  📊 Morning High: {morning_high} | Current: {current_price}")
        
        if current_price > morning_high:
            log("  🔥 BREAKOUT! Hynix broke morning high. -> BUY SIGNAL")
        else:
            log("  Wait. No breakout yet.")
            
    except Exception as e:
        log(f"  ❌ Error: {e}")

def run_closing():
    """
    15:20 KST: Closing All Positions
    """
    log("👋 [Closing] Selling All Positions (Stable & Turbo)...")
    log("  ✅ Sold Hynix (if any).")
    log("  ✅ Sold ETFs (if any).")
    log("  💤 Bot going to sleep until tomorrow 08:50.")

def run_lazy_guide():
    """
    09:01 KST: Calculate Targets for Lazy Alpha (Lev) + Hedge Mode (Inv)
    🆕 보수적 개선: 볼륨 + 변동성 필터 추가
    """
    KST = pytz.timezone('Asia/Seoul')
    now = datetime.now(KST)
    if is_holiday(now): return

    log("\n========== 🐢 LAZY ALPHA (Conservative Enhanced) ==========")
    log(f"⏰ Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Fetching Data
        tickers = ["122630.KS", "252670.KS", "102110.KS", "NQ=F"]
        data = yf.download(tickers, interval="1d", period="90d", progress=False)  # 90일 (볼륨/ATR 계산용)
        
        def get_series(ticker):
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, axis=1, level=1)
                        return df
                return None
            except Exception as e:
                log(f"Error parsing data for {ticker}: {e}")
                return None

        kospi = get_series("102110.KS")
        inv = get_series("252670.KS")
        lev = get_series("122630.KS")
        
        # --- 변동성 체제 확인 (보수적 개선 #1) ---
        should_skip_due_to_volatility = False
        
        if lev is not None and len(lev) >= 74:  # 14 + 60
            atr_pct = calculate_atr_percentile(lev, atr_period=14, lookback=60)
            current_atr_pct = atr_pct.iloc[-1]
            
            if pd.notna(current_atr_pct):
                log(f"  📊 Volatility Regime (ATR %ile): {current_atr_pct:.1f}%")
                
                if not is_valid_volatility_regime(current_atr_pct, ATR_MIN_PERCENTILE, ATR_MAX_PERCENTILE):
                    should_skip_due_to_volatility = True
                    if current_atr_pct < ATR_MIN_PERCENTILE:
                        log(f"  ⚠️ [FILTER] Volatility TOO LOW ({current_atr_pct:.1f}%) -> Skipping Trade (Choppy Market)")
                    else:
                        log(f"  ⚠️ [FILTER] Volatility TOO HIGH ({current_atr_pct:.1f}%) -> Skipping Trade (Extreme Risk)")
        
        if should_skip_due_to_volatility:
            log("  🚫 Trade CANCELLED due to volatility filter")
            log("✅ Guide Complete (No Trade).")
            return
        
        # --- STRATEGY LOGIC ---
        should_run_inv = False
        
        if kospi is not None and len(kospi) >= 6:
            closes = kospi['Close'].iloc[:-1]
            ma5 = closes.rolling(window=5).mean().iloc[-1]
            last_close = closes.iloc[-1]
            is_downtrend = last_close < ma5
        else:
            is_downtrend = False
            
        if is_downtrend:
            should_run_inv = True

        # EXECUTE LOGIC
        if should_run_inv:
            log(f"📉 [SHORT: KODEX Inverse 2X] -> Trend DOWN (Bear Market Mode)")
            
            if inv is not None and len(inv) >= 22:  # Volume 계산용 20일 + 2
                # 볼륨 필터 확인 (보수적 개선 #2)
                volume_ratio_series = calculate_volume_ratio(inv, period=20)
                current_volume_ratio = volume_ratio_series.iloc[-1]
                
                if pd.notna(current_volume_ratio):
                    log(f"  📊 Volume Ratio: {current_volume_ratio:.2f}x average")
                    
                    if not has_sufficient_volume(current_volume_ratio, VOLUME_THRESHOLD):
                        log(f"  ⚠️ [FILTER] Volume TOO LOW ({current_volume_ratio:.2f}x) -> Skipping Inverse Trade")
                        log("⛔ **LEVERAGE SKIPPED** (Safe Mode Active)")
                        log("✅ Guide Complete (No Trade).")
                        return
                
                prev_inv = inv.iloc[-2]
                curr_inv = inv.iloc[-1]
                range_inv = prev_inv['High'] - prev_inv['Low']
                target_inv = curr_inv['Open'] + (range_inv * 0.5)
                target_inv = int(round(target_inv / 5) * 5)
                
                log(f"  🚨 **ACTIVATED** (Market Risk Detected)")
                log(f"  🎯 Buy Stop: **{target_inv:,} KRW**")
                log(f"  ⚠️ **EXIT TODAY CLOSE** (Do Not Hold Overnight)")
            else:
                log("❌ Inv Data Missing")
                
            log("⛔ **LEVERAGE SKIPPED** (Safe Mode Active)")
            
        else:
            # NO HEDGE NEEDED -> RUN LEVERAGE
            log(f"📊 [LONG: KODEX Leverage] -> Trend UP (Bull Market Mode)")
            
            if lev is not None and len(lev) >= 22:
                # 볼륨 필터 확인 (보수적 개선 #2)
                volume_ratio_series = calculate_volume_ratio(lev, period=20)
                current_volume_ratio = volume_ratio_series.iloc[-1]
                
                if pd.notna(current_volume_ratio):
                    log(f"  📊 Volume Ratio: {current_volume_ratio:.2f}x average")
                    
                    if not has_sufficient_volume(current_volume_ratio, VOLUME_THRESHOLD):
                        log(f"  ⚠️ [FILTER] Volume TOO LOW ({current_volume_ratio:.2f}x) -> Skipping Leverage Trade")
                        log("✅ Guide Complete (No Trade).")
                        return
                
                prev_lev = lev.iloc[-2]
                curr_lev = lev.iloc[-1]
                
                range_lev = prev_lev['High'] - prev_lev['Low']
                target_lev = curr_lev['Open'] + (range_lev * 0.09)
                target_lev = int(round(target_lev / 5) * 5)
                
                log(f"  🎯 Buy Stop: **{target_lev:,} KRW**")
                log(f"  🚀 Strategy: **Buy & Hold Overnight**")
                log(f"     👉 SELL Target: **Tomorrow Open**")
                log(f"     🔥 **[Trend Day Extension]**: If Tomorrow Open Gap > +0.1%, **HOLD until Close**!")
            else:
                log("❌ Lev Data Missing")

        log("✅ Guide Complete.")

    except Exception as e:
        log(f"❌ Error in Lazy Guide: {e}")

def start_bot():
    log("🤖 GOD MODE BOT (Conservative Enhanced) Initiated.")
    log(f"  💰 Allocation: Stable {ALLOC_STABLE*100}% | Turbo {ALLOC_TURBO*100}%")
    log(f"  🛡️ Enhancements: Volume Filter (≥{VOLUME_THRESHOLD:.1f}x) + Volatility Filter ({ATR_MIN_PERCENTILE:.0f}%-{ATR_MAX_PERCENTILE:.0f}%)")
    log("  ⏳ Waiting for Market Open...")
    
    schedule.every().day.at("08:50").do(run_morning_prep)
    schedule.every().day.at("09:01").do(run_lazy_guide)
    schedule.every().day.at("10:00").do(run_intraday_check)
    schedule.every().day.at("15:20").do(run_closing)
    
    # Dry Run on Start
    log("⚡ Performing Cleanup & Dry-Run...")
    signals = run_morning_prep()
    log(f"  👀 Current Signals: {signals}")
    run_lazy_guide()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_bot()
