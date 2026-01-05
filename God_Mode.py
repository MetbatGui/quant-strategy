import os
import sys
import time
import schedule
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from ml_predictor import train_xgboost_model, predict_trade_return
from indicators import calculate_volume_ratio, calculate_rsi, calculate_atr

# --- 1. Capital Management ---
TOTAL_CAPITAL = 10_000_000  # 전체 자본

# --- 2. 4개 ETF 구성 (일반 ETF - 30만원 가능) ---
ETF_POOL = {
    "069500.KS": "KODEX 200",         # 코스피 200
    "091160.KS": "KODEX 반도체",      # 반도체 섹터
    "371460.KS": "KODEX 2차전지",     # 2차전지 섹터
    "091170.KS": "KODEX 금융",        # 금융 섹터
}

# --- 3. 전역 변수 ---
etf_data = {}
etf_models = {}
etf_features = {}
selected_etf = None
selected_score = 0
entry_price = 0

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def is_holiday(date):
    if date.weekday() >= 5:
        return True
    return False

# --- Strategy Functions ---

def load_etf_data():
    """ETF 데이터 로드"""
    global etf_data
    
    log("📥 ETF 데이터 로드 시작...")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    for ticker, name in ETF_POOL.items():
        try:
            df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
            
            if not df_raw.empty and all(col in df_raw.columns for col in ['Open', 'High', 'Low', 'Close']):
                df = pd.DataFrame()
                df['Open'] = df_raw['Open']
                df['High'] = df_raw['High']
                df['Low'] = df_raw['Low']
                df['Close'] = df_raw['Close']
                if 'Volume' in df_raw.columns:
                    df['Volume'] = df_raw['Volume']
                
                etf_data[ticker] = df
                log(f"  ✅ {name}: {len(df)} 데이터")
            else:
                log(f"  ⚠️ {name}: 데이터 불완전")
        except Exception as e:
            log(f"  ❌ {name}: {e}")
    
    return len(etf_data) > 0

def train_models():
    """ETF별 XGBoost 모델 학습"""
    global etf_models, etf_features
    
    log("🧠 XGBoost 모델 학습 시작...")
    
    for ticker in etf_data.keys():
        try:
            name = ETF_POOL[ticker]
            model, features, metrics = train_xgboost_model(etf_data[ticker], test_size=0.3)
            etf_models[ticker] = model
            etf_features[ticker] = features
            log(f"  ✅ {name}: 방향성 {metrics['direction_accuracy']:.1%}")
        except Exception as e:
            log(f"  ❌ {name}: {e}")

def calculate_etf_score(ticker, date):
    """ETF 품질 점수 계산 (0-100점)"""
    try:
        df = etf_data[ticker]
        score = 0
        
        # 1. XGBoost 예측 (0-30점)
        if ticker in etf_models:
            pred = predict_trade_return(df, date, etf_models[ticker], etf_features[ticker])
            if pred >= 1.0:
                score += 30
            elif pred >= 0.7:
                score += 20
            elif pred >= 0.4:
                score += 10
        
        # 2. 추세 (0-20점)
        ma5 = df['Close'].rolling(5).mean()
        ma20 = df['Close'].rolling(20).mean()
        
        if date in ma5.index and date in ma20.index:
            price = df.loc[date, 'Close']
            m5 = ma5.loc[date]
            m20 = ma20.loc[date]
            
            if price > m5 > m20:
                score += 20
            elif price > m5:
                score += 10
        
        # 3. 모멘텀 (0-20점)
        if date in df.index:
            idx = df.index.get_loc(date)
            if idx >= 5:
                momentum_5d = (df['Close'].iloc[idx] / df['Close'].iloc[idx-5] - 1) * 100
                if momentum_5d > 3:
                    score += 20
                elif momentum_5d > 1:
                    score += 10
                elif momentum_5d > 0:
                    score += 5
        
        # 4. 거래량 (0-15점)
        vol_ratio = calculate_volume_ratio(df, 20)
        if date in vol_ratio.index:
            vr = vol_ratio.loc[date]
            if vr >= 1.5:
                score += 15
            elif vr >= 1.2:
                score += 10
            elif vr >= 1.0:
                score += 5
        
        # 5. RSI (0-15점)
        rsi = calculate_rsi(df['Close'], 14)
        if date in rsi.index:
            r = rsi.loc[date]
            if 40 < r < 70:
                score += 15
            elif 30 < r < 80:
                score += 10
        
        return score
        
    except Exception as e:
        log(f"  ❌ 점수 계산 오류 ({ETF_POOL.get(ticker, ticker)}): {e}")
        return 0

def select_best_etf():
    """최적 ETF 선택"""
    global selected_etf, selected_score
    
    KST = pytz.timezone('Asia/Seoul')
    now = datetime.now(KST)
    
    if is_holiday(now):
        log("📅 주말/공휴일입니다. 거래 안함.")
        return
    
    log("🌅 [ETF 선택] 시작...")
    
    # 가장 최근 거래일 찾기
    latest_date = None
    for df in etf_data.values():
        if not df.empty:
            latest_date = df.index[-1]
            break
    
    if latest_date is None:
        log("  ❌ 데이터 없음")
        return
    
    # 각 ETF 점수 계산
    scores = {}
    for ticker in etf_data.keys():
        score = calculate_etf_score(ticker, latest_date)
        scores[ticker] = score
        log(f"  📊 {ETF_POOL[ticker]}: {score}점")
    
    if not scores:
        log("  ❌ 점수 계산 실패")
        return
    
    # 최고 점수 ETF 선택
    best_ticker = max(scores, key=scores.get)
    best_score = scores[best_ticker]
    
    if best_score < 60:
        log(f"  ⚠️ 최고 점수 {best_score}점 < 60점. 오늘 거래 안함.")
        selected_etf = None
        selected_score = 0
        return
    
    selected_etf = best_ticker
    selected_score = best_score
    
    log(f"  ✅ 선택: {ETF_POOL[best_ticker]} ({best_score}점)")
    
    # 순위 표시
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for idx, (ticker, score) in enumerate(sorted_scores, 1):
        mark = "👑" if idx == 1 else f"{idx}위"
        log(f"     {mark} {ETF_POOL[ticker]}: {score}점")

def check_entry():
    """진입 체크"""
    global entry_price
    
    KST = pytz.timezone('Asia/Seoul')
    now = datetime.now(KST)
    
    if is_holiday(now):
        return
    
    if selected_etf is None:
        log("⏰ [진입 체크] 선택된 ETF 없음. 대기 중...")
        return
    
    log(f"⏰ [진입 체크] {ETF_POOL[selected_etf]} 분석...")
    
    try:
        df = etf_data[selected_etf]
        
        if df.empty or len(df) < 2:
            log("  ❌ 데이터 부족")
            return
        
        latest_date = df.index[-1]
        row = df.iloc[-1]
        prev_range = (df['High'] - df['Low']).iloc[-2]
        
        if pd.isna(prev_range):
            log("  ❌ 전일 변동폭 없음")
            return
        
        target = row['Open'] + (prev_range * 0.03)
        entry_price = target
        
        log(f"  📊 시가: {row['Open']:,.0f}원")
        log(f"  📊 전일 변동폭: {prev_range:,.0f}원")
        log(f"  🎯 진입 목표가: {target:,.0f}원")
        log(f"  📈 현재가 추정: {row['Close']:,.0f}원")
        
        if row['High'] > target:
            log(f"  ✅ 목표가 도달! {target:,.0f}원에 매수 진입")
        else:
            log(f"  ⏳ 목표가 대기 중... (아직 미달)")
            
    except Exception as e:
        log(f"  ❌ 오류: {e}")

def run_closing():
    """
    15:20 KST: 모든 포지션 청산
    (Day Trading Only - No Overnight Risk)
    """
    log("👋 [청산] 모든 포지션 청산...")
    log("  ✅ ETF 매도 완료 (if any)")
    log("  💤 다음 거래일 08:50까지 대기")

# --- Scheduler ---

def start_bot():
    log("🤖 GOD MODE BOT (4개 ETF 전략) 시작")
    log(f"  💰 총 자본: {TOTAL_CAPITAL:,}원")
    log(f"  📊 ETF 풀: {len(ETF_POOL)}개")
    for ticker, name in ETF_POOL.items():
        log(f"     - {name} ({ticker})")
    log("  ⏳ 장 시작 대기 중...")
    
    # 초기 데이터 로드 및 모델 학습
    log("\n⚡ 초기화 시작...")
    if load_etf_data():
        train_models()
        log("✅ 초기화 완료\n")
    else:
        log("❌ 데이터 로드 실패. 봇 종료.")
        return
    
    # 스케줄 설정
    schedule.every().day.at("08:50").do(select_best_etf)
    schedule.every().day.at("10:00").do(check_entry)
    schedule.every().day.at("15:20").do(run_closing)
    
    # Dry Run (즉시 실행)
    log("⚡ Dry Run 시작...")
    select_best_etf()
    check_entry()
    
    # 스케줄러 실행
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_bot()
