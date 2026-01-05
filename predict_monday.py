"""
금요일 데이터로 월요일 거래 종목 및 진입가 계산
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from ml_predictor import train_xgboost_model, predict_trade_return
from indicators import calculate_volume_ratio, calculate_rsi

# 최종 확정 7종 ETF
ETF_POOL = {
    "069500.KS": "KODEX 200",
    "091160.KS": "KODEX 반도체",
    "091170.KS": "KODEX 금융",
    "091180.KS": "KODEX 자동차",
    "229200.KS": "KODEX 코스닥150",
    "305720.KS": "KODEX 2차전지K-뉴딜",
    "261140.KS": "KODEX 바이오",
}

def calculate_etf_score(ticker, df, model, features, date):
    """ETF 품질 점수 계산"""
    try:
        score = 0
        details = {}
        
        # XGBoost 예측
        pred = predict_trade_return(df, date, model, features)
        if pred >= 1.0:
            score += 30
            details['XGBoost'] = f"30점 (예측 {pred:.2f}%)"
        elif pred >= 0.7:
            score += 20
            details['XGBoost'] = f"20점 (예측 {pred:.2f}%)"
        elif pred >= 0.4:
            score += 10
            details['XGBoost'] = f"10점 (예측 {pred:.2f}%)"
        else:
            details['XGBoost'] = f"0점 (예측 {pred:.2f}%)"
        
        # 추세
        ma5 = df['Close'].rolling(5).mean()
        ma20 = df['Close'].rolling(20).mean()
        
        if date in ma5.index and date in ma20.index:
            price = df.loc[date, 'Close']
            m5 = ma5.loc[date]
            m20 = ma20.loc[date]
            
            if price > m5 > m20:
                score += 20
                details['추세'] = "20점 (강세)"
            elif price > m5:
                score += 10
                details['추세'] = "10점 (약세상승)"
            else:
                details['추세'] = "0점 (약세)"
        
        # 모멘텀
        if date in df.index:
            idx = df.index.get_loc(date)
            if idx >= 5:
                momentum_5d = (df['Close'].iloc[idx] / df['Close'].iloc[idx-5] - 1) * 100
                if momentum_5d > 3:
                    score += 20
                    details['모멘텀'] = f"20점 (5일 {momentum_5d:.2f}%)"
                elif momentum_5d > 1:
                    score += 10
                    details['모멘텀'] = f"10점 (5일 {momentum_5d:.2f}%)"
                elif momentum_5d > 0:
                    score += 5
                    details['모멘텀'] = f"5점 (5일 {momentum_5d:.2f}%)"
                else:
                    details['모멘텀'] = f"0점 (5일 {momentum_5d:.2f}%)"
        
        # 거래량
        vol_ratio = calculate_volume_ratio(df, 20)
        if date in vol_ratio.index:
            vr = vol_ratio.loc[date]
            if vr >= 1.5:
                score += 15
                details['거래량'] = f"15점 (평균 대비 {vr:.2f}배)"
            elif vr >= 1.2:
                score += 10
                details['거래량'] = f"10점 (평균 대비 {vr:.2f}배)"
            elif vr >= 1.0:
                score += 5
                details['거래량'] = f"5점 (평균 대비 {vr:.2f}배)"
            else:
                details['거래량'] = f"0점 (평균 대비 {vr:.2f}배)"
        
        # RSI
        rsi = calculate_rsi(df['Close'], 14)
        if date in rsi.index:
            r = rsi.loc[date]
            if 40 < r < 70:
                score += 15
                details['RSI'] = f"15점 (RSI {r:.1f})"
            elif 30 < r < 80:
                score += 10
                details['RSI'] = f"10점 (RSI {r:.1f})"
            else:
                details['RSI'] = f"0점 (RSI {r:.1f})"
        
        return score, details
    except Exception as e:
        return 0, {'Error': str(e)}

def main():
    print("="*80)
    print("📅 월요일(2026-01-06) 거래 예측 (일요일 2026-01-05 최신 데이터)")
    print("="*80)
    print()
    
    # 데이터 로드
    end_date = "2026-01-06"
    start_date = "2024-06-01"
    friday_date = "2026-01-05"  # 최신 거래일
    monday_date = "2026-01-06"  # 월요일
    
    etf_data = {}
    etf_models = {}
    etf_features = {}
    
    print("📥 데이터 로드 및 모델 학습 중...")
    for ticker, name in ETF_POOL.items():
        try:
            df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if isinstance(df_raw.columns, pd.MultiIndex):
                df_raw.columns = [col[0] if isinstance(col, tuple) else col for col in df_raw.columns]
            
            if not df_raw.empty:
                df = pd.DataFrame()
                df['Open'] = df_raw['Open']
                df['High'] = df_raw['High']
                df['Low'] = df_raw['Low']
                df['Close'] = df_raw['Close']
                if 'Volume' in df_raw.columns:
                    df['Volume'] = df_raw['Volume']
                
                etf_data[ticker] = df
                
                # 모델 학습
                model, features, metrics = train_xgboost_model(df, test_size=0.3)
                etf_models[ticker] = model
                etf_features[ticker] = features
                
                print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print()
    print("="*80)
    print(f"🔍 최신 거래일({friday_date}) 기준 품질 점수 분석")
    print("="*80)
    print()
    
    # 금요일 데이터로 품질 점수 계산
    scores = {}
    score_details = {}
    friday_prices = {}
    
    friday_dt = pd.Timestamp(friday_date)
    
    for ticker, name in ETF_POOL.items():
        if ticker in etf_data and ticker in etf_models:
            df = etf_data[ticker]
            
            if friday_dt in df.index:
                score, details = calculate_etf_score(ticker, df, etf_models[ticker], etf_features[ticker], friday_dt)
                scores[ticker] = score
                score_details[ticker] = details
                friday_prices[ticker] = df.loc[friday_dt]
                
                print(f"📊 {name} ({ticker})")
                print(f"   총점: {score}점")
                for key, value in details.items():
                    print(f"   - {key}: {value}")
                print()
    
    # 최고 점수 ETF 선정
    if scores:
        best_ticker = max(scores, key=scores.get)
        best_score = scores[best_ticker]
        best_name = ETF_POOL[best_ticker]
        
        print("="*80)
        print("🎯 월요일 거래 종목")
        print("="*80)
        print()
        
        if best_score >= 60:
            friday_data = friday_prices[best_ticker]
            
            # 변동폭 계산 (전일 High - Low)
            df = etf_data[best_ticker]
            friday_idx = df.index.get_loc(friday_dt)
            if friday_idx > 0:
                prev_data = df.iloc[friday_idx - 1]
                range_value = prev_data['High'] - prev_data['Low']
            else:
                range_value = friday_data['High'] - friday_data['Low']
            
            # 월요일 시가는 알 수 없으므로 금요일 종가를 기준으로 가정
            estimated_monday_open = friday_data['Close']
            entry_price = estimated_monday_open + (range_value * 0.03)
            
            print(f"✅ 선정 종목: {best_name}")
            print(f"   티커: {best_ticker}")
            print(f"   품질 점수: {best_score}점")
            print()
            print(f"📈 최신 거래일({friday_date}) 종가: {friday_data['Close']:,.0f}원")
            print(f"   변동폭(전일): {range_value:,.0f}원")
            print()
            print(f"🎯 월요일 진입 전략:")
            print(f"   예상 시가: {estimated_monday_open:,.0f}원 (최신 종가 기준)")
            print(f"   진입가(시가+변동폭3%): {entry_price:,.0f}원")
            print()
            print(f"💡 실전 가이드:")
            print(f"   1. 월요일 시가가 {estimated_monday_open:,.0f}원 근처라면")
            print(f"   2. 당일 고가가 {entry_price:,.0f}원 돌파 시 매수")
            print(f"   3. 화요일 갭 상승 시 화요일 종가에 청산")
            print(f"   4. 화요일 갭 하락 시 화요일 시가에 청산")
            print()
        else:
            print(f"⚠️  최고 점수 종목: {best_name} ({best_score}점)")
            print(f"   품질 점수가 60점 미만이므로 거래 보류 권장")
            print()
    else:
        print("❌ 분석 가능한 데이터가 없습니다.")
    
    print("="*80)

if __name__ == "__main__":
    main()
