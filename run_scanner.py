from quant_strategy.application.services.market_scanner import MarketScanner

def main():
    # 스캔하고 싶은 종목 리스트 (네이버 금융 등에서 가져온 주요 종목들)
    target_tickers = {
        # [반도체/AI] - 메모리 & 디자인 & 장비
        "005930": "삼성전자",        # 반도체 대장
        "000660": "SK하이닉스",      # HBM 대장
        "005935": "삼성전자우",      # 배당/우선주
        "042700": "한미반도체",      # TC본딩 장비
        "402340": "SK스퀘어",        # SK하이닉스 지분 보유 (투자사)
        "200710": "에이디테크놀로지", # 디자인하우스 (ARM 관련주)

        # [방산/우주]
        "012450": "한화에어로스페이스", # 우주/방산 대장

        # [바이오/플랫폼/비만치료] - 핫한 기술수출 & 신약
        "196170": "알테오젠",        # SC제형 변경 플랫폼 (바이오 대장)
        "347850": "디앤디파마텍",    # GLP-1 경구용 비만치료제
        "298380": "에이비엘바이오",  # 이중항체 플랫폼
        "087010": "펩트론",          # 스마트데포 (지속형 비만치료제)
        "226950": "올릭스",          # RNA 간섭 치료제
    } 
    

    scanner = MarketScanner()
    result_df = scanner.scan(target_tickers)

    # 보기 좋게 출력 (상태가 '매수 포착'이나 '관망'인 것만 필터링 추천)
    print("\n📊 [터틀 전략 스캔 결과]")
    
    # 1. 매수 신호 뜬 종목
    buy_signals = result_df[result_df['상태'].str.contains("매수")]
    if not buy_signals.empty:
        print("\n🔥 [긴급] 오늘 매수 신호 발생!")
        print(buy_signals[['종목명', '현재가', '돌파기준가(55일고가)', '상태']])
    else:
        print("\n💨 오늘 매수 신호 없음")

    # 2. 돌파 임박 종목
    watch_list = result_df[result_df['상태'].str.contains("관망")]
    if not watch_list.empty:
        print("\n👀 [관심] 곧 돌파합니다 (D-3% 이내)")
        print(watch_list[['종목명', '현재가', '이격도(%)', '상태']])

if __name__ == "__main__":
    main()