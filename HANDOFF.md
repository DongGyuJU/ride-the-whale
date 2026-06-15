# Smart Money Following — Phase 4 인계서

## 프로젝트 목적

한국 시장에서 **외국인/기관 수급 데이터를 활용한 알파 추출**.

학계 근거:
- Choe, Kho, Stulz (1999, JFE): 외국인은 positive feedback + herding, 가격 영향력 큼
- Choe, Kho, Stulz (2005, RFS): 외국인 정보 우위 (단, 개인은 단기 우위)
- Ahn, Kang, Ryu 등: 외국인은 earnings surprise 사전 매매

목표 IC: 0.04~0.06 (Phase 1 OHLCV-only Ridge baseline = 0.0305)

## 이전 시도 결과 (참고용)

| Phase | 모델 | Test IC | 비고 |
|---|---|---|---|
| 0 | 룰 기반 백테스트 | -47% OOS | 완전 폐기 |
| 1 | Ridge (25 OHLCV 피처) | **0.0305** | ✅ 베이스라인 |
| 1 | Random Forest | 0.0044 | 비선형성 없음 |
| 2.1 | TCN supervised | 0.0231 | 오버피팅 |
| 2.3v1 | TCN + InfoNCE contrastive | -0.0036 | positive 정의 잘못 |
| 2.3v2 | TCN + SupCon (bin) | 0.0040 | bin 정의 한계 |
| 2.5 | TCN + Wavelet + DAE | 0.0584 (누설) / -0.0073 (실제) | wavelet look-ahead bias |

**핵심 교훈**:
- OHLCV만으로는 IC 0.03이 천장
- 딥러닝 모델이 단순 Ridge 못 이김
- 노이즈 제거는 causal해야 함 (rolling)
- → **외인 수급 추가가 가장 합리적 다음 단계**

## 현재 상태

Docker 기반 깨끗한 새 코드베이스 시작. 아래까지 만들어짐:

```
smart_money/
├── Dockerfile                  ✅
├── docker-compose.yml          ✅
├── requirements.txt            ✅
├── .env.example                ✅
├── .gitignore                  ✅
├── data/
│   ├── env_loader.py           ✅ KRX_ID/PW 안전 로드
│   └── krx_loader.py           ✅ OHLCV + 수급 통합
├── features/                   ⏳ TODO
├── labels/                     ⏳ TODO
├── models/                     ⏳ TODO
├── evaluate/                   ⏳ TODO
└── scripts/                    ⏳ TODO
```

## 다음 톡방에서 진행할 작업

### Stage 1: 데이터 다운로드 + 검증 (오늘)
- [ ] `scripts/01_download.py` 작성
  - KOSPI / KOSDAQ 분리 다운로드
  - 기간: 2018-01-01 ~ 2024-12-31 (7년)
- [ ] `scripts/02_diagnose.py` — 다운로드 후 데이터 품질 점검
  - 외인 수급 0이 아닌 종목 비율
  - 시장별 종목 수
  - NaN / 이상치
- [ ] 디케이님 실제 실행 → KRX 로그인 + 다운로드 시간 확인

### Stage 2: 피처 엔지니어링 + Ridge baseline
- [ ] `features/technical.py` — 모멘텀/MA/RSI/BB (Phase 1과 유사, 25개)
- [ ] `features/supply.py` — 수급 피처 ~20개:
  - 외인/기관 누적 (5d, 10d, 20d)
  - 거래량/시총 대비 정규화
  - Concordance (외인+기관 합의)
  - 연속 매수일 streak
  - Cross-sectional z-score
- [ ] `features/pipeline.py` — 전체 빌드
- [ ] `labels/forward_return.py` — 5/10/20일 forward return + IC 평가
- [ ] `models/ridge.py` — Ridge 학습
- [ ] `models/temporal_split.py` — train/val/test 시간 분할
- [ ] `evaluate/ic.py` — Spearman IC + IR
- [ ] `evaluate/long_short.py` — 10% top/bottom 포트폴리오 수익률
- [ ] `scripts/03_train_ridge.py` — KOSPI / KOSDAQ 각각 학습

**목표**: 
- KOSPI Ridge IC vs KOSDAQ Ridge IC 비교
- OHLCV만 vs OHLCV+수급 ablation

### Stage 3: LightGBM + 변수 중요도
- [ ] `models/lightgbm.py`
- [ ] `evaluate/attribution.py` — 변수 중요도 분석
- [ ] `scripts/04_train_lgb.py`

**목표**:
- LightGBM이 Ridge 이기는지
- 어떤 수급 피처가 진짜 알파인지 (foreign vs inst vs concordance)

### Stage 4: 단순 룰 백테스트
- [ ] "외인 5일 연속 매수 + 60일선 위" 룰 백테스트
- [ ] ML 알파 vs 룰 알파 비교
- [ ] 둘 다 작동하면 앙상블

## 환경 설정 (Docker)

```bash
cd /root/projects/smart_money  # 또는 디케이님 서버 위치
cp .env.example .env
# .env 편집해서 KRX_ID, KRX_PW 입력

# 빌드 (한 번만)
docker compose build

# 실행
docker compose run --rm cli python3 scripts/01_download.py

# Jupyter 사용 시
docker compose up -d jupyter
# 브라우저에서 http://server-ip:8888
```

## 알려진 위험 & 대응

1. **KRX 로그인 차단**: 도커 IP가 호스트 IP와 같으면 이전과 동일 차단 가능
   - 대응: 첫 다운로드는 천천히 (workers=2), 자정 이후 시도
2. **공개 수급 = 알파 빨리 소멸**: T일 외인 매수 발표 → T+1엔 이미 가격 반영
   - 대응: forward return을 T+5 이후로 설정 (즉시 반응 제외)
3. **Size factor 함정**: 외인 매수 = 대형주 = 단순 size factor일 수 있음
   - 대응: 시총 control + market_cap 그룹별 분리 분석
4. **KOSDAQ 데이터 품질**: 거래 적은 종목 수급 노이즈 큼
   - 대응: universe 필터 (시총 하위 30% 제외, 평균 거래대금 임계값)

## 중요 결정 사항 (확정됨)

- [x] Docker 환경
- [x] KOSPI / KOSDAQ 분리 실험
- [x] 외인 수급 우선, 미장은 OHLCV만 (외인 데이터 없음)
- [x] 점진 구축 (각 단계 IC 측정)

## 디케이님 컨텍스트 (참고)

- 본업: spine reconstruction (VerSe2020), Clifford QML 논문 진행 중
- 시드: 200만원 (전체 자산 1500만 중 일부)
- KRX 계정 보유 (doch0216), 한국 증권 계좌 보유
- 코랩 사용 가능 (T4 GPU 무료), 도커 서버 사용 가능
- 직접 개발 환경 다룰 수 있음 (PhD level)
