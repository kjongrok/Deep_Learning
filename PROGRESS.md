# AI_1team 프로젝트 진행 상황

## 프로젝트 목표
- **Deep Learning 집중:** CCTV 기반 교통량/교통 상황 예측 및 분석 (Computer Vision + Anomaly Detection)
- 비전 모델(YOLO)과 이상 탐지 모델(Autoencoder 등)의 성능 비교 및 튜닝 과정을 실험하고 보고서로 작성

## 오늘 완료된 작업 요약
- [x] **프로젝트 환경 초기화 및 셋팅:** 기존 파이썬 스크립트 기반을 버리고, 보고서 작성에 특화된 Jupyter Notebook 환경(`.venv` + `requirements.txt`)으로 전면 재구축.
- [x] **통합 파이프라인 노트북 (`Total_Pipeline.ipynb`) 완료:**
  - 3개로 나뉘어 있던 노트북 병합 및 초기 `os` 모듈 import 에러 수정 완료.
  - **데이터 전처리 및 EDA:** 트래픽 변화량 및 밀집도 시각화.
  - **YOLO 벤치마크:** Nano, Small, Medium 3종 모델 다중 비교.
  - **이상 탐지 벤치마크:** Autoencoder, Isolation Forest, One-Class SVM 성능 비교 및 PCA 시각화.
- [x] **모델 저장 스크립트 (`train_and_save.py`) 작성 및 구동 완료:**
  - 주피터 노트북의 Autoencoder를 스크립트로 분리 학습.
  - `models` 폴더에 `autoencoder.keras` (모델), `scaler.pkl` (정규화 스케일러), `config.json` (임계값) 저장.
- [x] **파이썬 기반 실시간 데모 시뮬레이션 프로그램 구축 완료:**
  - 단일 이미지 판별 테스트 스크립트 (`demo.py`) 정상 작동 확인.
  - CCTV 이미지 시퀀스를 연결해 실시간 동영상처럼 무한 반복 재생하고, 그 위에 YOLO 바운딩 박스와 "정상/이상" 상태 텍스트를 실시간으로 렌더링하는 발표용 GUI 프로그램 (`demo_video.py`) 완성.

## 다음 세션에 해야 할 작업 (TODO)
- [ ] 발표 시연용 시뮬레이터(`demo_video.py`) 화면 디테일(UI) 점검 및 수정
- [ ] 실시간 CCTV API 키 발급 완료 시 API 연동 스크립트 작성
- [ ] (보류 중) 통합 파이프라인 노트북 최종 실행 및 보고서(Word/PPT)용 그래프/표 캡처본 수집하기

---

## 🚨 [필독] 새 PC 환경 셋팅 가이드 (PC 이동 시)
새로운 PC에서 이 프로젝트를 바로 구동하기 위해 아래 3가지를 반드시 지켜주세요:
1. **동일한 폴더 경로 사용:** 현재 스크립트 코드의 모든 데이터/모델 경로가 `c:\AI_1team`에 맞춰져 있습니다. 새 PC에서도 반드시 **C드라이브 직속으로 `AI_1team` 폴더**를 생성하시고 그 안에서 깃허브 코드를 내려받아 주세요.
2. **데이터셋 원상 복구:** 별도로 가져오신 `dataset` 폴더를 새 PC의 `c:\AI_1team\dataset` 위치에 똑같이 복사해 넣어주세요.
3. **라이브러리 셋팅:** 터미널에서 `pip install -r requirements.txt` 명령어를 한 번만 실행하여 기존 환경과 똑같이 파이썬 라이브러리들을 설치해 주세요.
