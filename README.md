# SE Game — LLaMA Fine-Tuned AI Projects

> LLaMA 모델 파인튜닝을 활용한 캐릭터 스크립트 생성 & 코드 리뷰 AI

LLaMA 모델을 **파인튜닝(Fine-Tuning)**하여 두 가지 AI 프로젝트를 구현했습니다.

---

## Project 1: 캐릭터 기반 유튜브 스크립트 AI

유튜브용 캐릭터 스크립트를 자동 생성하는 AI입니다. 캐릭터의 성격과 말투를 반영한 파인튜닝 모델로 자연스러운 스토리텔링 스크립트를 제작합니다.

### 주요 기능
- 캐릭터별 성격/말투 반영 스크립트 자동 생성
- 유튜브 콘텐츠 스타일에 최적화된 출력
- 한국어 대화 데이터셋으로 파인튜닝

### 파인튜닝 과정
```
한국어 대화 데이터셋 (.xlsx)
        │
        ▼
  데이터 전처리 (json_trans.py)
        │
        ▼
  LLaMA Fine-Tuning (llama_fine.py)
        │
        ▼
  Ollama 모델 배포 (Modelfile)
        │
        ▼
  스크립트 생성 API
```

---

## Project 2: LLaMA 기반 코드 리뷰 AI

Python 코드를 자동으로 점검하고 개선점을 제안하는 AI입니다. 프로젝트별 코드 스타일에 맞춘 파인튜닝으로 특화된 코드 리뷰를 제공합니다.

### 주요 기능
- 코드 품질 및 잠재적 버그 검출
- 코드 스타일 일관성 점검
- 프로젝트 특화 리뷰 제공

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| Language | Python |
| AI Model | LLaMA + Fine-Tuning |
| Serving | Ollama |
| Libraries | PyTorch, Transformers |
| Dataset | 한국어 단발성 대화 데이터셋 |

## 프로젝트 구조

```
se_game/
├── llama_fine.py        # LLaMA 파인튜닝 스크립트
├── json_trans.py        # 데이터 전처리 (XLSX → JSON)
├── Modelfile            # Ollama 모델 설정
├── test.py              # 테스트 스크립트
├── games/               # 게임 스크립트 데이터
├── llama/               # LLaMA 모델 관련 파일
├── 한국어_단발성_대화_데이터셋.xlsx  # 학습 데이터
└── README.md
```

## 실행 방법

```bash
# 데이터 전처리
python json_trans.py

# 파인튜닝 실행
python llama_fine.py

# Ollama로 모델 배포
ollama create se_game -f Modelfile

# 테스트
python test.py
```
