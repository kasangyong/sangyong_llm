# sangyong_llm

파이썬 코드를 생성하는 53M 파라미터 트랜스포머. 밑바닥부터 직접 구현.

PyTorch에서 빌려 쓰는 것은 텐서 연산 / autograd / CUDA / 융합 어텐션 커널
네 가지뿐이다. 토크나이저, 어텐션, RoPE, RMSNorm, 학습 루프, 데이터
파이프라인, 평가 채점기는 전부 직접 작성했다. `transformers`,
`tokenizers`, `datasets` 라이브러리는 쓰지 않는다.

설계 근거와 시행착오는 [docs/design/2026-08-13-sangyong-llm-design.md](docs/design/2026-08-13-sangyong-llm-design.md)에 있다.

## 구성

| 경로 | 역할 |
|---|---|
| `tokenizer/bpe.py` | 바이트 단위 BPE. 학습/인코딩/디코딩/저장 |
| `model/transformer.py` | RoPE, RMSNorm, GQA 어텐션, SwiGLU, KV 캐시 생성 |
| `data/download.py` | codeparrot-clean 샤드 다운로드 |
| `data/prepare.py` | 필터 → 토크나이저 학습 → 토큰화 (3단계) |
| `train/train.py` | 프리트레이닝 루프 (bf16, 기울기 누적, 체크포인트) |
| `train/sample.py` | 학습된 모델로 코드 생성 |
| `eval/harness.py` | 문법 유효율 + pass@k 채점 |
| `finetune/make_dataset.py` | 코퍼스에서 (독스트링 → 함수) 지시-응답 쌍 추출 |
| `finetune/format.py` | `### 지시:` / `### 코드:` 프롬프트 포맷 |
| `finetune/dataset.py` | SFT 데이터셋. 프롬프트 구간 손실 마스킹, 에포크 순회 |
| `finetune/sft.py` | 인스트럭션 튜닝 루프 (프리트레이닝 lr의 1/10) |
| `tools/protocol.py` | `### 검색:` 툴 호출 파싱, 검색 결과 컨텍스트 포맷 |
| `tools/search.py` | Brave / Tavily / Serper 클라이언트 |
| `tools/pipeline.py` | 생성 → 툴 호출 → 검색 → 재주입 루프 |
| `scripts/verify_env.py` | GPU/CUDA 환경 검증 |
| `scripts/verify_tokenizer.py` | 실전 토크나이저를 held-out 코퍼스로 검증 |
| `scripts/probe_vram.py` | 안전한 batch_size 실측 (시스템 RAM 유출 탐지) |
| `scripts/train_detached.py` | 학습을 세션과 분리해 실행 / 상태 확인 / 중단 |
| `scripts/run_tests.py` | 전체 검증 일괄 실행 |

## 모델

| 항목 | 값 |
|---|---|
| 파라미터 | 53,507,200 |
| d_model / layers / heads | 640 / 10 / 10 |
| KV heads (GQA) | 2 |
| FFN | SwiGLU, d_ff 1728 |
| vocab / context | 16,384 / 1,024 |

## 실행

환경은 Python 3.12 venv + PyTorch cu128을 쓴다. Python 3.14에는 CUDA
휠이 없다.

```bash
uv venv --python 3.12 .venv
uv pip install torch numpy pyarrow huggingface_hub --index-url https://download.pytorch.org/whl/cu128
```

전체 검증부터 돌린다. 통과하지 않으면 다음으로 넘어가지 않는다.

```bash
.venv/Scripts/python.exe scripts/run_tests.py
```

데이터를 준비한다. 샤드 8개면 약 1B 토큰, 전량(54개)이면 약 6.6B 토큰이다.
`filter`는 이미 처리한 샤드를 건너뛰므로 나눠서 받아도 된다.

```bash
.venv/Scripts/python.exe data/download.py --shards 54
```

```bash
.venv/Scripts/python.exe data/prepare.py filter
```

```bash
.venv/Scripts/python.exe data/prepare.py tokenizer --sample-mb 100
```

```bash
.venv/Scripts/python.exe data/prepare.py tokenize
```

학습한다. 스텝 수는 `train.bin` 크기와 `--epochs`로 자동 계산된다.

며칠 걸리는 작업이므로 터미널·세션과 분리해서 띄운다. 부모에 묶여 있으면
창을 닫거나 접속이 끊길 때 같이 죽는다.

```bash
.venv/Scripts/python.exe scripts/train_detached.py start
```

```bash
.venv/Scripts/python.exe scripts/train_detached.py status
```

```bash
.venv/Scripts/python.exe scripts/train_detached.py stop
```

중단한 뒤에는 `--resume`으로 이어진다. 모델 가중치뿐 아니라 옵티마이저
모멘텀과 스텝 수까지 복원하므로 궤적이 끊기지 않는다.

```bash
.venv/Scripts/python.exe scripts/train_detached.py start --resume
```

노트북이 절전에 들어가면 학습도 멈췄다가 깨어날 때 이어진다. 데이터는
망가지지 않지만 벽시계 시간이 그만큼 늘어난다. 며칠 돌릴 거면 전원 설정에서
절전을 꺼두는 편이 낫다.

생성해 본다.

```bash
.venv/Scripts/python.exe train/sample.py --prompt "def quicksort(xs):" --num-samples 3
```

채점한다.

```bash
.venv/Scripts/python.exe eval/harness.py --ckpt checkpoints/best.pt --k 5
```

## 인스트럭션 튜닝 (SFT)

**프리트레이닝이 아직 안 끝나서 실제 SFT는 못 돌린다.** `--base`가 가리키는
`checkpoints/best.pt`가 없으면 시작하지 않는다. 아래는 기반 모델이 나온 뒤의
절차이고, 지금 검증된 것은 데이터셋 생성과 학습 루프의 동작뿐이다(CPU 초소형
모델로 확인).

데이터셋은 필터링된 코퍼스에서 (독스트링 → 함수 본문) 쌍을 뽑아 만든다.
현재 `finetune/data/`에 train 2,845 / val 155 샘플이 들어 있다.

```bash
.venv/Scripts/python.exe finetune/make_dataset.py build --target 5000
```

```bash
.venv/Scripts/python.exe finetune/make_dataset.py peek --n 3
```

```bash
.venv/Scripts/python.exe finetune/sft.py --base checkpoints/best.pt
```

포맷은 특수 토큰 없이 일반 텍스트 마커를 쓴다. 어휘 16,384는 프리트레이닝과
동일해야 하므로 SFT는 토크나이저를 건드리지 않는다. 어휘가 어긋나면 시작
시점에 멈춘다.

```
### 지시:
<지시문>

### 코드:
<코드>
```

손실은 `### 코드:` 뒤 완성 구간에서만 계산한다. 프롬프트 구간과 패딩은
라벨 -1로 마스킹한다.

기본 저장 경로는 `checkpoints/sft/`다. `--out-dir`로 `checkpoints/`를 직접
주면 거부한다 — `latest.pt`를 덮어써 프리트레이닝을 날리기 때문이다.

## 검색/툴 레이어

**실제 검색에는 API 키가 필요하다.** 키가 없으면 `MissingAPIKeyError`로
멈춘다. 제공자별 환경변수는 `BRAVE_SEARCH_API_KEY`(또는 `BRAVE_API_KEY`),
`TAVILY_API_KEY`, `SERPER_API_KEY`다. 테스트는 HTTP 함수를 주입해 돌리므로
키도 네트워크도 쓰지 않는다.

툴 호출은 줄 머리의 한 줄짜리 마커다. 닫을 것이 없어 53M 모델이 배우기
쉽고, `### 지시:`와 같은 모양이다.

```
### 검색: <질의>
### 검색결과:
[1] 제목 (url)
    스니펫
### 답변:
```

라이브러리로만 제공한다. CLI는 없다.

```python
from tools.pipeline import run_with_search
from tools.search import make_client

client = make_client("brave")            # 키 없으면 여기서 예외
result = run_with_search(
    prompt,
    generate=my_generate,                # generate(prompt) -> str
    search=client.search,                # search(query) -> list[SearchResult]
    max_calls=2,
    max_context=cfg.max_seq_len - max_new_tokens,
)
```

`max_context`를 안 주면 프롬프트가 라운드마다 누적되어 2~3라운드에서
`max_seq_len`(1,024)을 넘긴다. 자리가 없으면 검색을 쏘기 전에
`stop_reason="context_full"`로 멈춘다.

검색 결과는 신뢰하지 않는다. 스니펫/제목/URL이 줄 머리에 마커를 만들면
앞에 공백을 넣어 무력화한다. 안 그러면 검색 제공자가 다음 질의를 고른다.

## 검증 원칙

각 단계는 먼저 깨뜨려보고, 통과한 것만 다음으로 넘긴다.

| 스위트 | 항목 수 | 무엇을 잡는가 |
|---|---|---|
| `verify_env.py` | 9 | GPU 행렬곱 정확도, bf16, GQA SDPA, 가용 VRAM |
| `test_tokenizer.py` | 13 | 적대적 입력 28종, 무작위 유니코드 1000건, 어휘 크기 불변 |
| `test_model.py` | 14 | 인과 마스크 누설, RoPE 상대위치, KV 캐시 등가 |
| `test_training.py` | 8 | 기울기 누적 등가, 단일배치 과적합, 재개 궤적 일치 |
| `test_eval.py` | 12 | 정답/오답 판별, `sys.exit(0)` 우회 차단 |
| `test_sft.py` | 23 | 손실 마스킹 경계, 어휘 불변, 평가가 에포크를 갉아먹는지 |
| `test_tools.py` | 47 | 파서 경계, 검색 오류 구분, 마커 위조, 컨텍스트 예산 |
| `test_regress_correctness.py` | 7 | 적대적 검증에서 재현된 결함 7종의 회귀 고정 |

특히 **인과 마스크 누설**은 손실 곡선만 봐서는 절대 못 잡는다. 뚫려
있으면 손실은 예쁘게 떨어지지만 생성은 전혀 안 된다.

## 실측치 (RTX 4050 Laptop 6GB)

| 항목 | 값 |
|---|---|
| 데이터 | codeparrot-clean 8샤드 → 419,773 문서 / 3.63GB (유지율 52.5%) |
| 토큰 | train 1,027,146,559 / val 2,103,160 |
| 토크나이저 | 병합 16,127개, 압축률 3.763 바이트/토큰 |
| 배치 | batch 4 × 누적 32 = 131,072 토큰/스텝 (peak VRAM 4.45GB) |
| 처리량 | 5,601 tok/s |
| 1에포크 | 7,800스텝 ≈ **51시간** |

### VRAM 주의사항

Windows(WDDM)는 VRAM이 모자라도 **OOM을 내지 않는다.** 드라이버가 조용히
시스템 RAM으로 흘려보내서 "돌긴 도는데 20배 느린" 상태가 된다.
`scripts/probe_vram.py`가 이걸 잡아준다.

```bash
.venv/Scripts/python.exe scripts/probe_vram.py
```

| batch | peak | 초/스텝 | 판정 |
|---|---|---|---|
| 2 | 2.59GB | 0.818 | OK |
| 4 | 4.45GB | 1.585 | OK (최적) |
| 6 | 6.30GB | - | 안전선 초과 → 시스템 RAM 유출 |

## 한계

- **6GB VRAM이 상한이다.** 1B 파라미터는 안 들어가고, 들어간다 해도 약
  9개월 걸린다. Claude Opus 5급(학습비 $200M~500M)은 범위 밖이다.
- GPU를 다른 작업과 공유하면 클럭이 3105MHz에서 645MHz까지 떨어지고
  처리량이 절반 이하가 된다. 학습 전에 `nvidia-smi`로 확인할 것.
- 평가 격리는 별도 프로세스 + 타임아웃 수준이다. 컨테이너나 seccomp를 쓴
  진짜 샌드박스는 아니다.
- **SFT는 아직 한 번도 못 돌렸다.** 기반 모델(`checkpoints/best.pt`)이
  프리트레이닝 중이라 없다. 코드와 데이터셋은 준비됐고 CPU 초소형 모델로만
  검증한 상태다.
- 검색은 API 키가 있어야 실제로 나간다. 키 없이 돌아가는 것은 테스트뿐이다.
- 툴 호출 마커는 한국어라 이 토크나이저에서 9~15토큰을 먹는다. 같은 뜻의
  ASCII 마커는 4~5토큰이다. SFT 시작 전이면 바꾸는 편이 컨텍스트 예산에
  유리하다(`tools/protocol.py` 상수 세 줄).
