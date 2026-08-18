# GPU 서버에서 학습하기

로컬(RTX 4050 6GB)에서 만든 것을 A100급 서버로 옮겨 본 학습을 돌리는 절차다.

## 먼저 알아둘 것

**학습은 GPU가 있는 한 곳에서만 일어난다.** 서버에서 돌리면 로컬은 논다.
두 대를 묶어 분산 학습하는 것도 가능하지만, 6GB 노트북이 A100 발목을 잡아
오히려 느려진다. 서버 하나로 몰아주는 게 맞다.

**학습 결과는 로컬에서 그대로 쓸 수 있다.** 체크포인트는 파일 하나다.
내려받아서 `train/sample.py`로 돌리면 된다. 추론은 학습보다 훨씬 가벼워서
300M 모델도 bf16이면 600MB라 6GB 노트북에서 여유롭다.

## 무엇을 옮기고 무엇을 다시 만드는가

| 대상 | 크기 | 방법 |
|---|---|---|
| 코드 | 300KB | `git clone` |
| `tokenizer/tokenizer.json` | 190KB | **git에 포함돼 있다. 절대 다시 학습하지 말 것** |
| `finetune/data/*.jsonl` | 2MB | git에 포함 |
| `data/raw` (원본 샤드) | 12GB | 서버에서 직접 다운로드 (업로드보다 빠르다) |
| `data/processed` (토큰 바이너리) | 35GB | 서버에서 재생성 |
| `.venv` | - | **옮기면 안 된다.** CUDA 버전 종속이라 새로 만든다 |
| 체크포인트 | 개당 0.6~3GB | 학습 후 로컬로 내려받는다 |

토크나이저를 다시 학습하면 어휘가 달라져 로컬 체크포인트와 호환되지 않는다.
git에 포함된 파일을 그대로 쓸 것.

## 설치

```bash
git clone https://github.com/kasangyong/sangyong_llm.git
cd sangyong_llm
```

```bash
python -m venv .venv && . .venv/bin/activate
```

서버 CUDA 버전에 맞는 torch를 설치한다. `nvidia-smi`로 확인한 CUDA 버전에
맞춰 인덱스를 고른다.

```bash
pip install torch numpy pyarrow huggingface_hub --index-url https://download.pytorch.org/whl/cu128
```

환경부터 검증한다. 통과하지 않으면 다음으로 넘어가지 않는다.

```bash
python scripts/verify_env.py
```

## 데이터 준비

전량 54샤드면 약 6.6B 토큰이 나온다. 서버 CPU 코어 수에 맞춰 `--workers`를
조절한다.

```bash
python data/download.py --shards 54
```

```bash
python data/prepare.py filter
```

`filter`는 이미 처리한 샤드를 건너뛰므로 중단됐다 다시 돌려도 안전하다.

```bash
python data/prepare.py tokenize --workers 16
```

**`data/prepare.py tokenizer`는 돌리지 말 것.** 토크나이저를 다시 학습하는
명령이라 어휘가 바뀐다. git에 있는 `tokenizer/tokenizer.json`을 쓴다.

준비가 끝나면 확인한다.

```bash
python scripts/verify_tokenizer.py
```

## 모델 크기 — A100이면 키우는 게 맞다

53M은 순전히 노트북 6GB 제약 때문에 고른 값이다. A100 40GB면 그 제약이 없다.

6.6B 토큰에 대한 Chinchilla 최적 모델 크기는 약 330M이다. 53M에 6.6B를 쓰면
데이터의 6.5배 과잉이라 수익이 크게 체감된다.

`model/transformer.py`의 `ModelConfig` 기본값을 이렇게 바꾼다:

```python
vocab_size = 16384      # 바꾸지 말 것
d_model = 1024
n_layers = 24
n_heads = 16            # head_dim 64
n_kv_heads = 4          # GQA
d_ff = 2752
max_seq_len = 2048      # A100이면 늘릴 수 있다. 코드에 유리하다
```

파라미터 약 282.6M. 6.6B 토큰과 Chinchilla 비율이 거의 맞는다.

`max_seq_len`을 바꾸면 `TrainConfig.block_size`도 같이 맞춰야 한다.

바꾼 뒤 반드시 모델 검증을 다시 돌린다. 파라미터 수 손계산 테스트가
새 설정에 맞게 갱신돼야 한다.

```bash
python tests/test_model.py
```

## 배치 크기는 실측으로 정한다

계산으로 추정하지 말고 실제로 잰다. A100은 리눅스라 VRAM 초과 시 정직하게
OOM이 나지만(Windows WDDM처럼 시스템 RAM으로 새지 않는다), 그래도 처리량이
가장 높은 지점은 재봐야 안다.

```bash
python scripts/probe_vram.py
```

`TrainConfig`의 `batch_size`와 `grad_accum`을 결과에 맞춰 조정한다.
유효 배치(= batch_size × grad_accum × block_size)는 30만~100만 토큰
범위가 무난하다.

## 학습

세션이 끊겨도 살아남도록 분리 실행한다. 스텝 수는 `train.bin` 크기에서
자동 계산된다.

```bash
python scripts/train_detached.py start
```

```bash
python scripts/train_detached.py status
```

리눅스에서는 `tmux`나 `nohup`도 같은 목적을 달성한다.

예상 시간(A100 40GB, MFU 40% 가정):

| 모델 | 6.6B 토큰 |
|---|---|
| 53M | 약 6시간 |
| 282M | 약 25시간 |

MFU는 구현과 배치에 따라 크게 달라진다. 첫 100스텝의 tok/s를 보고 다시
계산할 것.

## Colab을 쓰는 경우

Colab은 세션이 끝나면 디스크가 사라진다. 그대로 쓰면 매번 12GB를 다시 받고
몇 시간을 다시 토큰화해야 한다.

1. **Google Drive를 붙이고 거기에 데이터와 체크포인트를 둔다.**
   `data/processed`는 12GB 이상이라 무료 15GB로는 빠듯하다.
2. **체크포인트 경로를 Drive로 돌린다.** `CKPT_DIR`을 Drive 아래로 바꾸거나
   심볼릭 링크를 건다. 안 그러면 세션이 끊길 때 학습분이 날아간다.
3. **세션 시간 제한이 있다.** 25시간짜리 학습은 2~3번 나눠 재개해야 한다.
   재개는 검증돼 있다(재개 전후 손실 차이 0.00e+00).

```bash
python scripts/train_detached.py start --resume
```

`ckpt_interval`을 줄여 저장을 자주 하는 편이 안전하다. 기본값은 250스텝이다.

## 결과 가져오기

```bash
scp server:~/sangyong_llm/checkpoints/best.pt ./checkpoints/
```

로컬에서 바로 돌린다.

```bash
python train/sample.py --ckpt checkpoints/best.pt --prompt "def quicksort(xs):"
```

```bash
python eval/harness.py --ckpt checkpoints/best.pt --k 5
```

## 현재 상태 (2026-08-18)

- 검증 140항목 통과 (8개 스위트)
- 로컬에서 53M을 1B 토큰으로 250스텝까지 돌려봤다.
  train 3.7225 / val 3.7627 / ppl 43.06. 과적합 징후 없음
- 54샤드 필터 완료: 2,868,339 문서 / 24.2GB
- 로컬 토큰화 진행 중 (약 6.6B 토큰 예상)
- SFT 파이프라인과 검색/툴 레이어는 코드와 테스트가 완성됐고,
  기반 모델이 나오면 바로 붙일 수 있다
