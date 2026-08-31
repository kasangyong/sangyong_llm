# GPU 서버에서 학습하기

로컬(RTX 4050 6GB)에서 만든 것을 GPU 서버로 옮겨 본 학습을 돌리는 절차다.

실제로 쓰는 서버는 **Tesla V100S-PCIE-32GB 10장**(jupyter03, Ubuntu 22.04.5)이고,
그중 1장(`CUDA_VISIBLE_DEVICES=3`)을 쓴다. 아래 수치는 2026-08-31 그 서버에서
실측한 값이다. 다른 서버를 쓴다면 전부 다시 재야 한다.

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

**V100(sm_70)에서는 최신 PyTorch를 쓸 수 없다.** 2.11+는 Volta를 빌드 타겟에서
뺐다(최소 sm_75). cu128 휠을 깔면 설치는 되는데 실행 시점에 죽는다:

```
CUDA error: no kernel image is available for execution on the device
```

sm_70이 빌드에 들어 있는 마지막 조합은 **torch 2.6.0 + cu124**다.
실측 확인: `['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`.

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch==2.6.0+cu124" numpy pyarrow huggingface_hub
```

Ampere(sm_80) 이상 서버라면 최신 cu128을 그대로 쓰면 된다.

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

## 정밀도 — V100에서 bf16은 함정이다

이 레포는 원래 bf16 autocast로 짜여 있었다. V100에는 bf16 텐서코어가 없고
소프트웨어 에뮬레이션만 있는데, **그 에뮬레이션은 fp32보다도 느리다.**

4096x4096 matmul 실측 (V100S-PCIE-32GB, 2026-08-31):

| 정밀도 | TFLOPS | fp32 대비 |
|---|---:|---:|
| fp32 | 13.2 | 1.0x |
| **fp16** | **88.8** | **6.7x** |
| bf16 | 10.0 | 0.76x |

더 나쁜 것은 `torch.cuda.is_bf16_supported()`가 V100에서도 **True를 반환한다**는
점이다. 지원 여부만 보는 검증은 초록불을 켜고, 학습만 조용히 8분의 1 속도로
돈다. 그래서 정밀도는 `model/precision.py`가 **compute capability로** 고른다
(sm_80 이상 bf16 / 미만 fp16). `scripts/verify_env.py`도 지원 여부가 아니라
실측 TFLOPS가 fp32보다 빠른지를 판정 근거로 쓴다.

fp16을 쓰면 GradScaler가 따라온다. 기울기 클리핑은 반드시 `scaler.unscale_()`
**뒤에** 와야 한다. 순서가 바뀌면 유효 학습률이 스케일 배수만큼 무너지는데
손실 곡선에는 아무 징후가 없다(실측: 실제 기울기 노름 0.25 -> 0.0078).
`tests/test_training.py`의 "fp16 클리핑은 unscale 뒤에"가 이걸 고정한다.

정밀도는 자동으로 정해지지만 강제할 수도 있다.

```bash
python train/train.py --precision fp16
```

## 모델 크기 — 32GB면 키우는 게 맞다

53M은 순전히 노트북 6GB 제약 때문에 고른 값이다. V100S 32GB면 그 제약이 없다.

6.6B 토큰에 대한 Chinchilla 최적 모델 크기는 약 330M이다. 53M에 6.6B를 쓰면
데이터의 6.5배 과잉이라 수익이 크게 체감된다.

설정을 손으로 고칠 필요는 없다. `model/transformer.py`에 프리셋이 들어 있다.

| 프리셋 | d_model | 층 | heads | KV | d_ff | context | 파라미터 |
|---|---|---|---|---|---|---|---|
| `53m` | 640 | 10 | 10 | 2 | 1728 | 1024 | 53,507,200 |
| `282m` | 1024 | 24 | 16 | 4 | 2752 | 2048 | 282,641,408 |

`--model`로 고른다. `TrainConfig.block_size`는 프리셋의 `max_seq_len`으로
자동으로 맞춰진다 — 손으로 두 곳을 고치다 어긋나는 사고를 막기 위해서다.

```bash
python train/train.py --model 282m
```

`vocab_size`는 어느 프리셋도 정하지 않는다. 항상 `tokenizer/tokenizer.json`에서
읽는다. 16,384가 임베딩 크기와 묶여 있어 바꾸면 체크포인트가 전부 무용지물이
되기 때문이다.

프리셋을 손대면 파라미터 수 손계산 테스트가 잡는다. 282,641,408을 못으로
박아뒀다.

```bash
python tests/test_model.py
```

## 배치 크기는 실측으로 정한다

계산으로 추정하지 말고 실제로 잰다. 리눅스는 VRAM 초과 시 정직하게
OOM이 나지만(Windows WDDM처럼 시스템 RAM으로 새지 않는다), 그래도 처리량이
가장 높은 지점은 재봐야 안다.

```bash
python scripts/probe_vram.py --model 282m
```

`TrainConfig`의 `batch_size`와 `grad_accum`을 결과에 맞춰 조정한다.
프로브가 권장 유효 배치(282m 기준 524,288 토큰/스텝)에 맞는 누적 수와
2.68B / 6.6B 토큰 예상 시간까지 같이 계산해준다.
유효 배치(= batch_size × grad_accum × block_size)는 30만~100만 토큰
범위가 무난하다.

## 학습

세션이 끊겨도 살아남도록 분리 실행한다. 스텝 수는 `train.bin` 크기와
`--epochs`에서 자동 계산된다.

```bash
python scripts/train_detached.py start --model 282m
```

```bash
python scripts/train_detached.py status
```

POSIX에서는 `start_new_session`으로 세션을 새로 파서 SSH가 끊길 때 오는
SIGHUP을 안 받는다. `tmux`나 `nohup`도 같은 목적을 달성하므로, 이미 tmux를
쓰고 있으면 그 안에서 `train.py`를 직접 돌려도 된다.

**`--model`을 빼면 기본값 53m으로 돈다.** 헤더에 모델 이름과 파라미터 수가
찍히니 첫 줄을 확인할 것.

예상 시간(V100S 32GB 1장, fp16, MFU 30~40% **가정** — 실측 아님):

| 모델 | 2.68B 토큰 | 6.6B 토큰 |
|---|---|---|
| 53M | 약 15~25시간 | 약 37~61시간 |
| 282M | 약 42시간 | 약 4.3일 |

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

## 현재 상태 (2026-08-31)

- 검증 138항목 통과 (8개 스위트). fp16 전환, 282m 프리셋, 회귀 테스트 포함
- 로컬에서 53M을 1B 토큰으로 250스텝까지 돌려봤다.
  train 3.7225 / val 3.7627 / ppl 43.06. 과적합 징후 없음
- 54샤드 필터 완료: 2,868,339 문서 / 24.2GB
- 로컬 토큰화는 2,677,510,268 토큰(train) 지점에서 중단됐다. 파일은 문서
  경계에서 끝나 있어 그대로 쓸 수 있다. 전량은 약 6.6B 예상
- `data/prepare.py tokenize`는 재개가 안 된다(출력을 "wb"로 연다).
  다시 돌리면 처음부터 쓴다
- SFT 파이프라인과 검색/툴 레이어는 코드와 테스트가 완성됐고,
  기반 모델이 나오면 바로 붙일 수 있다
