"""프리트레이닝 루프. 직접 구현.

노트북 GPU에서 도는 것을 전제로 짰다:
  - 혼합정밀도 autocast (GPU에 맞춰 bf16/fp16 자동 선택 — model/precision.py)
  - 기울기 누적으로 유효 배치를 키운다
  - 매 N스텝 체크포인트. 중단은 사고가 아니라 기본 전제다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from model.precision import autocast, label, needs_scaler
from model.precision import resolve as resolve_precision
from model.transformer import PRESETS, ModelConfig, Transformer, make_config

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class TrainConfig:
    # 유효 배치 = batch_size * grad_accum * block_size 토큰 (= 131,072)
    #
    # batch_size는 scripts/probe_vram.py 실측으로 정한다. Windows(WDDM)에서는
    # VRAM을 넘겨도 OOM이 안 나고 시스템 RAM으로 조용히 흘러 10배 이상
    # 느려지므로, 안전선(전체 VRAM의 85%) 안에 들어가는 값을 써야 한다.
    # 실측(6GB 전용): batch 4가 peak 4.45GB로 안전선(5.10GB) 안에 들어가고
    # 처리량도 가장 높다. batch 6은 6.30GB로 시스템 RAM에 유출된다.
    batch_size: int = 4
    grad_accum: int = 32
    block_size: int = 1024

    lr: float = 6e-4
    min_lr_frac: float = 0.1  # 최종 lr = lr * min_lr_frac
    # 총 스텝의 1% 안팎. 5만 스텝 규모에서 200스텝(0.4%)은 짧은 편이라
    # 초반 기울기가 튀기 쉽다. 늘려도 비용은 거의 없다.
    warmup_iters: int = 500
    # 실제 값은 train()에서 train.bin 크기와 --epochs로 계산해 덮어쓴다.
    # 여기 값은 계산 전에 참조될 때를 위한 자리표시자다.
    max_iters: int = 7800
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # 평가 한 번에 eval_iters * 2(split) 회 forward가 든다. 약 50,000스텝짜리
    # 작업에서 250스텝마다 평가하면 평가에만 10시간을 쓴다. 1,000스텝이면
    # 약 1시간으로 줄면서 손실 곡선을 보기엔 충분히 촘촘하다.
    eval_interval: int = 1000
    eval_iters: int = 50
    log_interval: int = 10
    # 며칠짜리 작업이라 체크포인트를 자주 남겨야 장애 때 잃는 게 적다.
    # 250스텝은 약 55분치이고, 저장 비용은 612MB 쓰기 수 초다.
    ckpt_interval: int = 250

    seed: int = 1337
    # "auto"면 GPU에 맞춰 고른다(model/precision.py). Ampere 이상은 bf16,
    # V100(sm_70)급은 fp16. 강제하려면 bf16 / fp16 / fp32.
    precision: str = "auto"
    compile_model: bool = False  # Windows에서는 대체로 불안정하다

    @property
    def tokens_per_iter(self) -> int:
        return self.batch_size * self.grad_accum * self.block_size


def lr_at(it: int, cfg: TrainConfig) -> float:
    """워밍업 후 코사인 감쇠."""
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    if it >= cfg.max_iters:
        return cfg.lr * cfg.min_lr_frac
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * coeff)


class BinDataset:
    """uint16 토큰 바이너리에서 무작위 구간을 뽑는다."""

    def __init__(self, path: Path, block_size: int):
        if not path.exists():
            raise FileNotFoundError(f"토큰 바이너리가 없다: {path}")
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.data) < block_size + 1:
            raise ValueError(
                f"{path.name}의 토큰이 너무 적다: {len(self.data):,} < {block_size + 1}"
            )

    def __len__(self):
        return len(self.data)

    def close(self):
        """memmap을 놓아준다. Windows는 열려 있는 동안 파일을 잠그기 때문에
        학습 중 데이터를 다시 만들거나 교체하려면 이게 필요하다."""
        data = getattr(self, "data", None)
        if data is not None:
            mm = getattr(data, "_mmap", None)
            if mm is not None:
                mm.close()
            self.data = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def batch(self, batch_size: int, device, generator=None):
        hi = len(self.data) - self.block_size - 1
        ix = torch.randint(hi, (batch_size,), generator=generator)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.block_size].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    self.data[i + 1 : i + 1 + self.block_size].astype(np.int64)
                )
                for i in ix
            ]
        )
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def make_optimizer(model: Transformer, cfg: TrainConfig):
    """2차원 이상 파라미터에만 weight decay를 적용한다.

    정규화 가중치나 편향에 decay를 걸면 성능이 나빠진다는 것이 관례적으로
    확인돼 있다.
    """
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused 커널은 CUDA에서만 쓸 수 있다
    fused = torch.cuda.is_available() and all(
        p.is_cuda for g in groups for p in g["params"]
    )
    return torch.optim.AdamW(
        groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=fused
    )


@torch.no_grad()
def estimate_loss(model, datasets, cfg, device, generator=None, amp_dtype=None):
    if amp_dtype is None:
        amp_dtype, _ = resolve_precision(device, getattr(cfg, "precision", "auto"))
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.batch(cfg.batch_size, device, generator)
            with autocast(amp_dtype):
                _, loss, _ = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def clip_and_step(scaler, optimizer, model, grad_clip: float):
    """기울기를 자르고 한 스텝 밟는다. (gnorm, 실제로 밟았는지)를 돌려준다.

    unscale_을 클리핑보다 **먼저** 불러야 한다. 순서가 바뀌면 손실 곡선에
    아무 징후 없이 유효 학습률이 무너진다. 스케일된 기울기(노름이 scale배)를
    grad_clip으로 자르니 1/scale에 가까운 계수가 곱해지고, 그 뒤 unscale_이
    한 번 더 나누기 때문이다. 실측(scale 128, grad_clip 1.0): 실제 기울기
    노름이 0.25 -> 0.0078로 32배 줄었다. 보고되는 gnorm은 반대로 128배
    부풀려져서 로그만 봐서는 "기울기가 큰 구간"으로 읽힌다.

    scaler가 꺼져 있으면(bf16/fp32) unscale_과 step은 그대로 통과한다.
    """
    scaler.unscale_(optimizer)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # inf/nan이 섞인 스텝은 scaler가 건너뛰고 스케일을 낮춘다. 건너뛴 횟수를
    # 세어두지 않으면 "학습은 도는데 진도가 안 나가는" 상태를 못 알아챈다.
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = not scaler.is_enabled() or scaler.get_scale() >= scale_before
    return gnorm, stepped


def save_checkpoint(path: Path, model, optimizer, mcfg, tcfg, it, best_val, scaler=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(mcfg),
        "train_config": asdict(tcfg),
        "iter": it,
        "best_val": best_val,
    }
    if scaler is not None and scaler.is_enabled():
        # fp16 손실 스케일도 학습 상태다. 빠뜨리면 재개 직후 스케일이 초기값
        # (65536)으로 되돌아가 몇 스텝이 오버플로로 버려지고 궤적이 어긋난다.
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, tmp)
    tmp.replace(path)  # 저장 중 죽어도 기존 체크포인트가 안 깨지도록


def load_checkpoint(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg = TrainConfig()
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.grad_accum:
        tcfg.grad_accum = args.grad_accum
    if args.precision:
        tcfg.precision = args.precision

    # 정밀도는 GPU 세대가 정한다. is_bf16_supported()는 V100에서도 True를
    # 돌려주므로 근거로 쓸 수 없다 (model/precision.py에 실측치 있음).
    amp_dtype, precision_why = resolve_precision(device, tcfg.precision)
    tcfg.precision = label(amp_dtype)
    scaler = torch.amp.GradScaler(device, enabled=needs_scaler(amp_dtype))

    torch.manual_seed(tcfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 토크나이저에서 실제 어휘 크기를 읽어 모델 설정과 강제로 일치시킨다.
    tok_path = ROOT / "tokenizer" / "tokenizer.json"
    if not tok_path.exists():
        raise SystemExit("토크나이저가 없다. data/prepare.py tokenizer를 먼저 돌릴 것.")
    with open(tok_path, encoding="utf-8") as f:
        payload = json.load(f)
    vocab_size = 256 + len(payload["merges"]) + len(payload["specials"]) + payload.get(
        "n_reserved", 0
    )

    mcfg = make_config(args.model, vocab_size)
    # block_size와 모델 컨텍스트가 어긋나면 RoPE 캐시 범위를 넘거나(크면)
    # 컨텍스트를 놀리게(작으면) 된다. 모델 쪽을 진실로 삼는다.
    tcfg.block_size = mcfg.max_seq_len
    train_ds = BinDataset(PROC_DIR / "train.bin", tcfg.block_size)
    val_ds = BinDataset(PROC_DIR / "val.bin", tcfg.block_size)
    datasets = {"train": train_ds, "val": val_ds}

    # 스텝 수는 실제 데이터 크기에서 계산한다. 하드코딩하면 코퍼스를 늘릴 때마다
    # 조용히 어긋나고, lr 스케줄이 데이터 끝나기 한참 전에 최소값에 닿거나
    # 반대로 감쇠를 다 못 쓴 채 끝난다.
    if args.max_iters:
        tcfg.max_iters = args.max_iters
    else:
        tcfg.max_iters = max(1, int(args.epochs * len(train_ds) / tcfg.tokens_per_iter))

    resume_path = CKPT_DIR / "latest.pt"
    start_iter = 0
    best_val = float("inf")
    if args.resume and resume_path.exists():
        model, ck = load_checkpoint(resume_path, device)
        mcfg = ModelConfig(**ck["model_config"])
        # 체크포인트가 --model과 다른 크기면 여기서 멈춘다. 그냥 두면 데이터
        # 슬라이싱만 어긋난 채로 학습이 계속 돌아간다.
        if mcfg.max_seq_len != tcfg.block_size:
            raise SystemExit(
                f"체크포인트 컨텍스트 {mcfg.max_seq_len} != --model {args.model}의 "
                f"{tcfg.block_size}. 같은 크기로 재개하거나 --resume을 뺄 것."
            )
        optimizer = make_optimizer(model, tcfg)
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck["iter"] + 1
        best_val = ck["best_val"]
        if scaler.is_enabled() and "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        print(f"[resume] {resume_path.name}에서 iter {start_iter}부터 재개")
    else:
        model = Transformer(mcfg).to(device)
        optimizer = make_optimizer(model, tcfg)

    n_params = model.num_params()
    print("=" * 60)
    print(f"장치        : {device}")
    print(f"어휘        : {vocab_size:,}")
    print(f"모델        : {args.model} (d_model {mcfg.d_model}, {mcfg.n_layers}층, "
          f"context {mcfg.max_seq_len})")
    print(f"파라미터    : {n_params:,}")
    print(f"학습 토큰   : {len(train_ds):,} / 검증 {len(val_ds):,}")
    print(f"정밀도      : {tcfg.precision} ({precision_why})")
    print(f"유효 배치   : {tcfg.tokens_per_iter:,} 토큰/스텝")
    print(f"스텝        : {tcfg.max_iters:,} (총 {tcfg.max_iters * tcfg.tokens_per_iter / 1e9:.2f}B 토큰)")
    print("=" * 60)

    log_path = CKPT_DIR / "trainlog.jsonl"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()
    # 마지막 로그 이후 지난 스텝 수. log_interval로 고정해서 나누면 첫 줄이
    # 10배 부풀려진 처리량을 보고한다 (그때는 1스텝만 지났으므로).
    iters_since_log = 0
    # fp16에서 오버플로가 난 스텝은 scaler가 통째로 버린다. 몇 번 버려졌는지
    # 세어두지 않으면 "돌긴 도는데 진도가 안 나가는" 상태를 못 알아챈다.
    n_skipped = 0

    for it in range(start_iter, tcfg.max_iters):
        lr = lr_at(it, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_ds.batch(tcfg.batch_size, device)
            with autocast(amp_dtype):
                _, loss, _ = model(x, targets=y)
            # 누적 스텝 수로 나눠야 전체 배치 평균과 같아진다
            scaler.scale(loss / tcfg.grad_accum).backward()
            total_loss += loss.item() / tcfg.grad_accum

        gnorm, stepped = clip_and_step(scaler, optimizer, model, tcfg.grad_clip)
        if not stepped:
            n_skipped += 1
        iters_since_log += 1

        if it % tcfg.log_interval == 0:
            dt = time.time() - t0
            tps = tcfg.tokens_per_iter * iters_since_log / max(dt, 1e-9)
            mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0
            skip_note = f" | skip {n_skipped}" if n_skipped else ""
            print(
                f"iter {it:6d} | loss {total_loss:.4f} | lr {lr:.2e} "
                f"| gnorm {gnorm:.2f} | {tps:,.0f} tok/s | vram {mem:.2f}GB{skip_note}",
                flush=True,
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "iter": it,
                            "loss": total_loss,
                            "lr": lr,
                            "gnorm": float(gnorm),
                            "skipped": n_skipped,
                        }
                    )
                    + "\n"
                )
            t0 = time.time()
            iters_since_log = 0

        if it > 0 and it % tcfg.eval_interval == 0:
            losses = estimate_loss(model, datasets, tcfg, device, amp_dtype=amp_dtype)
            ppl = math.exp(min(losses["val"], 20))
            print(
                f"  [eval] iter {it} train {losses['train']:.4f} "
                f"val {losses['val']:.4f} ppl {ppl:.2f}",
                flush=True,
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"iter": it, "eval": losses, "val_ppl": ppl}) + "\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    CKPT_DIR / "best.pt", model, optimizer, mcfg, tcfg, it, best_val,
                    scaler=scaler,
                )

        if it > 0 and it % tcfg.ckpt_interval == 0:
            save_checkpoint(
                resume_path, model, optimizer, mcfg, tcfg, it, best_val, scaler=scaler
            )

    save_checkpoint(
        resume_path, model, optimizer, mcfg, tcfg, tcfg.max_iters - 1, best_val,
        scaler=scaler,
    )
    print(f"\n학습 종료. best val loss = {best_val:.4f}")
    if n_skipped:
        print(f"fp16 오버플로로 건너뛴 스텝: {n_skipped:,} / {tcfg.max_iters - start_iter:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--model", choices=sorted(PRESETS), default="53m",
        help="53m = 노트북 6GB 기준 / 282m = 32GB 서버 기준",
    )
    ap.add_argument(
        "--epochs", type=float, default=1.0, help="데이터를 몇 바퀴 돌지 (max-iters 미지정 시)"
    )
    ap.add_argument("--max-iters", type=int, default=None, help="직접 지정하면 epochs를 무시한다")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument(
        "--precision", choices=("auto", "bf16", "fp16", "fp32"), default=None,
        help="기본 auto — GPU 세대로 결정 (Ampere+ bf16 / V100급 fp16)",
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
