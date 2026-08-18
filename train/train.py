"""프리트레이닝 루프. 직접 구현.

노트북 GPU에서 도는 것을 전제로 짰다:
  - bf16 autocast (6GB에 53M 모델 + 옵티마이저 상태를 넣으려면 필수)
  - 기울기 누적으로 유효 배치를 키운다
  - 매 N스텝 체크포인트. 중단은 사고가 아니라 기본 전제다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import ModelConfig, Transformer

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
    block_size: int = 2048  # ModelConfig.max_seq_len과 반드시 같아야 한다
    # torchrun이 띄운 프로세스 수. train()에서 실측값으로 덮어쓴다.
    # 유효 배치와 스텝 수 계산에 들어가야 GPU 수를 바꿔도 같은 분량을 학습한다.
    world_size: int = 1

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
    compile_model: bool = False  # Windows에서는 대체로 불안정하다

    @property
    def tokens_per_iter(self) -> int:
        """한 옵티마이저 스텝이 소비하는 전체 토큰 수(모든 랭크 합산)."""
        return self.batch_size * self.grad_accum * self.block_size * self.world_size


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
def estimate_loss(model, datasets, cfg, device, generator=None):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(cfg.eval_iters)
        for i in range(cfg.eval_iters):
            x, y = ds.batch(cfg.batch_size, device, generator)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                _, loss, _ = model(x, targets=y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path: Path, model, optimizer, mcfg, tcfg, it, best_val):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": asdict(mcfg),
            "train_config": asdict(tcfg),
            "iter": it,
            "best_val": best_val,
        },
        tmp,
    )
    tmp.replace(path)  # 저장 중 죽어도 기존 체크포인트가 안 깨지도록


def load_checkpoint(path: Path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = Transformer(mcfg).to(device)
    model.load_state_dict(ck["model"])
    return model, ck


@dataclass
class DDPContext:
    """분산 학습 상태. torchrun 없이 돌리면 enabled=False로 떨어져
    기존 단일 GPU 경로와 완전히 같게 동작한다."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: str

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    def shutdown(self):
        if self.enabled:
            # 랭크 0이 마지막 체크포인트를 다 쓰기 전에 다른 랭크가 프로세스
            # 그룹을 부수면 저장이 깨진다.
            dist.barrier()
            dist.destroy_process_group()


def setup_ddp() -> DDPContext:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DDPContext(
            False, 0, 0, 1, "cuda" if torch.cuda.is_available() else "cpu"
        )
    if not torch.cuda.is_available():
        raise SystemExit("DDP는 CUDA가 필요하다")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DDPContext(True, rank, local_rank, world_size, f"cuda:{local_rank}")


def train(args):
    ddp = setup_ddp()
    device = ddp.device
    tcfg = TrainConfig()
    tcfg.world_size = ddp.world_size
    if args.batch_size:
        tcfg.batch_size = args.batch_size
    if args.grad_accum:
        tcfg.grad_accum = args.grad_accum

    # 랭크마다 시드를 달리 줘야 서로 다른 구간을 뽑는다. 같은 시드면 세 장이
    # 똑같은 배치를 돌고 all-reduce가 같은 기울기를 평균해, 3배 느려지기만 하고
    # 1장으로 돌린 것과 결과가 같아진다.
    torch.manual_seed(tcfg.seed + ddp.rank)
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

    mcfg = ModelConfig(vocab_size=vocab_size, max_seq_len=tcfg.block_size)
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
        # 모든 랭크가 같은 파일을 직접 읽는다. 랭크 0만 읽고 브로드캐스트하는
        # 것보다 단순하고 가중치가 어긋날 여지가 없다.
        raw_model, ck = load_checkpoint(resume_path, device)
        mcfg = ModelConfig(**ck["model_config"])
        optimizer = make_optimizer(raw_model, tcfg)
        optimizer.load_state_dict(ck["optimizer"])
        start_iter = ck["iter"] + 1
        best_val = ck["best_val"]
        if ddp.is_master:
            print(f"[resume] {resume_path.name}에서 iter {start_iter}부터 재개")
    else:
        raw_model = Transformer(mcfg).to(device)
        optimizer = make_optimizer(raw_model, tcfg)

    # DDP가 초기 가중치를 랭크 0 기준으로 브로드캐스트하므로, 시드가 랭크마다
    # 달라도 세 장이 같은 모델로 출발한다.
    model = DDP(raw_model, device_ids=[ddp.local_rank]) if ddp.enabled else raw_model

    n_params = raw_model.num_params()
    if ddp.is_master:
        print("=" * 60)
        print(f"장치        : {device} (랭크 {ddp.world_size}개)")
        print(f"어휘        : {vocab_size:,}")
        print(f"파라미터    : {n_params:,}")
        print(f"학습 토큰   : {len(train_ds):,} / 검증 {len(val_ds):,}")
        print(
            f"유효 배치   : {tcfg.tokens_per_iter:,} 토큰/스텝 "
            f"(랭크당 {tcfg.batch_size}x{tcfg.grad_accum}x{tcfg.block_size})"
        )
        print(f"스텝        : {tcfg.max_iters:,} (총 {tcfg.max_iters * tcfg.tokens_per_iter / 1e9:.2f}B 토큰)")
        print("=" * 60)

    log_path = CKPT_DIR / "trainlog.jsonl"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()
    # 마지막 로그 이후 지난 스텝 수. log_interval로 고정해서 나누면 첫 줄이
    # 10배 부풀려진 처리량을 보고한다 (그때는 1스텝만 지났으므로).
    iters_since_log = 0

    for it in range(start_iter, tcfg.max_iters):
        lr = lr_at(it, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for micro in range(tcfg.grad_accum):
            x, y = train_ds.batch(tcfg.batch_size, device)
            # 마지막 마이크로스텝에서만 기울기를 동기화한다. 막지 않으면 누적
            # 횟수만큼(기본 32회) all-reduce가 돌아 DDP 이득이 통신비로 날아간다.
            last = micro == tcfg.grad_accum - 1
            sync_ctx = nullcontext() if (last or not ddp.enabled) else model.no_sync()
            with sync_ctx:
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
                ):
                    _, loss, _ = model(x, targets=y)
                # 누적 스텝 수로 나눠야 전체 배치 평균과 같아진다
                (loss / tcfg.grad_accum).backward()
            total_loss += loss.item() / tcfg.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tcfg.grad_clip)
        optimizer.step()
        iters_since_log += 1

        if it % tcfg.log_interval == 0:
            dt = time.time() - t0
            tps = tcfg.tokens_per_iter * iters_since_log / max(dt, 1e-9)
            mem = (
                torch.cuda.max_memory_allocated() / 1024**3
                if device.startswith("cuda")
                else 0
            )
            if ddp.is_master:
                print(
                    f"iter {it:6d} | loss {total_loss:.4f} | lr {lr:.2e} "
                    f"| gnorm {gnorm:.2f} | {tps:,.0f} tok/s | vram {mem:.2f}GB",
                    flush=True,
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"iter": it, "loss": total_loss, "lr": lr, "gnorm": float(gnorm)}
                        )
                        + "\n"
                    )
            t0 = time.time()
            iters_since_log = 0

        if it > 0 and it % tcfg.eval_interval == 0:
            # DDP 래퍼가 아니라 원본으로 돈다. 평가에는 기울기 동기화가 필요 없다.
            losses = estimate_loss(raw_model, datasets, tcfg, device)
            if ddp.enabled:
                # 랭크마다 다른 구간을 봤으므로 평균을 내야 전체 추정이 된다.
                # 동시에 이 all-reduce가 best_val을 전 랭크에서 같은 값으로 만든다.
                buf = torch.tensor(
                    [losses["train"], losses["val"]], device=device, dtype=torch.float32
                )
                dist.all_reduce(buf, op=dist.ReduceOp.AVG)
                losses = {"train": buf[0].item(), "val": buf[1].item()}
            ppl = math.exp(min(losses["val"], 20))
            if ddp.is_master:
                print(
                    f"  [eval] iter {it} train {losses['train']:.4f} "
                    f"val {losses['val']:.4f} ppl {ppl:.2f}",
                    flush=True,
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iter": it, "eval": losses, "val_ppl": ppl}) + "\n")
            if losses["val"] < best_val:
                best_val = losses["val"]
                if ddp.is_master:
                    save_checkpoint(
                        CKPT_DIR / "best.pt", raw_model, optimizer, mcfg, tcfg, it, best_val
                    )

        # 전 랭크의 파라미터와 옵티마이저 상태가 동일하므로 마스터만 저장하면 된다.
        if it > 0 and it % tcfg.ckpt_interval == 0 and ddp.is_master:
            save_checkpoint(resume_path, raw_model, optimizer, mcfg, tcfg, it, best_val)

    if ddp.is_master:
        save_checkpoint(
            resume_path, raw_model, optimizer, mcfg, tcfg, tcfg.max_iters - 1, best_val
        )
        print(f"\n학습 종료. best val loss = {best_val:.4f}")
    ddp.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--epochs", type=float, default=1.0, help="데이터를 몇 바퀴 돌지 (max-iters 미지정 시)"
    )
    ap.add_argument("--max-iters", type=int, default=None, help="직접 지정하면 epochs를 무시한다")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
