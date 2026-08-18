"""인스트럭션 튜닝(SFT) 루프.

프리트레이닝 루프의 부품을 그대로 가져다 쓴다(lr_at, make_optimizer,
estimate_loss, save_checkpoint, load_checkpoint). SFT에서 달라지는 것은
데이터가 오는 방식과 손실 마스킹뿐이고, 그건 dataset.py가 처리한다.
루프 자체를 복사하면 나중에 train.py를 고칠 때 여기만 안 고쳐지는 사고가
난다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from finetune.dataset import SFTDataset
from tokenizer.bpe import BPETokenizer
from train.train import (
    TrainConfig,
    estimate_loss,
    load_checkpoint,
    lr_at,
    make_optimizer,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parent.parent
BASE_CKPT_DIR = ROOT / "checkpoints"
# 기반 모델과 절대 섞이면 안 된다. 덮어쓰면 51시간짜리 프리트레이닝이 날아간다.
SFT_CKPT_DIR = ROOT / "checkpoints" / "sft"
DATA_DIR = ROOT / "finetune" / "data"
TOKENIZER_PATH = ROOT / "tokenizer" / "tokenizer.json"


@dataclass
class SFTConfig(TrainConfig):
    # 프리트레이닝 lr은 6e-4다. 그 1/10을 기본값으로 쓴다.
    #
    # 근거: SFT는 새 지식을 넣는 단계가 아니라 이미 학습된 코드 분포에
    # "지시 -> 코드" 형식을 얹는 단계다. 데이터가 프리트레이닝의 1/100,000
    # 규모라 프리트레이닝과 같은 lr을 쓰면 몇십 스텝 만에 SFT 데이터에
    # 과적합하면서 기반 모델이 배운 것을 밀어낸다(catastrophic forgetting).
    # 반대로 너무 낮으면 형식을 아예 못 배운다. 1/10은 그 사이에서 관례적으로
    # 쓰이는 지점이고, 실제 값은 val loss를 보고 조정할 것.
    lr: float = 6e-5
    min_lr_frac: float = 0.1
    # 총 스텝이 수백 규모라 200스텝 워밍업은 학습의 대부분을 워밍업으로
    # 써버린다. 짧게 잡되 0은 피한다(체크포인트 직후 큰 스텝은 위험하다).
    warmup_iters: int = 20
    max_iters: int = 600

    # 샘플이 짧아(수백 토큰) 프리트레이닝만큼 누적할 필요가 없다.
    batch_size: int = 8
    grad_accum: int = 4
    block_size: int = 1024

    epochs: int = 3

    eval_interval: int = 50
    eval_iters: int = 20
    log_interval: int = 10
    ckpt_interval: int = 100

    def iters_for(self, n_samples: int) -> int:
        """에포크 수를 스텝 수로 환산한다."""
        per_step = self.batch_size * self.grad_accum
        return max(1, math.ceil(n_samples * self.epochs / per_step))


def sft_loop(
    model,
    optimizer,
    datasets: dict,
    cfg: SFTConfig,
    device: str,
    mcfg=None,
    ckpt_dir: Path | None = None,
    verbose: bool = True,
) -> list[dict]:
    """SFT 스텝을 max_iters만큼 돈다. 스텝별 기록을 돌려준다.

    ckpt_dir가 None이면 아무것도 저장하지 않는다(테스트용 경로).
    """
    train_ds = datasets["train"]
    history: list[dict] = []
    best_val = float("inf")
    log_path = (ckpt_dir / "sftlog.jsonl") if ckpt_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    iters_since_log = 0

    for it in range(cfg.max_iters):
        lr = lr_at(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_ds.batch(cfg.batch_size, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                _, loss, _ = model(x, targets=y)
            (loss / cfg.grad_accum).backward()
            total_loss += loss.item() / cfg.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        iters_since_log += 1
        history.append({"iter": it, "loss": total_loss, "lr": lr})

        if verbose and it % cfg.log_interval == 0:
            dt = time.time() - t0
            sps = cfg.batch_size * cfg.grad_accum * iters_since_log / max(dt, 1e-9)
            print(
                f"iter {it:5d} | loss {total_loss:.4f} | lr {lr:.2e} "
                f"| gnorm {gnorm:.2f} | {sps:,.1f} 샘플/s",
                flush=True,
            )
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iter": it, "loss": total_loss, "lr": lr}) + "\n")
            t0 = time.time()
            iters_since_log = 0

        is_last = it == cfg.max_iters - 1
        if "val" in datasets and (is_last or (it > 0 and it % cfg.eval_interval == 0)):
            # 평가는 train_ds에서도 배치를 뽑는다. 그대로 두면 그 샘플들이
            # 이번 에포크의 학습 순회에서 빠진다(기본 설정에서 학습 draw의
            # 약 10%). 커서를 되돌려 놓는다.
            saved = train_ds.epoch_state()
            losses = estimate_loss(model, datasets, cfg, device)
            train_ds.restore_epoch_state(saved)
            if verbose:
                print(
                    f"  [eval] iter {it} train {losses['train']:.4f} "
                    f"val {losses['val']:.4f}",
                    flush=True,
                )
            history[-1]["eval"] = losses
            if losses["val"] < best_val:
                best_val = losses["val"]
                if ckpt_dir and mcfg is not None:
                    save_checkpoint(
                        ckpt_dir / "best.pt", model, optimizer, mcfg, cfg, it, best_val
                    )

        if ckpt_dir and mcfg is not None and it > 0 and it % cfg.ckpt_interval == 0:
            save_checkpoint(
                ckpt_dir / "latest.pt", model, optimizer, mcfg, cfg, it, best_val
            )

    if ckpt_dir and mcfg is not None:
        save_checkpoint(
            ckpt_dir / "latest.pt", model, optimizer, mcfg, cfg, cfg.max_iters - 1, best_val
        )
    return history


def run(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SFTConfig()
    if args.lr:
        cfg.lr = args.lr
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    torch.manual_seed(cfg.seed)

    base_ckpt = Path(args.base)
    if not base_ckpt.exists():
        raise SystemExit(
            f"기반 체크포인트가 없다: {base_ckpt}. 프리트레이닝이 끝난 뒤 돌릴 것."
        )
    tok = BPETokenizer.load(TOKENIZER_PATH)

    model, ck = load_checkpoint(base_ckpt, device)
    mcfg = model.cfg
    # 어휘가 어긋난 채로 돌면 임베딩 인덱스가 밀려 조용히 쓰레기를 학습한다.
    if mcfg.vocab_size != tok.vocab_size:
        raise SystemExit(
            f"어휘 불일치: 체크포인트 {mcfg.vocab_size} vs 토크나이저 {tok.vocab_size}. "
            "SFT는 어휘를 절대 바꾸지 않는다."
        )
    if cfg.block_size > mcfg.max_seq_len:
        cfg.block_size = mcfg.max_seq_len

    # 옵티마이저 상태는 이어받지 않는다. 프리트레이닝에서 쌓인 Adam 모멘트는
    # 10배 큰 lr과 다른 데이터 분포에서 만들어진 것이라, 그대로 쓰면 SFT
    # 첫 스텝부터 의도보다 훨씬 큰 갱신이 들어간다.
    optimizer = make_optimizer(model, cfg)

    train_ds = SFTDataset.from_jsonl(Path(args.train), tok, cfg.block_size)
    if len(train_ds) == 0:
        raise SystemExit(f"학습 샘플이 하나도 안 남았다: {train_ds.stats.report()}")
    datasets = {"train": train_ds}
    val_path = Path(args.val)
    if val_path.exists():
        val_ds = SFTDataset.from_jsonl(val_path, tok, cfg.block_size)
        if len(val_ds) == 0:
            # 여기서 안 막으면 첫 평가(기본 50스텝째)에 가서야 죽는다.
            # ckpt_interval이 100이라 그때까지 저장된 것도 없다.
            print(f"[sft] 경고: 검증 샘플이 없다 ({val_ds.stats.report()}). 평가를 건너뛴다.")
        else:
            datasets["val"] = val_ds

    cfg.max_iters = args.max_iters or cfg.iters_for(len(train_ds))
    out_dir = Path(args.out_dir)
    # 위 SFT_CKPT_DIR 주석의 의도를 코드로 강제한다. 기반 체크포인트 폴더에
    # 쓰면 sft_loop가 latest.pt를 SFT 가중치로 덮어쓰고, train.py --resume이
    # 그 파일을 읽으므로 프리트레이닝이 그대로 날아간다.
    if out_dir.resolve() == BASE_CKPT_DIR.resolve():
        raise SystemExit(
            f"--out-dir이 기반 체크포인트 경로({BASE_CKPT_DIR})다. "
            "latest.pt를 덮어써 프리트레이닝을 날린다. 하위 폴더를 쓸 것."
        )

    print("=" * 60)
    print(f"장치        : {device}")
    print(f"기반 체크포인트: {base_ckpt.name} (iter {ck.get('iter', -1)})")
    print(f"어휘        : {tok.vocab_size:,} (변경 없음)")
    print(f"학습 샘플   : {train_ds.stats.report()}")
    if "val" in datasets:
        print(f"검증 샘플   : {datasets['val'].stats.report()}")
    print(f"lr          : {cfg.lr:.1e} (프리트레이닝 {TrainConfig.lr:.1e}의 1/10)")
    print(f"스텝        : {cfg.max_iters:,} ({cfg.epochs}에포크)")
    print(f"저장 경로   : {out_dir}")
    print("=" * 60)

    sft_loop(model, optimizer, datasets, cfg, device, mcfg=mcfg, ckpt_dir=out_dir)
    print(f"\nSFT 종료. 체크포인트: {out_dir / 'latest.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE_CKPT_DIR / "best.pt"))
    ap.add_argument("--train", default=str(DATA_DIR / "sft_train.jsonl"))
    ap.add_argument("--val", default=str(DATA_DIR / "sft_val.jsonl"))
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--device", default=None, help="cpu / cuda (기본: 자동)")
    ap.add_argument(
        "--out-dir", default=str(SFT_CKPT_DIR), help="SFT 체크포인트 저장 경로"
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
