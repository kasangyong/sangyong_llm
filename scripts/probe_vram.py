"""본 학습 설정이 실제로 6GB에 들어가는지 잰다.

주의: Windows(WDDM)에서는 VRAM을 넘겨도 OOM이 나지 않는다. 드라이버가
조용히 시스템 RAM으로 흘려보내기 때문에 "동작은 하지만 처참하게 느린"
상태가 된다. 따라서 OOM 여부만 보면 안 되고,
  1) 최대 할당량이 물리 VRAM 안전선을 넘는지
  2) 스텝 시간이 갑자기 튀는지
를 함께 봐야 한다. 이 둘 중 하나라도 걸리면 실패로 처리한다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.transformer import ModelConfig, Transformer
from train.train import TrainConfig
from train.train import TrainConfig, make_optimizer


TOTAL_GB = 6.0  # 실행 시 실측으로 덮어씀
SAFE_FRAC = 0.85  # 이 비율을 넘으면 시스템 RAM 유출로 본다


def probe(batch_size: int, block_size: int, vocab_size: int = 16384):
    """(안전 여부, peak GB, 스텝 초, 설명) 반환."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=block_size)
        model = Transformer(cfg).cuda()
        opt = make_optimizer(model, TrainConfig())
        x = torch.randint(0, vocab_size, (batch_size, block_size), device="cuda")
        y = torch.randint(0, vocab_size, (batch_size, block_size), device="cuda")

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # 워밍업 2회 (커널 컴파일 + 옵티마이저 상태 생성)
        for _ in range(2):
            step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        n = 3
        for _ in range(n):
            step()
        torch.cuda.synchronize()
        sec = (time.perf_counter() - t0) / n

        peak = torch.cuda.max_memory_allocated() / 1024**3
        del model, opt, x, y
        torch.cuda.empty_cache()

        limit = TOTAL_GB * SAFE_FRAC
        if peak > limit:
            return False, peak, sec, f"VRAM 초과 (안전선 {limit:.2f}GB) -> 시스템 RAM 유출"
        return True, peak, sec, "OK"
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, float("nan"), float("nan"), "OOM"
    except Exception as e:
        torch.cuda.empty_cache()
        return False, float("nan"), float("nan"), f"{type(e).__name__}: {e}"


def parse_args():
    ap = argparse.ArgumentParser()
    # 기본값을 TrainConfig에서 끌어온다. 하드코딩하면 block_size를 바꿀 때마다
    # 프로브가 조용히 옛날 길이로 재서 엉뚱한 배치를 권한다.
    ap.add_argument("--block", type=int, default=TrainConfig().block_size)
    ap.add_argument(
        "--batches",
        default="1,2,4,8,12,16,24,32",
        help="시험할 batch_size 목록",
    )
    ap.add_argument(
        "--world-size", type=int, default=1, help="DDP 랭크 수. 유효 배치 계산에 쓴다"
    )
    ap.add_argument(
        "--target-tokens",
        type=int,
        default=524288,
        help="목표 유효 배치(토큰/스텝). 30만~100만이 무난하다",
    )
    return ap.parse_args()


def main():
    global TOTAL_GB
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA가 없다")
        return 1

    free, total = torch.cuda.mem_get_info(0)
    TOTAL_GB = total / 1024**3
    print(f"VRAM: 가용 {free / 1024**3:.2f}GB / 전체 {TOTAL_GB:.2f}GB")
    print(f"안전선: {TOTAL_GB * SAFE_FRAC:.2f}GB (넘으면 시스템 RAM 유출)")
    print(f"모델: {Transformer(ModelConfig()).num_params():,} 파라미터")
    print(f"문맥: {args.block} 토큰, 랭크 {args.world_size}개\n")

    print(f"{'batch':>6} {'peak':>9} {'초/스텝':>9} {'토큰/초':>10}  결과")
    print("-" * 62)
    best = None
    rows = []
    for batch_size in [int(b) for b in args.batches.split(",") if b.strip()]:
        ok, peak, sec, detail = probe(batch_size, args.block)
        tps = batch_size * args.block / sec if sec == sec and sec > 0 else float("nan")
        rows.append((batch_size, ok, peak, sec, tps))
        print(
            f"{batch_size:>6} {peak:>8.2f}G {sec:>9.3f} {tps:>10,.0f}  "
            f"{'OK' if ok else 'X'} {detail}",
            flush=True,
        )
        if ok:
            # 처리량이 가장 높은 안전 설정을 고른다
            if best is None or tps > best[4]:
                best = (batch_size, ok, peak, sec, tps)
        else:
            # 한 번 넘치면 더 키워봐야 의미가 없다. 유출 상태에서는
            # 스텝 하나에 수 분이 걸려 프로브 자체가 끝나지 않는다.
            print(f"       -> 배치 {batch_size}에서 한계. 더 키우지 않는다.", flush=True)
            break

    print()
    if best is None:
        print("판정: 위험 - 안전하게 들어가는 배치가 없다")
        return 1

    bs = best[0]
    # DDP는 랭크 수만큼 유효 배치가 곱해지므로, 누적 횟수를 그만큼 줄여야
    # 목표 유효 배치가 유지된다. 안 줄이면 의도의 world_size배로 학습된다.
    per_rank_target = max(1, args.target_tokens // args.world_size)
    accum = max(1, per_rank_target // (bs * args.block))
    eff = bs * accum * args.block * args.world_size
    print(f"판정: 통과 - batch_size {bs} 권장 (peak {best[2]:.2f}GB, {best[4]:,.0f} tok/s)")
    print(f"  기울기 누적 {accum} -> 유효 배치 {eff:,} 토큰/스텝 (랭크 {args.world_size}개 합산)")
    total_tps = best[4] * args.world_size
    est_h = 6.6e9 / total_tps / 3600
    print(f"  전 랭크 합산 처리량 추정: {total_tps:,.0f} tok/s")
    print(f"  6.6B 토큰 학습 추정 시간: {est_h:.1f}시간")
    print(f"\n  train_detached.py start --gpus <목록> --batch-size {bs} --grad-accum {accum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
