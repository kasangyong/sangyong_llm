"""체크포인트에서 추론에 필요한 것만 남긴다.

학습 체크포인트의 3분의 2는 AdamW 옵티마이저 상태(exp_avg, exp_avg_sq)다.
학습을 이어서 하려면 필요하지만 추론에는 쓸모가 없다. 떼어내면 파일이
3분의 1로 줄어 서버에서 로컬로 옮기기가 훨씬 수월하다.

  53M  : 612MB -> 204MB
  282M : 3.2GB -> 1.1GB

--half을 주면 bf16으로 저장해 한 번 더 반으로 줄인다. 추론 품질 차이는
사실상 없다(어차피 학습도 bf16 autocast로 했다). 다만 이 파일로는 학습을
이어서 할 수 없다.

  python scripts/export_weights.py --ckpt checkpoints/best.pt
  python scripts/export_weights.py --ckpt checkpoints/best.pt --half
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--out", default=None, help="기본: <입력>_weights.pt")
    ap.add_argument("--half", action="store_true", help="bf16으로 저장해 절반으로 줄인다")
    args = ap.parse_args()

    src = Path(args.ckpt)
    if not src.exists():
        raise SystemExit(f"체크포인트가 없다: {src}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("model", "model_config"):
        if key not in ck:
            raise SystemExit(f"체크포인트에 '{key}'가 없다. 형식이 다르다.")

    state = ck["model"]
    if args.half:
        # 정규화 가중치까지 내리면 수치가 불안정해질 수 있어 2차원 이상만 내린다.
        state = {
            k: (v.to(torch.bfloat16) if v.dim() >= 2 else v) for k, v in state.items()
        }

    payload = {
        "model": state,
        "model_config": ck["model_config"],
        "iter": ck.get("iter", -1),
        "best_val": ck.get("best_val"),
        # 추론 전용임을 파일 자체에 남긴다. 이걸로 --resume 하면 옵티마이저
        # 상태가 없어 궤적이 끊긴다.
        "inference_only": True,
        "dtype": "bfloat16" if args.half else "float32",
    }

    out = Path(args.out) if args.out else src.with_name(src.stem + "_weights.pt")
    tmp = out.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(out)

    a, b = src.stat().st_size / 1024**2, out.stat().st_size / 1024**2
    # 입출력 임베딩이 공유돼 있으면 state_dict에 같은 저장소가 두 키로 들어간다.
    # 그냥 더하면 임베딩을 두 번 세서 파라미터 수가 부풀려진다.
    # id()로는 못 잡는다 — torch.load가 공유 텐서를 별개 Tensor 객체로 복원하기
    # 때문이다. 저장소 포인터로 봐야 한다.
    seen = set()
    n = 0
    for v in ck["model"].values():
        key = v.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        n += v.numel()
    print(f"입력 : {src.name}  {a:,.1f} MB")
    print(f"출력 : {out.name}  {b:,.1f} MB  ({b / a * 100:.0f}%)")
    print(f"파라미터: {n:,}  iter={payload['iter']}  dtype={payload['dtype']}")
    print()
    print("이 파일은 추론 전용이다. 학습을 이어서 하려면 원본 체크포인트를 쓸 것.")
    print(f"사용: python train/sample.py --ckpt {out}")


if __name__ == "__main__":
    main()
