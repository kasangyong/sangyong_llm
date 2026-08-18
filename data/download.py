"""codeparrot-clean(파이썬 전용, 중복 제거됨)의 샤드를 내려받는다.

데이터셋 선정 경위:
- the-stack-v2: gated. 게다가 본문이 아니라 blob_id만 있어서
  Software Heritage에서 따로 받아야 한다.
- stack-edu: gated는 아니지만 마찬가지로 본문이 없다(blob_id만).
- codeparrot-clean: gated 아님. content와 license가 직접 들어 있고
  파이썬 전용에 중복 제거까지 되어 있다. -> 이걸 쓴다.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from huggingface_hub import hf_hub_download

REPO = "codeparrot/codeparrot-clean"
N_SHARDS_TOTAL = 54
RAW_DIR = Path(__file__).resolve().parent / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4, help=f"받을 샤드 수 (최대 {N_SHARDS_TOTAL})")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n = max(1, min(args.shards, N_SHARDS_TOTAL))

    total_mb = 0.0
    for i in range(1, n + 1):
        name = f"file-{i:012d}.json.gz"
        print(f"[download] {name} ({i}/{n})", flush=True)
        path = hf_hub_download(
            repo_id=REPO,
            filename=name,
            repo_type="dataset",
            local_dir=str(RAW_DIR),
        )
        mb = Path(path).stat().st_size / 1024**2
        total_mb += mb
        print(f"[done] {name} ({mb:.1f} MB)", flush=True)

    print(f"\n샤드 {n}개 / {total_mb:.1f} MB -> {RAW_DIR}")


if __name__ == "__main__":
    main()
