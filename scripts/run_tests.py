"""모든 검증을 한 번에 돌린다. 하나라도 실패하면 0이 아닌 코드로 끝난다."""

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SUITES = [
    ("0단계 환경", ROOT / "scripts" / "verify_env.py"),
    ("1단계 토크나이저", ROOT / "tests" / "test_tokenizer.py"),
    ("2단계 모델", ROOT / "tests" / "test_model.py"),
    ("3단계 학습 루프", ROOT / "tests" / "test_training.py"),
    ("3단계 DDP", ROOT / "tests" / "test_ddp.py"),
    ("평가 하네스", ROOT / "tests" / "test_eval.py"),
    ("SFT 파이프라인", ROOT / "tests" / "test_sft.py"),
    ("검색/툴 레이어", ROOT / "tests" / "test_tools.py"),
    ("적대적 회귀", ROOT / "tests" / "test_regress_correctness.py"),
]


def main():
    results = []
    for name, path in SUITES:
        print(f"\n{'#' * 60}\n# {name}: {path.name}\n{'#' * 60}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(path)],
            env={**__import__("os").environ, "PYTHONUTF8": "1"},
        )
        results.append((name, proc.returncode == 0))

    print(f"\n{'=' * 60}\n전체 요약\n{'=' * 60}")
    for name, ok in results:
        print(f"  {'통과' if ok else '실패'}  {name}")
    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n판정: 위험 - {len(failed)}개 스위트 실패: {failed}")
        return 1
    print(f"\n판정: 통과 - {len(results)}개 스위트 전부 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
