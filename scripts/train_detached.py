"""학습을 터미널·세션과 완전히 분리해서 띄운다.

며칠짜리 작업이라 띄운 쪽이 사라져도 살아 있어야 한다. 부모 프로세스에
묶여 있으면 터미널을 닫거나 세션이 끊길 때 같이 죽는다.

  python scripts/train_detached.py start      # 분리 실행
  python scripts/train_detached.py status     # 진행 상황
  python scripts/train_detached.py stop       # 중단 (체크포인트는 남는다)

노트북이 절전으로 들어가면 학습도 멈췄다가 깨어날 때 이어진다. 데이터가
망가지진 않지만 벽시계 시간이 그만큼 늘어난다. 며칠 돌릴 거면 절전을 꺼두는
편이 낫다(전원 설정에서 직접 바꿀 것 — 이 스크립트는 시스템 설정을 건드리지
않는다).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
PID_FILE = CKPT_DIR / "train.pid"
LOG_FILE = CKPT_DIR / "train_stdout.log"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def _alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout
    return str(pid) in out


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def cmd_start(args):
    pid = _read_pid()
    if pid and _alive(pid):
        raise SystemExit(f"이미 돌고 있다 (PID {pid}). 먼저 stop 할 것.")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON), "-u", str(ROOT / "train" / "train.py")]
    if args.resume:
        cmd.append("--resume")
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.grad_accum:
        cmd += ["--grad-accum", str(args.grad_accum)]

    env = {**os.environ, "PYTHONUTF8": "1"}
    # append 모드로 열어 재개 시 이전 로그를 지우지 않는다
    log = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
    log.write(f"\n{'=' * 60}\n[detached] 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"[detached] {' '.join(cmd)}\n{'=' * 60}\n")
    log.flush()

    # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS: 부모가 죽어도 안 따라 죽는다.
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=creationflags,
        close_fds=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"분리 실행 시작: PID {proc.pid}")
    print(f"로그: {LOG_FILE}")
    print(f"상태 확인: {PYTHON.name} scripts/train_detached.py status")


def cmd_status(args):
    pid = _read_pid()
    if pid is None:
        print("PID 파일이 없다. 아직 start 하지 않았다.")
        return
    alive = _alive(pid)
    print(f"PID {pid}: {'실행 중' if alive else '종료됨'}")

    if LOG_FILE.exists():
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        iters = [l for l in lines if l.startswith("iter ")]
        evals = [l for l in lines if l.strip().startswith("[eval]")]
        print(f"로그 줄 수: {len(lines):,} (iter {len(iters):,}개, eval {len(evals)}개)")
        for l in iters[-3:]:
            print("  " + l)
        for l in evals[-2:]:
            print("  " + l.strip())
        mtime = time.strftime("%m-%d %H:%M:%S", time.localtime(LOG_FILE.stat().st_mtime))
        age = time.time() - LOG_FILE.stat().st_mtime
        print(f"마지막 로그: {mtime} ({age / 60:.1f}분 전)")
        if alive and age > 1800:
            print("  경고: 30분 넘게 로그가 안 늘었다. 절전이거나 멈춘 것이다.")

    for name in ("latest.pt", "best.pt"):
        p = CKPT_DIR / name
        if p.exists():
            mt = time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))
            print(f"{name}: {p.stat().st_size / 1024**2:.1f} MB ({mt})")


def cmd_stop(args):
    pid = _read_pid()
    if pid is None or not _alive(pid):
        print("돌고 있는 학습이 없다.")
        return
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    for _ in range(20):
        if not _alive(pid):
            break
        time.sleep(0.5)
    print(f"PID {pid} {'종료됨' if not _alive(pid) else '종료 실패'}")
    print("체크포인트는 남아 있다. --resume 으로 이어서 돌릴 수 있다.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)

    sub.add_parser("status")
    sub.add_parser("stop")

    args = ap.parse_args()
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
