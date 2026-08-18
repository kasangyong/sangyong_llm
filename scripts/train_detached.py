"""학습을 터미널·세션과 완전히 분리해서 띄운다.

며칠짜리 작업이라 띄운 쪽이 사라져도 살아 있어야 한다. 부모 프로세스에
묶여 있으면 터미널을 닫거나 세션이 끊길 때 같이 죽는다.

  python scripts/train_detached.py start                 # 분리 실행
  python scripts/train_detached.py start --gpus 1,2,3    # 3장 DDP
  python scripts/train_detached.py status                # 진행 상황
  python scripts/train_detached.py stop                  # 중단 (체크포인트는 남는다)

--gpus에 두 장 이상을 주면 torchrun으로 띄운다. 한 장이면 예전처럼 파이썬을
직접 부른다 — 단일 GPU 경로에 분산 계층을 끼우지 않기 위해서다.

노트북이 절전으로 들어가면 학습도 멈췄다가 깨어날 때 이어진다. 데이터가
망가지진 않지만 벽시계 시간이 그만큼 늘어난다. 며칠 돌릴 거면 절전을 꺼두는
편이 낫다(전원 설정에서 직접 바꿀 것 — 이 스크립트는 시스템 설정을 건드리지
않는다).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
PID_FILE = CKPT_DIR / "train.pid"
LOG_FILE = CKPT_DIR / "train_stdout.log"

IS_WINDOWS = os.name == "nt"
# venv 실행 파일 위치가 플랫폼마다 다르다. 서버(리눅스)와 노트북(Windows)에서
# 같은 스크립트를 쓰기 위해 여기서 갈라준다.
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
TORCHRUN = ROOT / ".venv" / ("Scripts/torchrun.exe" if IS_WINDOWS else "bin/torchrun")


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            errors="replace",
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)  # 신호 0은 존재 확인만 한다
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 남의 프로세스지만 살아는 있다
    return True


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def _build_cmd(args) -> tuple[list[str], list[str]]:
    """(실행 명령, 쓸 GPU 목록)을 만든다."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else []
    train_py = str(ROOT / "train" / "train.py")

    if len(gpus) > 1:
        if not TORCHRUN.exists():
            raise SystemExit(f"torchrun이 없다: {TORCHRUN}")
        # --standalone: 단일 노드. 랑데부 서버를 따로 띄우지 않는다.
        cmd = [
            str(TORCHRUN),
            "--standalone",
            f"--nproc_per_node={len(gpus)}",
            train_py,
        ]
    else:
        cmd = [str(PYTHON), "-u", train_py]

    if args.resume:
        cmd.append("--resume")
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.grad_accum:
        cmd += ["--grad-accum", str(args.grad_accum)]
    return cmd, gpus


def cmd_start(args):
    pid = _read_pid()
    if pid and _alive(pid):
        raise SystemExit(f"이미 돌고 있다 (PID {pid}). 먼저 stop 할 것.")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd, gpus = _build_cmd(args)

    env = {**os.environ, "PYTHONUTF8": "1"}
    if gpus:
        # 남이 쓰는 카드를 피해 명시적으로 고른다. 지정하지 않으면 0번부터
        # 잡는데, 0번에는 다른 서비스가 올라가 있을 수 있다.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)

    # append 모드로 열어 재개 시 이전 로그를 지우지 않는다
    log = open(LOG_FILE, "a", encoding="utf-8", errors="replace")
    log.write(f"\n{'=' * 60}\n[detached] 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"[detached] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '(전체)')}\n")
    log.write(f"[detached] {' '.join(cmd)}\n{'=' * 60}\n")
    log.flush()

    if IS_WINDOWS:
        # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS: 부모가 죽어도 안 따라 죽는다.
        extra = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
        }
    else:
        # 새 세션의 리더로 만든다. 부모(SSH/주피터)가 끊겨도 SIGHUP이 안 오고,
        # stop에서 프로세스 그룹째 죽일 수 있어 torchrun 워커가 고아로 안 남는다.
        extra = {"start_new_session": True}

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        **extra,
    )
    PID_FILE.write_text(str(proc.pid))
    n = len(gpus) if gpus else 1
    print(f"분리 실행 시작: PID {proc.pid} ({'torchrun ' if n > 1 else ''}{n}랭크)")
    print(f"로그: {LOG_FILE}")
    print(f"상태 확인: python scripts/train_detached.py status")


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

    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
    else:
        # 프로세스 그룹째 보낸다. torchrun만 죽이면 워커가 GPU를 쥔 채 고아로
        # 남아 다음 학습이 OOM으로 죽는다.
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    for _ in range(40):
        if not _alive(pid):
            break
        time.sleep(0.5)

    if _alive(pid) and not IS_WINDOWS:
        print("SIGTERM에 안 죽는다. SIGKILL로 보낸다.")
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1.0)

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
    p.add_argument(
        "--gpus",
        default=None,
        help="쓸 GPU 번호 (예: 1,2,3). 두 장 이상이면 torchrun으로 DDP 실행",
    )

    sub.add_parser("status")
    sub.add_parser("stop")

    args = ap.parse_args()
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
