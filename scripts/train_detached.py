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
# 리눅스 서버에는 Scripts/python.exe가 없다. 둘 다 보고 있는 쪽을 쓴다.
_WIN = ROOT / ".venv" / "Scripts" / "python.exe"
_NIX = ROOT / ".venv" / "bin" / "python"
PYTHON = _WIN if _WIN.exists() else _NIX


IS_WINDOWS = os.name == "nt"


def _alive(pid: int) -> bool:
    if not IS_WINDOWS:
        # 신호 0은 실제로 보내지 않고 존재 여부만 확인한다.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # 남의 프로세스지만 살아 있다
        # 좀비는 신호 0에 응답한다. 워치독은 학습을 자식으로 띄우므로, 죽은
        # 학습이 거둬지기 전까지 좀비로 남아 "실행 중"으로 보인다. 상태까지
        # 봐야 죽은 걸 죽었다고 판정한다.
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            return stat.rsplit(") ", 1)[1].split(" ", 1)[0] != "Z"
        except (OSError, IndexError):
            return True
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


def build_train_args(args) -> list[str]:
    """네임스페이스를 train.py 인자 목록으로 바꾼다. 워치독이 재시작할 때
    같은 인자를 그대로 다시 쓰려면 이 변환이 한 군데에 있어야 한다."""
    out = []
    if args.resume:
        out.append("--resume")
    if args.model:
        out += ["--model", args.model]
    if args.precision:
        out += ["--precision", args.precision]
    if getattr(args, "epochs", None) is not None:
        out += ["--epochs", str(args.epochs)]
    if args.batch_size:
        out += ["--batch-size", str(args.batch_size)]
    if args.grad_accum:
        out += ["--grad-accum", str(args.grad_accum)]
    return out


def launch(train_args: list[str], python: Path = PYTHON, root: Path = ROOT,
           ckpt_dir: Path = CKPT_DIR, log_file: Path = LOG_FILE,
           pid_file: Path = PID_FILE) -> int:
    """학습을 세션에서 떼어내 띄우고 PID를 돌려준다.

    워치독과 수동 실행이 같은 경로를 쓰도록 여기 하나로 모은다. 런처가 둘이면
    인자나 PID 파일이 어긋나서, 워치독이 자기가 안 띄운 프로세스를 감시하거나
    이미 도는 학습 위에 하나를 더 얹는 사고가 난다.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(python), "-u", str(root / "train" / "train.py"), *train_args]

    env = {**os.environ, "PYTHONUTF8": "1"}
    # append 모드로 열어 재개 시 이전 로그를 지우지 않는다
    log = open(log_file, "a", encoding="utf-8", errors="replace")
    log.write(f"\n{'=' * 60}\n[detached] 시작 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log.write(f"[detached] {' '.join(cmd)}\n{'=' * 60}\n")
    log.flush()

    # 부모가 죽어도 따라 죽지 않게 떼어낸다. Windows는 프로세스 그룹 분리
    # 플래그, POSIX는 setsid(start_new_session)로 세션을 새로 판다. SSH가
    # 끊길 때 날아오는 SIGHUP이 새 세션에는 전달되지 않는다.
    if IS_WINDOWS:
        extra = {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
        }
    else:
        extra = {"start_new_session": True}
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        **extra,
    )
    # 자식이 fd를 복제해 갔으니 이쪽 사본은 닫는다. 워치독처럼 이 함수를
    # 여러 번 부르는 쪽에서는 안 닫으면 재시작마다 fd가 하나씩 샌다.
    log.close()
    pid_file.write_text(str(proc.pid))
    return proc.pid


def cmd_start(args):
    pid = _read_pid()
    if pid and _alive(pid):
        raise SystemExit(f"이미 돌고 있다 (PID {pid}). 먼저 stop 할 것.")

    new_pid = launch(build_train_args(args))
    print(f"분리 실행 시작: PID {new_pid}")
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
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        # SIGTERM으로 먼저 부탁한다. 체크포인트 저장 중이면 그게 끝나고 죽는다.
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
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
    p.add_argument("--model", default=None, help="53m / 282m (train.py 기본값 53m)")
    p.add_argument("--precision", default=None, help="auto / bf16 / fp16 / fp32")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)

    sub.add_parser("status")
    sub.add_parser("stop")

    args = ap.parse_args()
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
