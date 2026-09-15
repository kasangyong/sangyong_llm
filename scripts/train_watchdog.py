"""학습이 죽거나 멈추면 다시 띄운다.

9/5 밤에 학습이 iter 5670에서 사라졌다. stdout 로그에 트레이스백도 OOM도
없고 서버는 재부팅된 적이 없다. 원인을 특정하지 못했으니 "또 같은 일이
일어난다"를 전제로, 원인과 무관하게 복구되도록 감시자를 따로 둔다.

  python scripts/train_watchdog.py start --model 282m --batch-size 2 --grad-accum 128
  python scripts/train_watchdog.py status
  python scripts/train_watchdog.py stop      # 워치독만 멈춘다(학습은 계속)

감시하는 두 가지:
  1. 프로세스 소멸 — PID가 사라지면 --resume으로 다시 띄운다
  2. 정지 - 살아 있는데 로그가 안 늘면(기본 45분) 죽이고 다시 띄운다.
     GPU 쪽이 걸려서 프로세스만 남아 있는 경우는 PID 확인으로 못 잡는다

재시작이 항상 옳은 것은 아니라서 세 가지 예외를 둔다:

  - 정상 종료(로그에 "학습 종료")면 다시 띄우지 않는다
  - 짧게 살다 죽기를 반복하면(기본 15분 안에 3회) 손을 뗀다. 설정이 잘못돼
    시작하자마자 죽는 상황에서 무한히 재시작하면 로그만 뒤덮고 체크포인트를
    덮어쓸 위험이 있다
  - stop 파일이 있으면 즉시 빠진다. 사람이 일부러 멈춘 학습을 되살리면 안 된다
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_detached import (  # noqa: E402
    CKPT_DIR,
    IS_WINDOWS,
    LOG_FILE,
    PID_FILE,
    PYTHON,
    ROOT,
    _alive,
    _read_pid,
    build_train_args,
    launch,
)

WD_PID_FILE = CKPT_DIR / "watchdog.pid"
WD_LOG_FILE = CKPT_DIR / "watchdog.log"
WD_STOP_FILE = CKPT_DIR / "watchdog.stop"
WD_STATE_FILE = CKPT_DIR / "watchdog.json"
WD_RESULT_FILE = CKPT_DIR / "RESULT.txt"

# 학습이 끝까지 갔을 때 train.py가 마지막에 찍는 문장. 이게 보이면 프로세스가
# 없는 게 정상이므로 재시작하지 않는다.
DONE_MARKER = "학습 종료"


@dataclass
class WatchdogConfig:
    poll: float = 300.0
    # 로그는 10 iter마다 찍히고 한 iter가 약 80초라 정상일 때도 13분은 조용하다.
    # 여기에 3.4GB 체크포인트 저장과 eval이 겹칠 수 있어 45분으로 잡는다.
    # 짧게 잡으면 멀쩡한 학습을 죽이는데, 그쪽 손해가 훨씬 크다.
    stall_timeout: float = 2700.0
    # 이보다 짧게 살다 죽으면 "시작 자체가 안 되는" 실패로 센다.
    min_healthy: float = 900.0
    max_fast_fails: int = 3


# SIGTERM을 즉사로 받으면 재시작 도중(Popen과 PID 파일 쓰기 사이)에 끊길 수
# 있다. 플래그만 세우고 루프 경계에서 빠진다. time.sleep은 신호를 받으면 바로
# 깨어나므로 주기를 기다리지 않고 반응한다.
_stop_requested = False


def _on_term(signum, frame) -> None:
    global _stop_requested
    _stop_requested = True


def _should_stop() -> bool:
    return _stop_requested or WD_STOP_FILE.exists()


def _stdout_is_log() -> bool:
    """분리 실행에서는 stdout이 이미 watchdog.log로 향한다. 그때 파일에 또
    쓰면 모든 줄이 두 번 찍힌다."""
    try:
        a = os.fstat(sys.stdout.fileno())
        b = WD_LOG_FILE.stat()
    except (OSError, ValueError):
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _stdout_is_log():
        return
    with open(WD_LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(line + "\n")


def is_finished(log_file: Path | None = None, marker: str = DONE_MARKER) -> bool:
    """정상 종료 문구가 로그 끝부분에 있는지 본다.

    끝의 4KB만 읽는다. 전체를 뒤지면 예전 실행이 남긴 문구까지 걸려서, 재개한
    학습이 죽어도 "이미 끝났다"고 잘못 판단한다.
    """
    # 기본값을 인자 자리에 두면 정의 시점에 묶여서 경로를 바꿔가며 시험할 수 없다.
    log_file = log_file or LOG_FILE
    if not log_file.exists():
        return False
    size = log_file.stat().st_size
    with open(log_file, "rb") as f:
        f.seek(max(0, size - 4096))
        tail = f.read().decode("utf-8", errors="replace")
    return marker in tail


def decide(*, alive: bool, finished: bool, log_age: float | None,
           cfg: WatchdogConfig) -> str:
    """지금 무엇을 해야 하는지 하나로 정한다.

    부수효과가 없어서 테스트로 조합을 다 밟아볼 수 있다. 판단과 실행을 섞으면
    "죽었을 때 재시작한다"는 규칙을 프로세스를 실제로 죽여봐야만 검증할 수 있다.
    """
    if finished:
        return "finished"
    if not alive:
        return "restart_dead"
    if log_age is not None and log_age > cfg.stall_timeout:
        return "restart_stall"
    return "ok"


def _log_age(log_file: Path | None = None) -> float | None:
    log_file = log_file or LOG_FILE
    if not log_file.exists():
        return None
    return time.time() - log_file.stat().st_mtime


def _kill(pid: int, grace: float = 120.0) -> None:
    """SIGTERM 먼저 보낸다. 체크포인트를 쓰는 중이면 그게 끝나고 죽는다."""
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if not _alive(pid):
            return
        # 촘촘히 본다. 이미 죽은 뒤 여기서 더 기다리면 그만큼 재시작이 늦다.
        time.sleep(0.2)
    if not IS_WINDOWS:
        # 멈춘 프로세스는 SIGTERM 핸들러까지 못 도는 경우가 있다.
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(2.0)


def write_result(outcome: str, detail: str, path: Path | None = None) -> Path:
    """끝난 사실을 파일로 남긴다.

    train_stdout.log에는 타임스탬프가 없어서 "언제 끝났는지"를 알 수 없다.
    감시하던 세션이 사라져도 결과가 남도록 여기서 시각과 함께 적는다.
    """
    path = path or WD_RESULT_FILE
    tail = []
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [l for l in lines if l.startswith("iter ") or "[eval]" in l][-5:]
    body = [
        f"결과   : {outcome}",
        f"시각   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"내용   : {detail}",
        "",
        "학습 로그 마지막 줄:",
        *[f"  {l}" for l in tail],
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def _reap() -> None:
    """죽은 자식을 거둔다. 안 거두면 좀비로 남아 살아 있는 것처럼 보인다."""
    if IS_WINDOWS:
        return
    try:
        while os.waitpid(-1, os.WNOHANG)[0] != 0:
            pass
    except (ChildProcessError, OSError):
        pass


def _sleep(seconds: float) -> None:
    """중단 요청을 확인하며 잘게 쪼개 잔다.

    한 번에 다 자면 stop 요청이 최대 주기(기본 5분)만큼 늦게 먹힌다. 그 사이에
    사람이 멈춘 학습을 워치독이 되살릴 수 있다. Windows에는 신호로 깨우는
    방법이 없어서 이 방식이 유일하다.
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        if _should_stop():
            return
        time.sleep(min(1.0, deadline - time.time()))


def run(train_args: list[str], cfg: WatchdogConfig) -> int:
    """감시 루프. 이 함수 자체가 분리된 프로세스로 돈다."""
    # 재시작은 반드시 이어서 돌아야 한다. --resume 없이 다시 띄우면 5,500스텝을
    # 버리고 처음부터 학습하면서 체크포인트까지 덮어쓴다.
    if "--resume" not in train_args:
        train_args = ["--resume", *train_args]

    # 신호 등록은 메인 스레드에서만 된다. 분리 실행에서는 항상 메인 스레드지만,
    # 테스트가 이 루프를 스레드로 돌리므로 조건을 붙인다.
    if not IS_WINDOWS and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _on_term)
        signal.signal(signal.SIGINT, _on_term)

    _log(f"워치독 시작 (PID {os.getpid()})")
    _log(f"  재시작 인자: {' '.join(train_args)}")
    _log(f"  주기 {cfg.poll:.0f}초 / 정지 판정 {cfg.stall_timeout / 60:.0f}분 "
         f"/ 조기사망 기준 {cfg.min_healthy / 60:.0f}분 x {cfg.max_fast_fails}회")

    fast_fails = 0
    launched_at = None  # 워치독이 띄운 실행의 시작 시각

    while True:
        if _should_stop():
            _log("중단 요청 확인. 워치독을 끝낸다 (학습은 건드리지 않는다).")
            WD_STOP_FILE.unlink(missing_ok=True)
            return 0

        pid = _read_pid()
        alive = bool(pid) and _alive(pid)
        action = decide(alive=alive, finished=is_finished(), log_age=_log_age(),
                        cfg=cfg)

        if action == "finished":
            _log("학습이 정상 종료됐다. 재시작하지 않고 워치독을 끝낸다.")
            _log(f"결과를 {write_result('정상 종료', '학습이 끝까지 갔다.')}에 남겼다.")
            return 0

        if action == "restart_stall":
            age = _log_age() or 0.0
            _log(f"정지 감지: PID {pid}는 살아 있는데 로그가 {age / 60:.1f}분째 "
                 f"안 늘었다. 죽이고 다시 띄운다.")
            _kill(pid)
            alive = False
            action = "restart_dead"

        if action == "restart_dead":
            if launched_at is not None and time.time() - launched_at < cfg.min_healthy:
                fast_fails += 1
                _log(f"조기 사망 {fast_fails}/{cfg.max_fast_fails} "
                     f"(산 시간 {(time.time() - launched_at) / 60:.1f}분)")
                if fast_fails >= cfg.max_fast_fails:
                    _log("계속 바로 죽는다. 설정이나 데이터 문제로 보고 워치독은 "
                         "손을 뗀다. train_stdout.log를 직접 볼 것.")
                    _log("판정: 위험 - 사람이 봐야 한다")
                    write_result(
                        "실패 - 사람이 봐야 한다",
                        f"{cfg.min_healthy / 60:.0f}분 안에 죽기를 "
                        f"{cfg.max_fast_fails}회 반복해 워치독이 손을 뗐다.",
                    )
                    return 1
            else:
                fast_fails = 0
            if _should_stop():
                _log("중단 요청이 들어와 재시작하지 않는다.")
                WD_STOP_FILE.unlink(missing_ok=True)
                return 0
            _log(f"학습 프로세스가 없다 (마지막 PID {pid}). 재시작한다.")
            new_pid = launch(train_args)
            launched_at = time.time()
            _log(f"재시작 완료: PID {new_pid}")

        _reap()
        _sleep(cfg.poll)


def cmd_start(args) -> None:
    pid = _read_wd_pid()
    if pid and _alive(pid):
        raise SystemExit(f"워치독이 이미 돌고 있다 (PID {pid}). 먼저 stop 할 것.")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    # 이전 stop 요청이 남아 있으면 새 워치독이 켜자마자 빠진다.
    WD_STOP_FILE.unlink(missing_ok=True)

    train_args = build_train_args(args)
    if "--resume" not in train_args:
        train_args = ["--resume", *train_args]
    WD_STATE_FILE.write_text(
        json.dumps({"train_args": train_args, "started": time.time()}, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [str(PYTHON), "-u", str(Path(__file__).resolve()), "run",
           "--poll", str(args.poll), "--stall-timeout", str(args.stall_timeout),
           "--min-healthy", str(args.min_healthy),
           "--max-fast-fails", str(args.max_fast_fails),
           "--", *train_args]

    env = {**os.environ, "PYTHONUTF8": "1"}
    log = open(WD_LOG_FILE, "a", encoding="utf-8", errors="replace")
    if IS_WINDOWS:
        extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                 | subprocess.DETACHED_PROCESS}
    else:
        extra = {"start_new_session": True}
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            close_fds=True, **extra)
    log.close()
    WD_PID_FILE.write_text(str(proc.pid))
    print(f"워치독 시작: PID {proc.pid}")
    print(f"로그: {WD_LOG_FILE}")


def _read_wd_pid() -> int | None:
    if not WD_PID_FILE.exists():
        return None
    try:
        return int(WD_PID_FILE.read_text().strip())
    except ValueError:
        return None


def cmd_status(args) -> None:
    pid = _read_wd_pid()
    if pid is None:
        print("워치독 PID 파일이 없다. 아직 start 하지 않았다.")
    else:
        print(f"워치독 PID {pid}: {'실행 중' if _alive(pid) else '종료됨'}")
    if WD_STATE_FILE.exists():
        st = json.loads(WD_STATE_FILE.read_text(encoding="utf-8"))
        print(f"재시작 인자: {' '.join(st['train_args'])}")
    tpid = _read_pid()
    print(f"학습 PID {tpid}: {'실행 중' if tpid and _alive(tpid) else '종료됨'}")
    age = _log_age()
    if age is not None:
        print(f"학습 로그: {age / 60:.1f}분 전")
    if WD_LOG_FILE.exists():
        print("--- 워치독 로그 마지막 10줄 ---")
        lines = WD_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for l in lines[-10:]:
            print("  " + l)


def cmd_stop(args) -> None:
    pid = _read_wd_pid()
    if pid is None or not _alive(pid):
        print("돌고 있는 워치독이 없다.")
        WD_STOP_FILE.unlink(missing_ok=True)
        return
    # 신호 대신 파일로 부탁한다. 루프 안에서 재시작을 하는 중에 신호로 끊으면
    # 학습을 띄우다 만 상태로 죽을 수 있다.
    WD_STOP_FILE.write_text("stop")
    print(f"워치독(PID {pid})에 중단을 요청했다. 1초 안에 빠진다.")
    print("학습은 계속 돈다. 학습까지 멈추려면 train_detached.py stop.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("start")
    p.add_argument("--model", default=None, help="53m / 282m")
    p.add_argument("--precision", default=None)
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--resume", action="store_true", default=True,
                   help="항상 켜져 있다. 재시작은 이어서 돌아야 한다.")
    for name, default in (("--poll", 300.0), ("--stall-timeout", 2700.0),
                          ("--min-healthy", 900.0)):
        p.add_argument(name, type=float, default=default)
    p.add_argument("--max-fast-fails", type=int, default=3)

    r = sub.add_parser("run", help="감시 루프 본체 (start가 내부적으로 부른다)")
    r.add_argument("--poll", type=float, default=300.0)
    r.add_argument("--stall-timeout", type=float, default=2700.0)
    r.add_argument("--min-healthy", type=float, default=900.0)
    r.add_argument("--max-fast-fails", type=int, default=3)
    r.add_argument("train_args", nargs="*")

    sub.add_parser("status")
    sub.add_parser("stop")

    args = ap.parse_args()
    if args.cmd == "run":
        cfg = WatchdogConfig(poll=args.poll, stall_timeout=args.stall_timeout,
                             min_healthy=args.min_healthy,
                             max_fast_fails=args.max_fast_fails)
        sys.exit(run(list(args.train_args), cfg))
    {"start": cmd_start, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)


if __name__ == "__main__":
    main()
