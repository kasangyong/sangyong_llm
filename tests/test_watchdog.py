"""워치독을 깨뜨리기 위한 테스트.

워치독이 틀리는 방향은 두 가지고, 위험도가 다르다.

  - 살려야 하는데 안 살린다 → 하루를 날린다 (지난번 그 일)
  - 살리면 안 되는데 살린다 → 사람이 일부러 멈춘 학습을 되살리거나, 시작하자
    마자 죽는 실행을 무한 반복하며 체크포인트를 덮어쓴다

두 번째가 더 나쁘다. 그래서 "재시작하면 안 되는 상황"을 더 촘촘히 노린다.
"""

import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_detached as td  # noqa: E402
import train_watchdog as tw  # noqa: E402

RESULTS = []
CFG = tw.WatchdogConfig(poll=0.1, stall_timeout=1.0, min_healthy=0.0,
                        max_fast_fails=3)


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


_TMP_ROOT = Path(tempfile.mkdtemp(prefix="wdtest_"))
atexit.register(shutil.rmtree, _TMP_ROOT, True)


def _tmpdir(tag):
    """진짜 checkpoints/ 밖에서 논다. 그 안에는 3.4GB 체크포인트와 돌고 있는
    학습의 PID 파일이 있어서, 시험이 실수로 건드리면 학습을 죽인다."""
    d = _TMP_ROOT / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _redirect(d):
    """워치독이 보는 경로를 임시 디렉터리로 돌린다. 진짜 학습 상태 파일을
    건드리면 테스트가 돌고 있는 학습을 죽일 수 있다."""
    td.PID_FILE = d / "train.pid"
    td.LOG_FILE = d / "train_stdout.log"
    tw.LOG_FILE = d / "train_stdout.log"
    tw.WD_LOG_FILE = d / "watchdog.log"
    tw.WD_STOP_FILE = d / "watchdog.stop"
    tw.WD_RESULT_FILE = d / "RESULT.txt"
    tw._stop_requested = False


def _sleeper(seconds=60):
    return subprocess.Popen([sys.executable, "-c", f"import time;time.sleep({seconds})"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# --- decide: 판단만 따로 본다 -------------------------------------------------

def c_healthy_no_restart():
    a = tw.decide(alive=True, finished=False, log_age=CFG.stall_timeout * 0.5, cfg=CFG)
    assert a == "ok", a
    return "살아 있고 로그 신선 -> ok"


def c_dead_restarts():
    a = tw.decide(alive=False, finished=False, log_age=0.0, cfg=CFG)
    assert a == "restart_dead", a
    return "프로세스 소멸 -> restart_dead"


def c_stall_restarts():
    a = tw.decide(alive=True, finished=False, log_age=CFG.stall_timeout + 0.1, cfg=CFG)
    assert a == "restart_stall", a
    return "살아 있으나 로그 정지 -> restart_stall"


def c_finished_beats_dead():
    """정상 종료 뒤에는 프로세스가 없는 게 맞다. 여기서 재시작하면 끝난 학습을
    다시 돌린다."""
    a = tw.decide(alive=False, finished=True, log_age=99999.0, cfg=CFG)
    assert a == "finished", a
    return "정상 종료 + 프로세스 없음 -> finished (재시작 안 함)"


def c_stall_boundary():
    """경계에서 멀쩡한 학습을 죽이면 안 된다. 초과해야 정지로 본다."""
    at = tw.decide(alive=True, finished=False, log_age=CFG.stall_timeout, cfg=CFG)
    assert at == "ok", at
    return f"log_age == stall_timeout({CFG.stall_timeout}) -> ok"


def c_missing_log_not_stall():
    """로그 파일이 아직 없다고 정지로 판정하면, 갓 띄운 학습을 바로 죽인다."""
    a = tw.decide(alive=True, finished=False, log_age=None, cfg=CFG)
    assert a == "ok", a
    return "로그 파일 없음 + 살아 있음 -> ok"


# --- is_finished: 꼬리만 읽어야 한다 ------------------------------------------

def c_finished_marker_at_tail():
    d = _tmpdir("fin")
    log = d / "train_stdout.log"
    log.write_text("iter 10 | loss 1.0\n학습 종료. best val loss = 0.9574\n",
                   encoding="utf-8")
    assert tw.is_finished(log) is True
    return "끝의 종료 문구를 찾는다"


def c_old_finish_marker_ignored():
    """예전 실행이 남긴 '학습 종료'가 로그 앞쪽에 남아 있다. 전체를 뒤지면
    재개한 학습이 죽어도 '이미 끝났다'며 재시작하지 않는다 - 지난번 사고가
    조용히 반복되는 경로다."""
    d = _tmpdir("oldfin")
    log = d / "train_stdout.log"
    log.write_text("학습 종료. best val loss = 1.2\n" + "iter 1 | loss 1.0\n" * 500,
                   encoding="utf-8")
    assert log.stat().st_size > 4096, "꼬리 4KB 밖으로 밀어내야 의미 있는 시험"
    assert tw.is_finished(log) is False
    return f"{log.stat().st_size:,}바이트 중 앞쪽 종료 문구는 무시"


def c_no_log_not_finished():
    d = _tmpdir("nolog")
    assert tw.is_finished(d / "없는파일.log") is False
    return "로그 없음 -> finished 아님"


# --- 루프: 실제 프로세스로 확인 ------------------------------------------------

def _run_loop(cfg, stop_after=None):
    """감시 루프를 스레드로 돌리고 결과를 받는다."""
    box = {}

    def target():
        box["rc"] = tw.run(["--model", "282m"], cfg)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    if stop_after is not None:
        time.sleep(stop_after)
        tw.WD_STOP_FILE.write_text("stop")
    t.join(timeout=30)
    if t.is_alive():
        # 샌 스레드가 다음 시험의 가짜 launch를 부르면 재시작 횟수가 오염된다.
        tw._stop_requested = True
        tw.WD_STOP_FILE.write_text("stop")
        t.join(timeout=10)
    assert not t.is_alive(), "루프가 안 끝났다"
    return box.get("rc")


def c_restarts_dead_process():
    d = _tmpdir("dead")
    _redirect(d)
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    td.PID_FILE.write_text("999999")  # 존재하지 않는 PID
    calls = []
    procs = []

    def fake_launch(args, **kw):
        calls.append(list(args))
        p = _sleeper(60)
        procs.append(p)
        td.PID_FILE.write_text(str(p.pid))
        (d / "train_stdout.log").write_text("iter 2\n", encoding="utf-8")
        return p.pid

    orig, tw.launch = tw.launch, fake_launch
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.1, stall_timeout=999, min_healthy=0.0),
                       stop_after=0.5)
    finally:
        tw.launch = orig
        for p in procs:
            p.kill()
    assert rc == 0, rc
    assert len(calls) == 1, f"재시작 {len(calls)}회 (1회여야 한다)"
    assert "--resume" in calls[0], calls[0]
    return f"죽은 PID 감지 -> 1회 재시작, 인자 {calls[0]}"


def c_healthy_process_untouched():
    """살아 있고 로그도 늘고 있으면 손대지 않아야 한다."""
    d = _tmpdir("alive")
    _redirect(d)
    p = _sleeper(60)
    td.PID_FILE.write_text(str(p.pid))
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    calls = []
    orig, tw.launch = tw.launch, lambda a, **k: calls.append(a) or 1
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.1, stall_timeout=999, min_healthy=0.0),
                       stop_after=0.5)
        alive_after = p.poll() is None
    finally:
        tw.launch = orig
        p.kill()
    assert rc == 0, rc
    assert calls == [], f"멀쩡한데 {len(calls)}회 재시작했다"
    assert alive_after, "멀쩡한 프로세스를 죽였다"
    return "5주기 동안 재시작 0회, 프로세스 생존"


def c_kills_stalled_process():
    """PID는 살아 있는데 로그가 멈춘 경우. PID 확인만 하면 영원히 못 잡는다."""
    d = _tmpdir("stall")
    _redirect(d)
    stuck = _sleeper(60)
    td.PID_FILE.write_text(str(stuck.pid))
    log = d / "train_stdout.log"
    log.write_text("iter 1\n", encoding="utf-8")
    old = time.time() - 3600
    os.utime(log, (old, old))  # 1시간째 안 늘어난 로그

    calls = []
    procs = []

    def fake_launch(args, **kw):
        calls.append(list(args))
        p = _sleeper(60)
        procs.append(p)
        td.PID_FILE.write_text(str(p.pid))
        log.write_text("iter 2\n", encoding="utf-8")
        # 재시작 직후 멈춘다. 시간으로 끊으면 새로 띄운 프로세스가 다시 정지
        # 판정에 걸려서(여기 stall_timeout은 1초다) 횟수가 흔들린다.
        tw.WD_STOP_FILE.write_text("stop")
        return p.pid

    orig, tw.launch = tw.launch, fake_launch
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.1, stall_timeout=1.0, min_healthy=0.0))
        stuck_dead = stuck.poll() is not None or not td._alive(stuck.pid)
    finally:
        tw.launch = orig
        stuck.kill()
        for p in procs:
            p.kill()
    assert rc == 0, rc
    assert stuck_dead, "멈춘 프로세스를 안 죽였다"
    assert len(calls) == 1, f"재시작 {len(calls)}회"
    return "정지 프로세스 종료 후 1회 재시작"


def c_crash_loop_gives_up():
    """시작하자마자 죽는 상황에서 무한 재시작하면 로그를 뒤덮고 체크포인트를
    덮어쓴다. 3회에서 손을 떼고 위험 판정을 내야 한다."""
    d = _tmpdir("crash")
    _redirect(d)
    td.PID_FILE.write_text("999999")
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    calls = []

    def fake_launch(args, **kw):
        calls.append(list(args))
        td.PID_FILE.write_text("999998")  # 띄우자마자 사라진 셈
        return 999998

    orig, tw.launch = tw.launch, fake_launch
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.05, stall_timeout=999,
                                         min_healthy=60.0, max_fast_fails=3))
    finally:
        tw.launch = orig
    assert rc == 1, f"종료코드 {rc} (위험은 1이어야 한다)"
    assert len(calls) == 3, f"재시작 {len(calls)}회 (3회에서 멈춰야 한다)"
    return f"조기 사망 3회 후 포기, 종료코드 {rc}"


def c_stop_file_blocks_restart():
    """사람이 stop을 건 직후 학습이 죽어 있는 상태. 되살리면 안 된다."""
    d = _tmpdir("stopfile")
    _redirect(d)
    td.PID_FILE.write_text("999999")
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    tw.WD_STOP_FILE.write_text("stop")
    calls = []
    orig, tw.launch = tw.launch, lambda a, **k: calls.append(a) or 1
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.05, stall_timeout=999, min_healthy=0.0))
    finally:
        tw.launch = orig
    assert rc == 0, rc
    assert calls == [], "stop 상태인데 재시작했다"
    assert not tw.WD_STOP_FILE.exists(), "stop 파일을 안 치웠다"
    return "stop 파일 -> 재시작 0회, 파일 정리됨"


def c_sigterm_stops_loop():
    """train_detached.py stop이 SIGTERM을 보낸다. 다음 주기를 기다리지 않고
    빠져야 학습을 죽인 직후 되살리는 창이 닫힌다."""
    d = _tmpdir("sigterm")
    _redirect(d)
    p = _sleeper(60)
    td.PID_FILE.write_text(str(p.pid))
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    box = {}

    def target():
        box["rc"] = tw.run(["--model", "282m"],
                           tw.WatchdogConfig(poll=30.0, stall_timeout=999,
                                             min_healthy=0.0))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    time.sleep(0.5)
    started = time.time()
    tw._on_term(signal.SIGTERM, None)  # 신호 핸들러가 하는 일과 동일
    t.join(timeout=10)
    p.kill()
    # 주기가 30초라 플래그를 안 보면 여기서 타임아웃 난다
    assert not t.is_alive(), "SIGTERM 뒤에도 루프가 안 끝났다"
    assert box.get("rc") == 0, box
    return f"주기 30초인데 {time.time() - started:.1f}초 만에 종료"


# --- 재시작 인자 -------------------------------------------------------------

def c_result_written_on_finish():
    """세션이 끊겨도 '언제 끝났는지'가 서버에 남아야 한다. train_stdout.log에는
    타임스탬프가 없어서 로그만으로는 종료 시각을 알 수 없다."""
    d = _tmpdir("result_ok")
    _redirect(d)
    (d / "train_stdout.log").write_text(
        "iter 13120 | loss 0.71 | lr 6.00e-05 | gnorm 0.10 | 6,500 tok/s\n"
        "학습 종료. best val loss = 0.7421\n", encoding="utf-8")
    td.PID_FILE.write_text("999999")
    calls = []
    orig, tw.launch = tw.launch, lambda a, **k: calls.append(a) or 1
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.05, stall_timeout=999, min_healthy=0.0))
    finally:
        tw.launch = orig
    assert rc == 0, rc
    assert calls == [], "끝난 학습을 재시작했다"
    text = tw.WD_RESULT_FILE.read_text(encoding="utf-8")
    assert "정상 종료" in text, text
    assert "iter 13120" in text, "마지막 진행 상황이 안 담겼다"
    assert time.strftime("%Y-%m-%d") in text, "종료 시각이 없다"
    return "RESULT.txt에 종료 시각 + 마지막 iter 기록"


def c_result_written_on_giveup():
    """워치독이 손을 떼는 건 사람이 반드시 알아야 하는 상태다. 조용히 끝나면
    학습이 멈춘 줄 모르고 며칠이 지나간다 - 지난번에 겪은 그 상황이다."""
    d = _tmpdir("result_fail")
    _redirect(d)
    (d / "train_stdout.log").write_text("iter 5670 | loss 1.01\n", encoding="utf-8")
    td.PID_FILE.write_text("999999")

    def fake_launch(args, **kw):
        td.PID_FILE.write_text("999998")
        return 999998

    orig, tw.launch = tw.launch, fake_launch
    try:
        rc = _run_loop(tw.WatchdogConfig(poll=0.05, stall_timeout=999,
                                         min_healthy=60.0, max_fast_fails=2))
    finally:
        tw.launch = orig
    assert rc == 1, rc
    text = tw.WD_RESULT_FILE.read_text(encoding="utf-8")
    assert "실패" in text, text
    assert "iter 5670" in text, "마지막 진행 상황이 안 담겼다"
    return "포기할 때도 RESULT.txt를 남긴다"


def c_resume_always_injected():
    """--resume 없이 재시작하면 5,500스텝을 버리고 처음부터 학습하면서
    체크포인트를 덮어쓴다. 워치독이 반드시 끼워 넣어야 한다."""
    d = _tmpdir("resume")
    _redirect(d)
    td.PID_FILE.write_text("999999")
    (d / "train_stdout.log").write_text("iter 1\n", encoding="utf-8")
    calls = []

    def fake_launch(args, **kw):
        calls.append(list(args))
        td.PID_FILE.write_text("999998")
        return 999998

    orig, tw.launch = tw.launch, fake_launch
    try:
        tw.run(["--model", "282m", "--batch-size", "2"],
               tw.WatchdogConfig(poll=0.05, stall_timeout=999, min_healthy=60.0,
                                 max_fast_fails=1))
    finally:
        tw.launch = orig
    assert calls and calls[0][0] == "--resume", calls
    assert calls[0].count("--resume") == 1, f"중복 주입: {calls[0]}"
    return f"{calls[0]}"


def c_build_train_args_roundtrip():
    """워치독은 8/31에 띄운 것과 같은 명령으로 재시작해야 한다. 배치/누적이
    빠지면 유효 배치가 달라져 학습 궤적이 바뀐다."""
    import argparse
    ns = argparse.Namespace(resume=True, model="282m", precision=None, epochs=None,
                            batch_size=2, grad_accum=128)
    got = td.build_train_args(ns)
    want = ["--resume", "--model", "282m", "--batch-size", "2", "--grad-accum", "128"]
    assert got == want, f"{got} != {want}"
    return " ".join(got)


def main():
    print("=" * 60)
    check("멀쩡하면 건드리지 않는다", c_healthy_no_restart)
    check("프로세스 소멸 감지", c_dead_restarts)
    check("정지 감지", c_stall_restarts)
    check("정상 종료는 재시작 안 함", c_finished_beats_dead)
    check("정지 판정 경계", c_stall_boundary)
    check("로그 없음을 정지로 오인 안 함", c_missing_log_not_stall)
    check("종료 문구 인식", c_finished_marker_at_tail)
    check("예전 종료 문구 무시(꼬리만 읽기)", c_old_finish_marker_ignored)
    check("로그 없으면 미완료", c_no_log_not_finished)
    check("죽은 프로세스 재시작", c_restarts_dead_process)
    check("살아 있는 프로세스 방치", c_healthy_process_untouched)
    check("멈춘 프로세스 강제 종료 후 재시작", c_kills_stalled_process)
    check("연속 조기 사망 시 포기", c_crash_loop_gives_up)
    check("stop 파일이 재시작을 막는다", c_stop_file_blocks_restart)
    check("SIGTERM에 즉시 반응", c_sigterm_stops_loop)
    check("정상 종료 시 결과 파일", c_result_written_on_finish)
    check("포기 시 결과 파일", c_result_written_on_giveup)
    check("--resume 항상 주입", c_resume_always_injected)
    check("재시작 인자 일치", c_build_train_args_roundtrip)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 워치독을 붙이면 안 된다")
        return 1
    print("\n판정: 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
