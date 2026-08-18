"""채점기를 채점한다.

모델을 평가하기 전에 평가 도구부터 믿을 수 있어야 한다. 정답을 통과시키고
오답을 떨어뜨리는지, 그리고 채점기를 속이는 코드에 넘어가지 않는지 확인한다.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import is_valid_python, run_candidate, truncate_completion
from eval.problems import PROBLEMS

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def c_solutions_pass():
    """모든 정답이 통과해야 한다. 하나라도 떨어지면 문제나 테스트가 틀린 것이다."""
    bad = []
    for p in PROBLEMS:
        r = run_candidate(p.prompt + p.solution, p.test)
        if not r.ok:
            bad.append((p.name, r.reason, r.detail))
    assert not bad, f"정답이 떨어졌다: {bad}"
    return f"정답 {len(PROBLEMS)}개 전부 통과"


def c_wrong_answers_fail():
    """일부러 틀린 답은 전부 떨어져야 한다. 통과하면 테스트가 허술한 것이다."""
    bad = []
    for p in PROBLEMS:
        r = run_candidate(p.prompt + p.wrong, p.test)
        if r.ok:
            bad.append(p.name)
    assert not bad, f"틀린 답이 통과했다 (테스트가 허술함): {bad}"
    return f"오답 {len(PROBLEMS)}개 전부 탈락"


def c_syntax_error_caught():
    r = run_candidate("def f(:\n  pass", "assert True")
    assert not r.ok and r.reason == "syntax", f"문법 오류를 못 잡았다: {r}"
    return "문법 오류 -> reason='syntax'"


def c_timeout_caught():
    """무한루프가 채점기를 멈추게 하면 안 된다."""
    r = run_candidate("def f():\n    while True:\n        pass", "f()", timeout=3)
    assert not r.ok and r.reason == "timeout", f"타임아웃을 못 잡았다: {r}"
    return "무한루프 -> reason='timeout' (3초)"


def c_early_exit_not_counted():
    """sys.exit(0)으로 테스트를 건너뛰는 코드는 통과로 잡히면 안 된다."""
    sneaky = "import sys\nsys.exit(0)\ndef add_two(a, b):\n    return a - b\n"
    r = run_candidate(sneaky, "assert add_two(2, 3) == 5")
    assert not r.ok, f"채점기가 속았다: {r}"
    assert r.reason == "early_exit", f"이유가 이상: {r.reason}"
    return "sys.exit(0) 우회 -> reason='early_exit' (차단)"


def c_os_exit_not_counted():
    """os._exit(0)도 같이 막혀야 한다."""
    sneaky = "import os\ndef add_two(a, b):\n    return 0\nos._exit(0)\n"
    r = run_candidate(sneaky, "assert add_two(2, 3) == 5")
    assert not r.ok, f"채점기가 속았다: {r}"
    return f"os._exit(0) 우회 -> reason='{r.reason}' (차단)"


def c_assert_vs_error():
    """실패 이유를 구분해야 디버깅이 된다."""
    r1 = run_candidate("def f():\n    return 1", "assert f() == 2")
    r2 = run_candidate("def f():\n    return 1 / 0", "f()")
    assert r1.reason == "assert", f"단정 실패를 못 구분: {r1.reason}"
    assert r2.reason == "error", f"예외를 못 구분: {r2.reason}"
    return "assert / error 구분됨"


def c_stdout_noise_ok():
    """모델이 쓸데없이 출력해도 정답이면 통과해야 한다."""
    code = "def add_two(a, b):\n    print('디버깅 출력')\n    return a + b\n"
    r = run_candidate(code, "assert add_two(1, 2) == 3")
    assert r.ok, f"출력이 있으면 떨어진다: {r}"
    return "표준출력 노이즈에도 정상 채점"


def c_truncate_at_dedent():
    """생성 결과에서 함수 본문만 잘라내야 한다."""
    prompt = "def f(x):\n"
    gen = prompt + "    return x + 1\n\ndef g():\n    return 99\nprint('쓰레기')\n"
    out = truncate_completion(prompt, gen)
    assert "def g" not in out, f"뒤따르는 코드를 못 잘랐다: {out!r}"
    assert "return x + 1" in out, f"본문이 사라졌다: {out!r}"
    assert is_valid_python(out), f"자른 결과가 유효한 파이썬이 아니다: {out!r}"
    return f"들여쓰기 풀리는 지점에서 절단: {out.strip()!r}"


def c_truncate_keeps_blank_and_nested():
    """빈 줄과 중첩 블록은 살려야 한다."""
    prompt = "def f(xs):\n"
    gen = prompt + "    total = 0\n\n    for x in xs:\n        total += x\n    return total\nEND = 1\n"
    out = truncate_completion(prompt, gen)
    assert "for x in xs" in out and "total += x" in out, f"중첩 블록 소실: {out!r}"
    assert "END" not in out, f"최상위 코드를 못 잘랐다: {out!r}"
    assert is_valid_python(out), "자른 결과가 유효하지 않다"
    return "빈 줄/중첩 블록 보존, 최상위 코드 절단"


def c_truncate_no_prompt_prefix():
    """모델 출력이 프롬프트로 시작하지 않는 경우도 처리해야 한다."""
    prompt = "def f(x):\n"
    out = truncate_completion(prompt, "    return x\nZ = 1\n")
    assert out.startswith(prompt), f"프롬프트가 안 붙었다: {out!r}"
    assert "Z = 1" not in out, "최상위 코드를 못 잘랐다"
    return "프롬프트 미포함 출력도 정상 처리"


def c_empty_completion():
    """빈 생성 결과는 문법 오류로 떨어져야 한다 (조용히 통과 금지)."""
    out = truncate_completion("def f(x):\n", "")
    r = run_candidate(out, "assert f(1) == 1")
    assert not r.ok, f"빈 본문인데 통과했다: {r}"
    return f"빈 본문 -> reason='{r.reason}'"


def main():
    print("=" * 60)
    print("평가 하네스 자체 검증")
    print("=" * 60)

    check("정답 전부 통과", c_solutions_pass)
    check("오답 전부 탈락", c_wrong_answers_fail)
    check("문법 오류 탐지", c_syntax_error_caught)
    check("무한루프 타임아웃", c_timeout_caught)
    check("sys.exit(0) 우회 차단", c_early_exit_not_counted)
    check("os._exit(0) 우회 차단", c_os_exit_not_counted)
    check("실패 이유 구분", c_assert_vs_error)
    check("표준출력 노이즈 허용", c_stdout_noise_ok)
    check("절단: 들여쓰기 기준", c_truncate_at_dedent)
    check("절단: 빈 줄/중첩 보존", c_truncate_keeps_blank_and_nested)
    check("절단: 프롬프트 미포함", c_truncate_no_prompt_prefix)
    check("빈 생성 결과 처리", c_empty_completion)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 채점기를 믿을 수 없다")
        return 1
    print("\n판정: 통과 - 채점 결과를 신뢰할 수 있다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
