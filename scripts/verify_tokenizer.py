"""실제 코퍼스로 학습된 토크나이저를 검증한다.

단위 테스트(tests/test_tokenizer.py)는 장난감 코퍼스로 로직을 확인한다.
이 스크립트는 실제로 학습에 쓸 tokenizer.json이 진짜 데이터에서
제대로 도는지 본다. 표본은 토크나이저 학습에 쓰지 않은 뒤쪽 문서에서 뽑는다.
"""

import gzip
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.bpe import END_OF_TEXT, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = ROOT / "data" / "processed"
TOK_PATH = ROOT / "tokenizer" / "tokenizer.json"

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def load_holdout(n=3000):
    """토크나이저 학습 표본(앞쪽 100MB)과 겹치지 않도록 마지막 샤드에서 뽑는다."""
    parts = sorted(PROC_DIR.glob("part-*.jsonl.gz"))
    if not parts:
        raise SystemExit("걸러낸 문서가 없다")
    docs = []
    with gzip.open(parts[-1], "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["c"])
            if len(docs) >= n:
                break
    return docs


TOK = BPETokenizer.load(TOK_PATH) if TOK_PATH.exists() else None
DOCS = load_holdout() if TOK_PATH.exists() else []


def c_vocab():
    assert TOK.vocab_size == 16384, f"어휘 크기 {TOK.vocab_size} != 16384"
    assert TOK.n_reserved == 0, f"예약 토큰이 {TOK.n_reserved}개 있다"
    assert len(TOK.merges) == 16127, f"병합 수 이상: {len(TOK.merges)}"
    assert TOK.vocab_size <= 65535, "uint16에 안 들어간다"
    return f"vocab 16,384 (병합 16,127 + 특수 1), 예약 0, uint16 적합"


def c_roundtrip_holdout():
    """토크나이저 학습에 안 쓴 문서 3000개 왕복. 하나라도 틀리면 데이터가 손상된다."""
    bad = []
    total = 0
    for i, d in enumerate(DOCS):
        if TOK.decode(TOK.encode(d, allow_special=False)) != d:
            bad.append(i)
        total += len(d.encode("utf-8"))
    assert not bad, f"왕복 실패 {len(bad)}건 (예: {bad[:3]})"
    return f"held-out {len(DOCS):,}문서 / {total / 1024**2:.1f}MB 무손실"


def c_compression():
    """코드 기준 압축률. 3.0 미만이면 토크나이저가 제 역할을 못 하는 것이다."""
    nbytes = sum(len(d.encode("utf-8")) for d in DOCS)
    ntok = sum(len(TOK.encode(d, allow_special=False)) for d in DOCS)
    ratio = nbytes / ntok
    assert ratio >= 3.0, f"압축률이 낮다: {ratio:.3f}"
    return f"바이트/토큰 = {ratio:.3f} (held-out 기준)"


def c_indent_efficiency():
    """코드 토크나이저의 핵심. 들여쓰기가 토큰을 낭비하면 안 된다."""
    code = "def f():\n    if x:\n        for i in y:\n            return i\n"
    ids = TOK.encode(code, allow_special=False)
    ratio = len(code.encode("utf-8")) / len(ids)
    assert ratio >= 2.5, f"들여쓰기 처리가 비효율적이다: {ratio:.2f} 바이트/토큰"
    return f"중첩 들여쓰기 코드 {len(ids)}토큰 ({ratio:.2f} 바이트/토큰)"


def c_eot_single_token():
    eot = TOK.special_to_id[END_OF_TEXT]
    ids = TOK.encode(f"a{END_OF_TEXT}b")
    assert ids.count(eot) == 1, "EOT가 단일 토큰이 아니다"
    assert eot == 16383, f"EOT id가 마지막이 아니다: {eot}"
    return f"EOT id={eot} (어휘 마지막)"


def c_adversarial_still_works():
    """실전 토크나이저도 이상한 입력에 무너지면 안 된다."""
    cases = [
        "",
        "\n\n\n",
        "\t\t대충 한글\t\t",
        "🔥" * 50,
        "".join(chr(i) for i in range(1, 256)),
        "x" * 10000,
        "def f():\r\n\treturn 1\r\n",
    ]
    for c in cases:
        assert TOK.decode(TOK.encode(c, allow_special=False)) == c, f"왕복 실패: {c[:20]!r}"
    return f"적대적 입력 {len(cases)}종 무손실"


def c_determinism_after_load():
    """저장/불러오기 후에도 같은 결과를 내야 한다."""
    fresh = BPETokenizer.load(TOK_PATH)
    d = DOCS[0]
    assert fresh.encode(d) == TOK.encode(d), "재로딩 후 인코딩이 다르다"
    return "재로딩 후 인코딩 일치"


def main():
    print("=" * 60)
    print("실전 토크나이저 검증 (tokenizer.json)")
    print("=" * 60)
    if TOK is None:
        print("tokenizer.json이 없다. data/prepare.py tokenizer를 먼저 돌릴 것.")
        return 1

    check("어휘 구성", c_vocab)
    check("held-out 왕복 무손실", c_roundtrip_holdout)
    check("압축률", c_compression)
    check("들여쓰기 효율", c_indent_efficiency)
    check("EOT 단일 토큰", c_eot_single_token)
    check("적대적 입력", c_adversarial_still_works)
    check("재로딩 결정성", c_determinism_after_load)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 이 토크나이저로 토큰화하면 안 된다")
        return 1
    print("\n판정: 통과 - 토큰화 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
