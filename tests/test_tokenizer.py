"""토크나이저를 깨뜨리기 위한 테스트.

"동작한다"를 확인하는 게 아니라 "어떻게 깨지는가"를 찾는 게 목적이다.
전부 통과해야만 1단계를 통과로 친다.
"""

import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.bpe import BPETokenizer, END_OF_TEXT, pre_tokenize

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


# 적대적 입력 모음. 실제로 토크나이저를 죽이는 것들만 골랐다.
ADVERSARIAL = [
    ("빈 문자열", ""),
    ("공백 하나", " "),
    ("개행만", "\n"),
    ("CRLF", "a\r\nb\r\n"),
    ("탭 들여쓰기", "def f():\n\tif x:\n\t\treturn 1\n"),
    ("스페이스 들여쓰기", "def f():\n    if x:\n        return 1\n"),
    ("혼합 들여쓰기", "def f():\n\t  if x:\n  \treturn 1\n"),
    ("한글", "안녕하세요 파이썬 코드입니다"),
    ("한글 주석 코드", "# 한글 주석\ndef 함수(인자):\n    return 인자 * 2\n"),
    ("이모지", "x = '🔥🚀'  # 이모지 💯"),
    ("결합 문자", "e\u0301cole nai\u0308ve"),  # 조합형 악센트
    ("서로게이트 범위 밖 문자", "\U0001F600\U0010FFFF"),
    ("제로폭 문자", "a\u200bb\u200c\u200dc"),
    ("RTL", "شفرة عربية"),
    ("널 문자", "a\x00b"),
    ("제어 문자", "".join(chr(i) for i in range(1, 32))),
    ("모든 라틴1", "".join(chr(i) for i in range(32, 256))),
    ("긴 한 줄", "x = " + "1 + " * 5000 + "1"),
    ("깊은 들여쓰기", "\n".join(" " * (4 * i) + "pass" for i in range(1, 40))),
    ("연속 공백", "a" + " " * 500 + "b"),
    ("연속 개행", "a" + "\n" * 200 + "b"),
    ("f-string 중첩", 'f"{a!r:>{w}} {b["k"]}"'),
    ("독스트링", '"""여러 줄\n독스트링\n"""'),
    ("숫자 다양", "0x1F 0b1010 1_000_000 1e-5 3.14j"),
    ("연산자 뭉치", "a//=b;c**=d;e:=f;g@=h;i>>=j"),
    ("유니코드 식별자", "변수 = 1; π = 3.14; ℓ = 2"),
    ("특수토큰 문자열", f"before {END_OF_TEXT} after"),
    ("역슬래시", "path = 'C:\\\\Users\\\\SSAFY\\n'"),
]


def _make_tokenizer():
    """작은 코퍼스로 학습한 토크나이저. 테스트 전용."""
    corpus = [
        "def hello(name):\n    return f'hi {name}'\n",
        "class Foo:\n    def __init__(self):\n        self.x = 1\n",
        "import os\nimport sys\nfrom pathlib import Path\n",
        "for i in range(100):\n    print(i, i * 2)\n",
        "# 한글 주석도 넣는다\nresult = [x ** 2 for x in data if x > 0]\n",
        "try:\n    main()\nexcept Exception as e:\n    print(e)\n",
    ] * 40
    return BPETokenizer.train(corpus, vocab_size=1000, verbose=False)


TOK = _make_tokenizer()


def c_pretokenize_lossless():
    """사전 분할이 문자를 하나라도 잃으면 모든 게 무너진다."""
    bad = []
    for name, text in ADVERSARIAL:
        if "".join(pre_tokenize(text)) != text:
            bad.append(name)
    assert not bad, f"사전 분할에서 문자 소실: {bad}"
    return f"{len(ADVERSARIAL)}개 입력 전부 무손실"


def c_roundtrip_adversarial():
    """encode -> decode가 원문과 정확히 같아야 한다."""
    bad = []
    for name, text in ADVERSARIAL:
        got = TOK.decode(TOK.encode(text))
        if got != text:
            bad.append((name, repr(text[:40]), repr(got[:40])))
    assert not bad, f"왕복 실패 {len(bad)}건: {bad[:3]}"
    return f"{len(ADVERSARIAL)}개 적대적 입력 왕복 무손실"


def c_roundtrip_random_unicode():
    """무작위 유니코드 1000개. 사람이 생각 못 한 조합을 찾는다."""
    rng = random.Random(1234)
    bad = 0
    for _ in range(1000):
        n = rng.randint(1, 60)
        s = "".join(chr(rng.randint(1, 0x10FFFF)) for _ in range(n))
        # 서로게이트는 애초에 유효한 파이썬 str이 아니므로 제외
        try:
            s.encode("utf-8")
        except UnicodeEncodeError:
            continue
        if TOK.decode(TOK.encode(s)) != s:
            bad += 1
    assert bad == 0, f"무작위 유니코드 왕복 실패 {bad}건"
    return "무작위 유니코드 1000건 무손실"


def c_roundtrip_real_python():
    """실제 파이썬 소스로 왕복. 이 저장소의 코드를 쓴다."""
    root = Path(__file__).resolve().parent.parent
    files = sorted(root.rglob("*.py"))
    files = [f for f in files if ".venv" not in f.parts][:20]
    assert files, "테스트할 파이썬 파일을 못 찾았다"
    total = 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert TOK.decode(TOK.encode(text)) == text, f"왕복 실패: {f.name}"
        total += len(text)
    return f"{len(files)}개 파일 / {total:,}자 무손실"


def c_no_oov():
    """바이트 단위이므로 미등록 토큰이 나오면 안 된다."""
    ids = TOK.encode("전혀 학습 안 한 문자열 ¥€£ ᚠᚢᚦ �examples")
    assert all(0 <= i < TOK.vocab_size for i in ids), "어휘 범위 밖 id 발생"
    return f"모든 id가 [0, {TOK.vocab_size}) 범위 안"


def c_special_token():
    """특수 토큰이 단일 id로 잡히고 다시 복원돼야 한다."""
    ids = TOK.encode(f"a{END_OF_TEXT}b")
    eot = TOK.special_to_id[END_OF_TEXT]
    assert eot in ids, "특수 토큰이 단일 id로 안 잡힘"
    assert ids.count(eot) == 1, f"특수 토큰 개수 이상: {ids.count(eot)}"
    assert TOK.decode(ids) == f"a{END_OF_TEXT}b", "특수 토큰 복원 실패"
    # allow_special=False면 일반 텍스트로 쪼개져야 한다
    plain = TOK.encode(END_OF_TEXT, allow_special=False)
    assert eot not in plain, "allow_special=False인데 특수 id가 나옴"
    return f"특수 id={eot}, 켜고 끄기 모두 정상"


def c_determinism():
    """같은 입력은 항상 같은 출력. 캐시가 결과를 바꾸면 안 된다."""
    text = "def f(x):\n    return x + 1\n" * 5
    a = TOK.encode(text)
    b = TOK.encode(text)
    fresh = BPETokenizer(merges=TOK.merges, specials=TOK.specials).encode(text)
    assert a == b, "같은 인스턴스에서 결과가 달라짐"
    assert a == fresh, "캐시가 비어 있을 때와 결과가 다름"
    return f"{len(a)}토큰, 3회 일치"


def c_save_load():
    """저장하고 불러온 토크나이저가 완전히 동일해야 한다."""
    import tempfile

    text = "class A:\n    def m(self):\n        return 42\n"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "tok.json"
        TOK.save(p)
        loaded = BPETokenizer.load(p)
        assert loaded.vocab_size == TOK.vocab_size, "어휘 크기 불일치"
        assert loaded.merges == TOK.merges, "병합 규칙 불일치"
        assert loaded.encode(text) == TOK.encode(text), "인코딩 결과 불일치"
    return "저장/불러오기 후 완전 일치"


def _stdlib_corpus(n_files=40):
    """파이썬 표준 라이브러리 소스. 손쉽게 구할 수 있는 진짜 파이썬 코드다."""
    import sysconfig

    lib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(lib.glob("*.py"))[:n_files]
    texts = []
    for f in files:
        try:
            texts.append(f.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue
    return texts


def c_vocab_size_exact():
    """충분한 코퍼스에서는 예약 토큰 없이 요청 크기를 정확히 채워야 한다."""
    corpus = _stdlib_corpus()
    assert len(corpus) >= 20, f"표준 라이브러리 소스를 못 찾았다: {len(corpus)}개"
    t = BPETokenizer.train(corpus, vocab_size=2000, verbose=False)
    assert t.vocab_size == 2000, f"어휘 크기 {t.vocab_size} != 2000"
    assert t.n_reserved == 0, f"예약 토큰이 {t.n_reserved}개 생겼다 (코퍼스가 충분한데)"
    nbytes = sum(len(x.encode("utf-8")) for x in corpus)
    return f"stdlib {len(corpus)}파일/{nbytes:,}B -> vocab 2000 정확히, 예약 0개"


def c_vocab_too_small():
    """말이 안 되는 어휘 크기는 조용히 넘어가지 말고 죽어야 한다."""
    try:
        BPETokenizer.train(["abc"], vocab_size=100, verbose=False)
    except ValueError:
        return "vocab_size < 257이면 ValueError (의도된 동작)"
    raise AssertionError("너무 작은 vocab_size인데 예외가 안 났다")


def c_merge_ids_valid():
    """모든 병합 규칙은 자기보다 앞선 id만 참조해야 한다(순환 금지)."""
    for i, (a, b) in enumerate(TOK.merges):
        new_id = 256 + i
        assert a < new_id and b < new_id, f"병합 {i}가 미래 id 참조: ({a},{b})"
    return f"병합 {len(TOK.merges)}개 전부 순서 정상"


def c_compression_monotonic():
    """어휘를 키우면 압축이 나아져야 한다. 이게 BPE가 실제로 학습됐다는 증거다.

    (절대 압축률 기준은 여기서 재지 않는다. 장난감 코퍼스로 학습한
    토크나이저에 실전 수치를 요구하는 건 의미 없는 테스트다. 실제 수치는
    실코퍼스로 학습할 때 train_tokenizer.py에서 검증한다.)
    """
    root = Path(__file__).resolve().parent.parent
    text = (root / "tokenizer" / "bpe.py").read_text(encoding="utf-8")
    corpus = [text]
    nbytes = len(text.encode("utf-8"))

    ratios = []
    for vs in (300, 1000, 4000):
        t = BPETokenizer.train(corpus, vocab_size=vs, verbose=False)
        ratios.append(nbytes / len(t.encode(text)))

    assert all(r > 1.0 for r in ratios), f"압축이 전혀 안 됨: {ratios}"
    assert ratios[0] < ratios[1] < ratios[2], f"어휘를 키워도 압축이 안 나아짐: {ratios}"
    return "vocab 300/1000/4000 -> 바이트/토큰 " + "/".join(f"{r:.2f}" for r in ratios)


def c_vocab_padding_invariant():
    """병합이 소진돼도 vocab_size는 요청값을 정확히 지켜야 한다."""
    t = BPETokenizer.train(["abcabcabc def def ghi" * 50], vocab_size=400, verbose=False)
    assert t.vocab_size == 400, f"어휘 크기 {t.vocab_size} != 400"
    assert t.n_reserved > 0, "이 코퍼스면 예약 토큰이 생겨야 정상"
    # 예약 토큰이 실제로 쓰이지 않는지 확인
    ids = t.encode("abcabc def ghi 전혀 다른 텍스트")
    assert all(i < t.reserved_start for i in ids), "encode가 예약 토큰을 뱉었다"
    # 예약 토큰을 억지로 넣어도 디코딩이 죽지 않아야 한다
    assert t.decode([t.reserved_start, t.reserved_start + 1]) == "", "예약 토큰 디코딩 실패"
    return f"vocab=400 (병합 {len(t.merges)} + 예약 {t.n_reserved}), 예약 토큰 미사용"


def main():
    print("=" * 60)
    print("1단계: BPE 토크나이저 적대적 검증")
    print("=" * 60)

    check("사전 분할 무손실", c_pretokenize_lossless)
    check("적대적 입력 왕복", c_roundtrip_adversarial)
    check("무작위 유니코드 왕복", c_roundtrip_random_unicode)
    check("실제 파이썬 소스 왕복", c_roundtrip_real_python)
    check("미등록 토큰 없음", c_no_oov)
    check("특수 토큰 처리", c_special_token)
    check("결정성", c_determinism)
    check("저장/불러오기", c_save_load)
    check("어휘 크기 정확도", c_vocab_size_exact)
    check("어휘 크기 불변(예약 토큰)", c_vocab_padding_invariant)
    check("잘못된 어휘 크기 거부", c_vocab_too_small)
    check("병합 규칙 순서", c_merge_ids_valid)
    check("어휘 확대 시 압축 개선", c_compression_monotonic)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 1단계 통과 불가")
        return 1
    print("\n판정: 통과 - 다음 단계 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
