"""예시 인스트럭션 데이터셋 생성.

외부 API도 gated 데이터셋도 쓰지 않는다. 이미 받아서 필터까지 끝낸
data/processed/의 파이썬 코퍼스에서 (독스트링 -> 함수 구현) 쌍을 뽑는다.
독스트링은 사람이 쓴 진짜 자연어 설명이고 함수 본문은 그 설명에 대응하는
진짜 구현이라, 합성 데이터보다 분포가 정직하다.

모듈 최상위 함수만 쓴다. 메서드는 self와 클래스 상태에 의존해서 함수만
떼어놓으면 설명과 구현이 안 맞고, 중첩 함수는 클로저 변수에 의존한다.
품질 낮은 샘플 몇 배보다 좁고 깨끗한 샘플이 낫다.

출력은 finetune/data/에 쓴다. data/는 프리트레이닝이 memmap으로 물고
있으므로 건드리지 않는다.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import textwrap
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.prepare import iter_docs
from finetune.format import CODE_MARKER, INSTRUCTION_MARKER

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "finetune" / "data"

MIN_DOC_CHARS = 25
MAX_DOC_CHARS = 500
MIN_DOC_WORDS = 4
MIN_BODY_LINES = 2
MAX_FUNC_LINES = 60
VAL_EVERY = 20  # 20개마다 1개를 검증셋으로

# 설명이 아니라 자리표시자인 독스트링. 이런 걸 학습시키면 모델이
# "TODO"를 정답으로 배운다.
JUNK_DOC = re.compile(
    r"^(todo|fixme|xxx|docstring|doc|see above|n/?a|deprecated)\b", re.I
)
# 파라미터 절/doctest가 시작되는 줄. 요약 문단만 남기려고 쓴다.
SECTION_START = re.compile(
    r"^\s*(:param|:type|:return|:rtype|:raises|>>>|args:|arguments:|params:|"
    r"parameters:|returns:|yields:|raises:|examples?:|note:|-{3,}|={3,})",
    re.I,
)


def clean_docstring(doc: str) -> str | None:
    """요약 문단만 남긴다. 못 쓰겠으면 None.

    첫 빈 줄이나 파라미터 절 앞까지만 쓴다. 뒷부분(:param, >>>, Args:)은
    형식이 라이브러리마다 제각각이라 그대로 넣으면 지시문에 형식 노이즈만
    늘어난다.
    """
    lines: list[str] = []
    for raw in doc.strip().splitlines():
        line = raw.strip()
        if not line and lines:
            break
        if SECTION_START.match(line):
            break
        if line:
            lines.append(line)
    summary = " ".join(lines).strip()
    summary = re.sub(r"\s+", " ", summary)

    if not (MIN_DOC_CHARS <= len(summary) <= MAX_DOC_CHARS):
        return None
    if len(summary.split()) < MIN_DOC_WORDS:
        return None
    if JUNK_DOC.match(summary):
        return None
    # 마커가 지시문 안에 있으면 문자열 기반 파싱이 어긋난다. 데이터 쪽에서
    # 없애는 게 싸다 (손실 마스킹 자체는 토큰 인덱스라 영향 없다).
    if INSTRUCTION_MARKER in summary or CODE_MARKER in summary:
        return None
    if "<|endoftext|>" in summary:
        return None
    # 글자 비율이 낮으면 설명이 아니라 기호 덩어리다.
    alpha = sum(c.isalpha() for c in summary)
    if alpha / len(summary) < 0.5:
        return None
    return summary


def strip_docstring(node: ast.FunctionDef, src: str) -> str | None:
    """함수 소스에서 독스트링 문장만 지운다.

    독스트링을 남기면 지시문과 정답이 거의 같은 문자열이 되어 모델이
    "설명을 그대로 베끼기"를 학습한다.
    """
    segment = ast.get_source_segment(src, node)
    if segment is None:
        return None
    doc_node = node.body[0]
    # 함수 시작줄 기준 상대 줄번호. get_source_segment는 def 줄부터 준다.
    start = doc_node.lineno - node.lineno
    end = doc_node.end_lineno - node.lineno
    lines = segment.splitlines()
    if start < 1 or end >= len(lines):
        return None
    kept = lines[:start] + lines[end + 1 :]
    return "\n".join(kept)


def extract_pairs(content: str) -> list[tuple[str, str]]:
    """문서 하나에서 (설명, 구현) 쌍 목록."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return []

    out: list[tuple[str, str]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.decorator_list:
            # 데코레이터는 소스 세그먼트에 안 잡히고, 붙은 함수는 바깥
            # 프레임워크 문맥이 있어야 말이 된다.
            continue
        if node.name.startswith("_"):
            continue
        doc = ast.get_docstring(node, clean=True)
        if not doc:
            continue
        summary = clean_docstring(doc)
        if summary is None:
            continue

        code = strip_docstring(node, content)
        if code is None:
            continue
        code = textwrap.dedent(code).rstrip()

        body_lines = [l for l in code.splitlines()[1:] if l.strip()]
        if not (MIN_BODY_LINES <= len(body_lines)) or len(code.splitlines()) > MAX_FUNC_LINES:
            continue
        # 본문이 pass / ... / raise NotImplementedError뿐이면 가르칠 게 없다.
        stripped = "\n".join(body_lines).strip()
        if stripped in ("pass", "...") or stripped.startswith("raise NotImplementedError"):
            continue
        # 독스트링을 지운 결과가 여전히 유효한 파이썬인지 확인한다.
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        if "<|endoftext|>" in code:
            continue

        out.append((summary, code))
    return out


def cmd_build(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "sft_train.jsonl"
    val_path = OUT_DIR / "sft_val.jsonl"

    warnings.simplefilter("ignore", SyntaxWarning)
    warnings.simplefilter("ignore", DeprecationWarning)

    seen: set[str] = set()
    n_kept = n_dupe = n_docs = 0
    ftrain = open(train_path, "w", encoding="utf-8")
    fval = open(val_path, "w", encoding="utf-8")
    try:
        for _h, content in iter_docs():
            n_docs += 1
            for summary, code in extract_pairs(content):
                # 같은 함수가 여러 저장소에 복사돼 있는 경우가 많다.
                # 구현만으로 중복을 판단한다. (설명+구현)으로 잡으면 독스트링
                # 표현만 다른 같은 함수가 둘 다 남고, 분할 기준이 설명 해시라
                # 같은 구현이 train과 val에 갈라져 들어간다. 그러면 val loss가
                # 이미 외운 코드를 재는 값이 되고 best.pt가 그걸로 뽑힌다.
                key = hashlib.sha1(code.encode("utf-8")).hexdigest()
                if key in seen:
                    n_dupe += 1
                    continue
                seen.add(key)
                rec = json.dumps(
                    {"instruction": summary, "output": code}, ensure_ascii=False
                )
                # 분할 기준은 순번이 아니라 지시문 해시다. 순번으로 나누면
                # 같은 독스트링에 구현만 조금 다른 쌍(중복 제거 키가 달라
                # 둘 다 남는다)이 train과 val에 갈라져 들어가 val loss가
                # 낙관적으로 나온다. best.pt 선택이 그 값에 걸려 있다.
                bucket = int(
                    hashlib.sha1(summary.encode("utf-8")).hexdigest()[:8], 16
                )
                (fval if bucket % VAL_EVERY == 0 else ftrain).write(rec + "\n")
                n_kept += 1
                if n_kept >= args.target:
                    break
            if n_kept >= args.target:
                break
            if args.max_docs and n_docs >= args.max_docs:
                break
            if n_docs % 20000 == 0:
                print(f"[sft-data] 문서 {n_docs:,} 훑음, 샘플 {n_kept:,}개", flush=True)
    finally:
        ftrain.close()
        fval.close()

    n_val = sum(1 for _ in open(val_path, encoding="utf-8"))
    n_train = n_kept - n_val
    print("\n[sft-data] 완료")
    print(f"  훑은 문서 : {n_docs:,}")
    print(f"  샘플      : {n_kept:,} (중복 제거 {n_dupe:,})")
    print(f"  train     : {n_train:,} -> {train_path}")
    print(f"  val       : {n_val:,} -> {val_path}")
    if n_kept < 200:
        print("[sft-data] 경고: 200개 미만이다. --max-docs를 늘릴 것.")


def cmd_peek(args):
    """만들어진 샘플을 눈으로 확인한다. 데이터는 봐야 믿을 수 있다."""
    from finetune.format import format_example

    path = OUT_DIR / args.split
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            d = json.loads(line)
            print("=" * 60)
            print(format_example(d["instruction"], d["output"]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--target", type=int, default=5000, help="목표 샘플 수")
    p.add_argument("--max-docs", type=int, default=200_000, help="훑을 문서 상한")
    q = sub.add_parser("peek")
    q.add_argument("--split", default="sft_train.jsonl")
    q.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    {"build": cmd_build, "peek": cmd_peek}[args.cmd](args)


if __name__ == "__main__":
    main()
