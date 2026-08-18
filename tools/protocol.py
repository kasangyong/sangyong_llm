"""툴 호출 포맷 정의와 파서.

포맷을 이렇게 고른 이유:

  ### 검색: 파이썬 리스트 정렬

한 줄, 인자 하나, 줄바꿈으로 끝. JSON이나 XML 태그를 쓰지 않는다.
53M 모델은 중첩 구조를 닫는 것(따옴표, 중괄호, 태그)을 거의 못 배운다.
여는 것과 닫는 것 사이의 의존이 길어질수록 실패율이 올라간다. 반면 이
포맷은 "줄 시작 마커 + 한 줄"이라 닫을 것이 없고, 데이터셋에 이미 있는
`### 지시:` / `### 코드:` 형태와 같은 모양이라 SFT에서 재사용된다.

특수 토큰은 쓰지 않는다. 어휘가 16,384로 고정돼 있고 임베딩 크기가 바뀌면
학습 중인 모델이 전부 무용지물이 된다. 마커는 전부 일반 텍스트다.

파서는 조용히 넘어가지 않는다. 잘린 호출이나 빈 질의는 무시하지 않고
ParseIssue로 보고한다. 검색 없이 답한 것과 검색이 깨져서 답한 것은
디버깅할 때 완전히 다른 문제다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 줄 시작에서만 인정한다. 코드 안의 `x = "### Search: ..."`는 마커가 줄 머리에
# 오지 않으므로 자연히 걸러진다.
#
# 마커를 ASCII로 쓰는 이유는 순전히 토큰 예산이다. 이 토크나이저는 파이썬
# 코드로 학습해서 한글 조각이 거의 병합되지 않았다. 실측:
#   "### 검색:" 9토큰 vs "### Search:" 3토큰
#   "### 검색결과:" 15토큰 vs "### Results:" 4토큰
#   "### 답변:" 9토큰 vs "### Answer:" 5토큰
# 검색 한 라운드에 33토큰 -> 12토큰. 컨텍스트가 1,024뿐이라 라운드마다
# 21토큰씩 아끼는 것은 결과 스니펫 하나 분량이다.
CALL_MARKER = "### Search:"
RESULT_MARKER = "### Results:"
ANSWER_MARKER = "### Answer:"

# 질의 길이 상한. 컨텍스트가 1024 토큰이라 이보다 긴 질의는 어차피 예산을
# 먹기만 하고, 대개 모델이 줄을 못 끊고 흘러넘친 경우다.
MAX_QUERY_CHARS = 200


@dataclass
class ToolCall:
    query: str
    line_no: int  # 1-based
    start: int  # 정규화된 텍스트 기준 마커 시작 오프셋
    end: int  # 호출 줄 끝(개행 제외) 오프셋
    truncated_query: bool = False


@dataclass
class ParseIssue:
    kind: str  # "truncated" | "empty_query" | "query_too_long" | "masked" | "indented"
    line_no: int
    detail: str = ""


@dataclass
class ParseResult:
    text: str  # 개행이 정규화된 원본
    calls: list[ToolCall] = field(default_factory=list)
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """호출을 하나라도 뽑았고 심각한 문제가 없으면 True."""
        return bool(self.calls) and not any(
            i.kind in ("truncated", "empty_query") for i in self.issues
        )


@dataclass
class SearchContext:
    text: str
    n_included: int
    n_dropped: int


# ------------------------------------------------------------------ 마스킹

_DELIMS = ("```", '"""', "'''")


def _masked_spans(text: str) -> list[tuple[int, int]]:
    """마커를 무시해야 하는 구간(펜스 코드블록, 삼중따옴표 문자열)을 찾는다.

    모델이 코드를 쓰다가 독스트링 안에 `### 검색:`를 우연히 넣는 경우가 있다.
    닫히지 않은 구간은 끝까지 마스킹한다 — 잘린 코드블록 안의 마커를 실제
    호출로 오인해서 엉뚱한 검색을 쏘는 쪽이 더 나쁘다.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        best = -1
        delim = ""
        for d in _DELIMS:
            j = text.find(d, i)
            if j != -1 and (best == -1 or j < best):
                best, delim = j, d
        if best == -1:
            break
        end = text.find(delim, best + len(delim))
        if end == -1:
            spans.append((best, len(text)))
            break
        spans.append((best, end + len(delim)))
        i = end + len(delim)
    return spans


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


# -------------------------------------------------------------------- 파서


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _clean_query(raw: str) -> str:
    q = raw.strip()
    # 모델이 질의를 따옴표로 감싸는 버릇을 자주 보인다. 한 겹만 벗긴다.
    for ch in ('"', "'", "`"):
        # 그 따옴표가 정확히 두 번(양끝)만 나올 때에만 벗긴다. 개수를 안 세면
        # `'sorted' vs 'sort'` 같은 질의가 `sorted' vs 'sort`로 망가진다.
        # 감싼 게 아니라 안에서 인용한 경우이므로 건드리면 안 된다.
        if len(q) >= 2 and q[0] == ch and q[-1] == ch and q.count(ch) == 2:
            q = q[1:-1].strip()
            break
    return re.sub(r"\s+", " ", q)


def parse_tool_calls(text: str, max_query_chars: int = MAX_QUERY_CHARS) -> ParseResult:
    """모델 출력에서 검색 호출을 뽑는다.

    앞뒤 잡음, 여러 호출, 호출 없음, 개행 이상(\\r\\n), 코드 안의 우연한
    마커를 모두 처리한다. 잘린 호출과 빈 질의는 버리되 issues에 남긴다.
    """
    if not text:
        return ParseResult(text="")

    norm = normalize_newlines(text)
    spans = _masked_spans(norm)
    res = ParseResult(text=norm)

    offset = 0
    lines = norm.split("\n")
    last_idx = len(lines) - 1
    # split 결과의 마지막 조각은 원문이 개행으로 끝나면 빈 문자열이다.
    # 즉 "개행 없이 끝난 줄"은 마지막 조각이 비어있지 않은 경우뿐이다.
    for idx, line in enumerate(lines):
        start = offset
        offset += len(line) + 1
        has_newline = idx != last_idx

        if not line.startswith(CALL_MARKER):
            # 마커 자체가 중간에서 잘린 경우 (예: 출력이 "### 검" 에서 끝남)
            if (
                not has_newline
                and line
                and CALL_MARKER.startswith(line)
                and line.startswith("#")
            ):
                res.issues.append(
                    ParseIssue("truncated", idx + 1, f"마커가 잘렸다: {line!r}")
                )
            # 들여쓴 마커는 호출로 인정하지 않는다. 파이썬에서 "    ### 검색: x"는
            # 그냥 주석이라 생성 코드 안에서 얼마든지 나오고, 그걸 실행하면
            # 엉뚱한 검색을 쏜다. 다만 조용히 버리면 "모델이 툴을 안 불렀다"와
            # 구별이 안 되므로 보고는 남긴다.
            elif line.lstrip().startswith(CALL_MARKER):
                res.issues.append(
                    ParseIssue("indented", idx + 1, "들여쓴 마커는 호출이 아니다")
                )
            continue

        if _in_spans(start, spans):
            res.issues.append(
                ParseIssue("masked", idx + 1, "코드/문자열 구간 안의 마커는 무시")
            )
            continue

        query = _clean_query(line[len(CALL_MARKER):])
        if not query:
            kind = "empty_query" if has_newline else "truncated"
            res.issues.append(ParseIssue(kind, idx + 1, "질의가 비어 있다"))
            continue

        clipped = False
        if len(query) > max_query_chars:
            res.issues.append(
                ParseIssue(
                    "query_too_long", idx + 1, f"{len(query)}자 -> {max_query_chars}자"
                )
            )
            query = query[:max_query_chars].rstrip()
            clipped = True

        res.calls.append(
            ToolCall(
                query=query,
                line_no=idx + 1,
                start=start,
                end=start + len(line),
                truncated_query=clipped,
            )
        )
    return res


def format_tool_call(query: str) -> str:
    """학습 데이터/프롬프트에 넣을 호출 한 줄을 만든다."""
    q = _clean_query(query)
    if not q:
        raise ValueError("빈 질의로는 툴 호출을 만들 수 없다")
    return f"{CALL_MARKER} {q}\n"


# ------------------------------------------------------------- 결과 포맷터


_LINE_MARKERS = (CALL_MARKER, RESULT_MARKER, ANSWER_MARKER)


def _neutralize_markers(text: str) -> str:
    """줄 머리의 프로토콜 마커를 깬다.

    검색 결과는 우리가 쓴 문자열이 아니라 바깥에서 온 문자열이다. 스니펫이
    "### 검색:"으로 시작하면 주입된 블록이 그대로 다음 프롬프트에 들어가고,
    다음 턴의 파서가 그걸 모델이 낸 호출로 읽어 검색 결과가 고른 질의를
    그대로 검색한다(질의를 결과 제공자에게 넘겨주는 셈이다). 마커는 줄
    머리에서만 인정되므로 앞에 한 칸을 넣으면 무력화된다.

    첫 줄만 보지 않고 모든 줄을 본다. 필드 하나가 개행을 품고 들어오면
    (예: url) 마커는 두 번째 줄 머리에 생기고, 첫 줄만 검사하면 그대로
    빠져나간다.
    """
    return "\n".join(
        " " + ln if ln.startswith(_LINE_MARKERS) else ln
        for ln in normalize_newlines(text).split("\n")
    )


def _render_entry(i: int, r) -> tuple[str, str]:
    """검색 결과 하나를 (머리줄, 스니펫)으로 렌더링한다. search.SearchResult 호환."""
    title = re.sub(r"\s+", " ", (getattr(r, "title", "") or "")).strip() or "(제목 없음)"
    snippet = re.sub(r"\s+", " ", (getattr(r, "snippet", "") or "")).strip()
    # url의 공백은 남기지 않고 지운다. 제목/스니펫과 달리 여기는 공백을
    # 하나로 줄이는 것으로 부족하다 — 개행이 하나라도 남으면 머리줄이 두 줄로
    # 쪼개져 뒷줄이 마커 자리가 된다. 정상 url에는 애초에 공백이 없다.
    url = re.sub(r"\s+", "", (getattr(r, "url", "") or ""))
    head = f"[{i}] {title} ({url})" if url else f"[{i}] {title}"
    return _neutralize_markers(head), _neutralize_markers(snippet)


def _fit_entry(head: str, snippet: str, limit: int, measure) -> str:
    """머리줄은 건드리지 않고 스니펫만 줄인다.

    머리줄에는 URL이 들어 있다. 여기를 자르면 `(https://docs.python.org/ko/3/...`
    같은 절반짜리 주소가 컨텍스트에 들어가고, 모델은 그걸 실제 출처인 양
    인용한다. 지어낸 URL과 구별되지 않는 조용한 실패라, 자를 자리가 모자라면
    스니펫을 통째로 버리고 머리줄만 남긴다.
    """
    if not snippet:
        return head
    full = f"{head}\n{snippet}"
    if measure(full) <= limit:
        return full
    if measure(head) >= limit:
        return head

    lo, hi = 0, len(snippet)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if measure(f"{head}\n{snippet[:mid].rstrip()}...") <= limit:
            lo = mid
        else:
            hi = mid - 1
    return f"{head}\n{snippet[:lo].rstrip()}..." if lo else head


def format_search_context(
    results,
    budget: int,
    measure=len,
    snippet_chars: int = 300,
    entry_budget: int | None = None,
) -> SearchContext:
    """검색 결과를 모델 컨텍스트에 넣을 블록으로 만든다.

    budget은 measure로 잰 길이의 상한이다. measure 기본값은 문자 수이고,
    토크나이저를 쓰려면 measure=lambda s: len(tok.encode(s, allow_special=False))
    를 넘긴다. 컨텍스트가 1024 토큰뿐이라 넘치는 건 반드시 버린다.

    두 단계로 줄인다.
      1) 결과 하나가 쓸 수 있는 길이를 entry_budget으로 제한한다(기본 예산의
         1/3). 이게 없으면 한국어처럼 토큰이 비싼 스니펫 하나가 예산을 통째로
         먹고 나머지가 전부 생략된다. 실제 토크나이저로 재보고 넣은 장치다.
      2) 그 다음 문서 경계에서만 자른다. 문서 목록 중간을 잘라 넣으면 모델이
         반쯤 읽은 사실로 나머지를 지어낸다.
    """
    per_entry = entry_budget if entry_budget is not None else max(1, budget // 3)
    entries = []
    for i, r in enumerate(results, 1):
        head, snippet = _render_entry(i, r)
        if len(snippet) > snippet_chars:
            snippet = snippet[:snippet_chars].rstrip() + "..."
        # 머리줄(제목+url)은 최소 단위다. 스니펫을 다 버려도 이건 남긴다.
        entries.append(_fit_entry(head, snippet, per_entry, measure))

    def build(items: list[str], dropped: int) -> str:
        parts = [RESULT_MARKER]
        parts.extend(items)
        if dropped > 0:
            parts.append(f"(길이 예산 초과로 {dropped}개 생략)")
        return "\n".join(parts) + "\n"

    included: list[str] = []
    for e in entries:
        trial = included + [e]
        if measure(build(trial, len(entries) - len(trial))) > budget:
            break
        included = trial

    dropped = len(entries) - len(included)
    text = build(included, dropped)
    if measure(text) > budget:
        # 헤더조차 안 들어가는 예산. 빈 블록을 주되 전부 버렸다고 알린다.
        return SearchContext("", 0, len(entries))
    return SearchContext(text, len(included), dropped)
