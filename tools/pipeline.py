"""모델 출력 -> 툴 호출 파싱 -> 검색 -> 컨텍스트 주입 -> 재호출.

모델과 검색 함수는 인자로 받는다(의존성 주입). 테스트에서 GPU도 네트워크도
쓰지 않기 위해서다.

무한 루프 방지가 이 파일의 핵심이다. SFT를 덜 받은 53M 모델은 컨텍스트에
`### 검색:`가 보이면 그 패턴을 그대로 따라 쓰는 경향이 있어서, 검색 결과를
넣어줄수록 또 검색을 부른다. max_calls를 넘으면 답을 못 만들었더라도 멈추고
stop_reason으로 알린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.protocol import ANSWER_MARKER, format_search_context, parse_tool_calls


@dataclass
class Turn:
    prompt: str
    output: str
    query: str = ""
    n_results: int = 0
    n_dropped: int = 0
    issues: list = field(default_factory=list)


@dataclass
class PipelineResult:
    final_output: str
    turns: list  # list[Turn]
    stop_reason: str  # "no_tool_call" | "max_calls" | "context_full"
    n_searches: int
    issues: list  # list[ParseIssue] — 마지막 출력의 파싱 문제


def run_with_search(
    prompt: str,
    generate,
    search,
    max_calls: int = 2,
    context_budget: int = 400,
    measure=len,
    max_results: int = 5,
    max_context: int | None = None,
) -> PipelineResult:
    """generate(prompt)->str, search(query)->list[SearchResult]를 엮어 돌린다.

    generate가 프롬프트를 포함한 전체 문자열을 돌려주든 이어쓴 부분만
    돌려주든 둘 다 받는다(eval/harness.truncate_completion과 같은 방침).

    context_budget은 검색 블록 하나의 상한이지 프롬프트 전체의 상한이 아니다.
    프롬프트는 라운드마다 (모델 출력 + 검색 블록)만큼 자라는데 모델 컨텍스트는
    ModelConfig.max_seq_len으로 고정이라, 상한을 안 걸면 두세 번째 라운드에서
    Transformer.forward가 "위치 N이 max_seq_len을 넘는다"로 죽거나, 왼쪽을
    잘라 넣는 배선(train/sample.py)에서는 지시문이 조용히 떨어져 나간다.
    실제로 물려서 확인한 값이다:

        max_context 없음, context_budget=600, max_calls=2
        -> 라운드별 프롬프트 61 / 969 / 1876 토큰 (max_seq_len=1024)

    그래서 모델과 함께 쓸 때는 max_context를 반드시 넘긴다. 단위는 measure가
    정한다 — 토크나이저로 재려면 둘을 같이 넘겨야 한다:

        run_with_search(
            ...,
            measure=lambda s: len(tok.encode(s, allow_special=False)),
            max_context=cfg.max_seq_len - max_new_tokens,
        )

    남은 자리에 맞춰 검색 블록을 줄이고, 그마저 없으면 검색을 쏘기 전에
    stop_reason="context_full"로 멈춘다(못 쓸 결과에 API 할당량을 태우지
    않는다). 기본값 None은 상한 없음이며, 모델 없이 파서만 쓸 때를 위한
    값이다.

    검색 실패(SearchError)는 잡지 않고 그대로 올린다. 여기서 삼키면 모델이
    빈 컨텍스트로 답을 지어내고, 그 답은 겉보기에 정상이라 못 잡는다.
    """
    if max_calls < 0:
        raise ValueError("max_calls는 0 이상이어야 한다")

    current = prompt
    turns: list[Turn] = []
    used = 0

    while True:
        raw = generate(current)
        out = raw[len(current):] if raw.startswith(current) else raw
        parsed = parse_tool_calls(out)

        if not parsed.calls:
            turns.append(Turn(current, out, issues=parsed.issues))
            return PipelineResult(out, turns, "no_tool_call", used, parsed.issues)

        if used >= max_calls:
            # 예산을 다 쓰고도 또 호출한다 = 루프. 결과를 더 주면 계속 돈다.
            turns.append(Turn(current, out, issues=parsed.issues))
            return PipelineResult(out, turns, "max_calls", used, parsed.issues)

        # 한 번에 하나만 처리한다. 모델이 여러 줄을 뱉어도 첫 호출 이후는
        # 검색 결과를 못 본 상태에서 쓴 것이라 신뢰할 수 없다.
        call = parsed.calls[0]
        # call.start/end는 parsed.text(개행 정규화본) 기준이다. 원본 out을
        # 그대로 자르면 "\r\n" 하나당 1자씩 밀려 호출 줄이 단어 중간에서
        # 잘린 채 다음 프롬프트에 들어간다.
        head = parsed.text[: call.end].rstrip("\n")

        # 마커는 줄 머리에서만 인정된다. 이어붙이는 자리마다 개행을 보장하지
        # 않으면 "### 답변: ### 검색: ..."처럼 한 줄에 붙어, 우리가 만든
        # 프롬프트를 우리 파서가 못 읽는 글이 된다. 마커 뒤를 개행으로 끝내는
        # 것도 finetune/format.py의 "### 코드:\n" 규약과 같은 모양이다.
        prefix = current if current.endswith("\n") else current + "\n"
        tail = f"{ANSWER_MARKER}\n"

        budget = context_budget
        if max_context is not None:
            room = max_context - measure(f"{prefix}{head}\n{tail}")
            budget = min(budget, room)
            if budget <= 0:
                turns.append(Turn(current, out, query=call.query, issues=parsed.issues))
                return PipelineResult(out, turns, "context_full", used, parsed.issues)

        results = search(call.query)[:max_results]
        used += 1

        block = format_search_context(results, budget, measure=measure)
        turns.append(
            Turn(
                prompt=current,
                output=out,
                query=call.query,
                n_results=block.n_included,
                n_dropped=block.n_dropped,
                issues=parsed.issues,
            )
        )
        current = f"{prefix}{head}\n{block.text}{tail}"
