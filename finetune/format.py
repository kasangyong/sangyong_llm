"""인스트럭션 샘플 -> 모델 입력 문자열 포맷터.

새 특수 토큰은 만들지 않는다. 어휘는 16,384로 고정돼 있고 지금 그 어휘로
기반 모델이 학습 중이라, 토큰을 하나라도 추가하면 임베딩 크기가 바뀌어
학습분이 전부 무용지물이 된다. 그래서 구간 구분은 일반 텍스트 마커
("### Instruction:", "### Code:")로만 한다. 문서 끝맺음에만 기존
END_OF_TEXT를 쓴다.

마커를 ASCII로 쓰는 이유는 토큰 예산이다. 이 토크나이저는 파이썬 코드로
학습해서 한글 조각이 거의 병합되지 않았다. "### 지시:"는 9토큰인데
"### Instruction:"은 4토큰, "### 코드:" 9토큰 대 "### Code:" 3토큰이다.
샘플마다 11토큰씩 아끼면 1,024 컨텍스트에서 그만큼 코드가 더 들어간다.

프롬프트/완성 경계는 문자열이 아니라 토큰 인덱스로 돌려준다. 문자열로
자르면 지시문 안에 "### Code:"가 들어 있을 때 경계가 밀리고, 그 결과
손실 마스킹이 조용히 어긋난다. 인덱스는 그런 입력에도 안 흔들린다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.bpe import END_OF_TEXT, BPETokenizer

INSTRUCTION_MARKER = "### Instruction:"
CODE_MARKER = "### Code:"


def build_prompt(instruction: str) -> str:
    """지시문을 프롬프트 구간 문자열로 만든다. 완성 구간 직전까지."""
    # 지시문 양끝 공백은 떼어낸다. 안 그러면 같은 내용인데 마커와의 간격이
    # 샘플마다 달라져서 모델이 형식을 배우는 데 방해가 된다.
    return f"{INSTRUCTION_MARKER}\n{instruction.strip()}\n\n{CODE_MARKER}\n"


def build_completion(output: str) -> str:
    """출력(코드)을 완성 구간 문자열로 만든다. EOT는 아직 안 붙는다."""
    # 코드 끝의 잔여 공백을 정리하고 개행 하나로 끝맺는다. EOT 직전에 오는
    # 토큰을 일정하게 만들어 모델이 종료 지점을 배우기 쉽게 한다.
    body = output.rstrip()
    return body + "\n" if body else ""


def format_example(instruction: str, output: str) -> str:
    """EOT를 뺀 전체 문서 문자열. 사람이 눈으로 확인할 때 쓴다."""
    return build_prompt(instruction) + build_completion(output)


def encode_example(
    tok: BPETokenizer,
    instruction: str,
    output: str,
    add_eot: bool = True,
) -> tuple[list[int], int]:
    """(토큰 id 목록, 프롬프트 토큰 수)를 돌려준다.

    두 번째 값이 완성 구간의 시작 인덱스다. ids[:n_prompt]가 프롬프트,
    ids[n_prompt:]가 완성(+EOT)이다. 손실 마스킹이 이 값에 의존한다.

    프롬프트와 완성을 따로 인코딩해서 이어붙인다. 통짜로 인코딩하면 경계에
    걸친 BPE 병합이 생길 수 있어 "몇 번째 토큰부터 완성인가"가 원리적으로
    정의되지 않는다. 따로 인코딩하면 경계가 구성상 정확하다. 바이트 단위
    BPE라 이어붙여도 디코딩 결과는 통짜와 동일하다(왕복 무손실).

    allow_special=False로 인코딩한다. 지시문에 "<|endoftext|>" 문자열이
    들어 있어도 진짜 EOT 토큰으로 바뀌면 안 된다 — 그 자리에서 문서가
    끝난 것으로 학습된다.
    """
    prompt_ids = tok.encode(build_prompt(instruction), allow_special=False)
    completion_ids = tok.encode(build_completion(output), allow_special=False)
    if add_eot:
        completion_ids.append(tok.special_to_id[END_OF_TEXT])
    return prompt_ids + completion_ids, len(prompt_ids)


def extract_code(text: str) -> str:
    """생성 결과에서 코드 구간만 뽑는다. 추론 쪽에서 쓴다.

    문자열 기반이라 지시문에 CODE_MARKER가 들어 있으면 어긋난다. 학습
    데이터에서는 이 함수를 쓰지 않고(encode_example의 인덱스를 쓴다),
    make_dataset.py가 마커를 포함한 샘플을 아예 걸러낸다.
    """
    _, sep, tail = text.partition(CODE_MARKER + "\n")
    if not sep:
        return ""
    return tail.split(END_OF_TEXT, 1)[0]
