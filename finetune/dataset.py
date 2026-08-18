"""인스트럭션 JSONL -> (input_ids, labels) 쌍.

핵심은 손실 마스킹이다. 프롬프트 구간의 라벨을 -1로 덮어야 한다.
모델 forward가 cross_entropy(..., ignore_index=-1)을 쓰므로 -1인 자리는
손실에도 기울기에도 기여하지 않는다. 이걸 안 하면 모델이 "### 지시:"와
지시문 자체를 외우는 데 용량을 쓴다. 우리가 원하는 건 지시문이 주어졌을 때
코드를 내놓는 조건부 분포 하나뿐이다.

시프트 위치가 미묘하다. 모델은 forward 안에서 시프트하지 않고 targets를
그대로 쓴다(train.py의 BinDataset도 x=data[i:i+B], y=data[i+1:i+1+B]로
바깥에서 시프트한다). 그래서:

    input_ids = ids[:-1]
    labels    = ids[1:]

이고 labels[j]는 input_ids[j]까지 보고 예측할 대상이다. 완성 구간의 첫
토큰 ids[n_prompt]는 j = n_prompt-1에서 예측되므로 그 자리는 반드시
살려야 한다. 마스킹 구간은 labels[:n_prompt-1], 즉 n_prompt-1개다.
n_prompt개를 가리면 "프롬프트를 다 읽고 코드를 시작하는" 전이를 통째로
못 배운다 — 손실 곡선만 봐서는 절대 안 보이는 종류의 버그다.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from finetune.format import encode_example
from tokenizer.bpe import END_OF_TEXT, BPETokenizer

IGNORE_INDEX = -1


@dataclass
class BuildStats:
    """샘플이 어떻게 걸러졌는지. 조용히 버리지 않기 위해 전부 센다."""

    total: int = 0
    kept: int = 0
    dropped_too_long: int = 0
    dropped_empty_output: int = 0
    dropped_bad_record: int = 0
    max_len_seen: int = 0
    kept_tokens: int = 0
    supervised_tokens: int = 0

    def report(self) -> str:
        if self.kept == 0:
            return f"전체 {self.total:,}개 중 남은 샘플이 없다"
        frac = 100 * self.supervised_tokens / max(self.kept_tokens, 1)
        return (
            f"전체 {self.total:,} / 유지 {self.kept:,} "
            f"(길이초과 {self.dropped_too_long:,}, 빈출력 {self.dropped_empty_output:,}, "
            f"형식오류 {self.dropped_bad_record:,}) | "
            f"최장 {self.max_len_seen:,} 토큰 | 손실 대상 비율 {frac:.1f}%"
        )


def read_jsonl(path: Path) -> list[dict]:
    """{"instruction": ..., "output": ...} 한 줄씩."""
    if not Path(path).exists():
        raise FileNotFoundError(f"인스트럭션 데이터가 없다: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_example(
    tok: BPETokenizer, instruction: str, output: str, block_size: int
) -> tuple[list[int], list[int], int] | None:
    """샘플 하나를 (input_ids, labels, n_prompt)로. 못 쓰면 None.

    반환되는 두 목록의 길이는 len(ids)-1이며 패딩은 하지 않는다.
    패딩은 배치를 만들 때 배치 안 최대 길이에 맞춰서 한다.
    """
    ids, n_prompt = encode_example(tok, instruction, output)

    # 시프트하면 길이가 1 줄어드니 block_size+1까지가 들어간다.
    if len(ids) - 1 > block_size:
        return None
    if len(ids) < 2 or n_prompt < 1:
        return None

    input_ids = ids[:-1]
    labels = ids[1:]
    # 위 주석의 n_prompt-1. 프롬프트 마지막 토큰이 완성 첫 토큰을 예측하는
    # 자리는 살린다.
    for j in range(n_prompt - 1):
        labels[j] = IGNORE_INDEX
    return input_ids, labels, n_prompt


class SFTDataset:
    """길이 초과 샘플은 자르지 않고 버린다.

    자르면 두 가지가 조용히 망가진다. 완성 구간을 자르면 "함수를 중간에서
    끊는" 것을 정답으로 학습시키게 되고, 프롬프트를 자르면 마커 구조가
    깨져서 그 샘플은 형식 자체를 잘못 가르친다. 코드 SFT에서 미완성 출력을
    정답으로 주는 비용이 샘플 몇 개 잃는 비용보다 훨씬 크다. 그래서 버리고,
    몇 개를 버렸는지 stats에 남겨 보고한다.
    """

    def __init__(
        self,
        rows: list[dict],
        tok: BPETokenizer,
        block_size: int,
        pad_id: int | None = None,
        seed: int = 1337,
    ):
        self.block_size = block_size
        # 패딩 입력 토큰은 EOT를 쓴다. 어차피 그 자리 라벨은 -1이라 손실에
        # 영향이 없고, 인과 어텐션이라 뒤쪽 패딩이 앞쪽 결과를 바꾸지도 않는다.
        self.pad_id = (
            pad_id if pad_id is not None else tok.special_to_id[END_OF_TEXT]
        )
        self.stats = BuildStats()
        self.examples: list[tuple[list[int], list[int]]] = []

        for row in rows:
            self.stats.total += 1
            instruction = row.get("instruction")
            output = row.get("output")
            if not isinstance(instruction, str) or not isinstance(output, str):
                self.stats.dropped_bad_record += 1
                continue
            if not output.strip():
                # 출력이 비면 "지시를 보면 곧바로 EOT"를 가르치게 된다.
                self.stats.dropped_empty_output += 1
                continue

            built = build_example(tok, instruction, output, block_size)
            if built is None:
                # 얼마나 넘쳤는지 알아야 block_size 판단이 되므로 길이는 잰다.
                ids, _ = encode_example(tok, instruction, output)
                self.stats.max_len_seen = max(self.stats.max_len_seen, len(ids))
                # build_example은 길이 초과 말고 퇴화 샘플(len<2 등)에도 None을
                # 준다. 전부 길이 초과로 세면 block_size를 아무리 키워도 안
                # 줄어드는 탈락이 생겨 원인을 못 찾는다.
                if len(ids) - 1 > block_size:
                    self.stats.dropped_too_long += 1
                else:
                    self.stats.dropped_bad_record += 1
                continue

            input_ids, labels, _ = built
            self.examples.append((input_ids, labels))
            self.stats.kept += 1
            self.stats.kept_tokens += len(labels)
            self.stats.supervised_tokens += sum(
                1 for v in labels if v != IGNORE_INDEX
            )
            self.stats.max_len_seen = max(self.stats.max_len_seen, len(input_ids) + 1)

        self._order: list[int] = []
        self._cursor = 0
        self._rng = torch.Generator().manual_seed(seed)

    @classmethod
    def from_jsonl(cls, path: Path, tok: BPETokenizer, block_size: int, **kw):
        return cls(read_jsonl(Path(path)), tok, block_size, **kw)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.examples[i]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def collate(self, indices) -> tuple[torch.Tensor, torch.Tensor]:
        """배치 안 최대 길이에 맞춰 패딩한다.

        block_size(1024)에 맞춰 항상 패딩하면 실제 샘플이 150토큰 안팎이라
        연산의 85%가 패딩에 쓰인다. 패딩 자리의 라벨은 -1이므로 손실은
        동일하다.
        """
        batch = [self.examples[i] for i in indices]
        width = max(len(x) for x, _ in batch)
        x = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        y = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        for r, (input_ids, labels) in enumerate(batch):
            n = len(input_ids)
            x[r, :n] = torch.tensor(input_ids, dtype=torch.long)
            y[r, :n] = torch.tensor(labels, dtype=torch.long)
        return x, y

    def epoch_state(self) -> tuple[list[int], int]:
        """에포크 순회 위치. estimate_loss가 batch()를 부르면 커서가 같이
        움직여서 그 샘플들이 그 에포크의 학습에서 빠진다. 평가 전후로
        이 값을 저장/복원해서 '전부 한 번씩'을 지킨다."""
        return list(self._order), self._cursor

    def restore_epoch_state(self, state: tuple[list[int], int]) -> None:
        self._order, self._cursor = list(state[0]), state[1]

    def batch(self, batch_size: int, device, generator=None):
        """다음 배치. 에포크 단위로 섞어 순회한다.

        BinDataset처럼 매번 무작위로 뽑으면 SFT 데이터가 수천 개 규모라
        어떤 샘플은 한 번도 안 보고 어떤 샘플은 여러 번 본다. 여기서는
        전부 한 번씩 보는 쪽이 낫다.
        """
        if not self.examples:
            raise ValueError("샘플이 하나도 없다")
        gen = generator if generator is not None else self._rng
        picked: list[int] = []
        while len(picked) < batch_size:
            if self._cursor >= len(self._order):
                self._order = torch.randperm(
                    len(self.examples), generator=gen
                ).tolist()
                self._cursor = 0
            take = min(batch_size - len(picked), len(self._order) - self._cursor)
            picked.extend(self._order[self._cursor : self._cursor + take])
            self._cursor += take
        x, y = self.collate(picked)
        return x.to(device), y.to(device)
