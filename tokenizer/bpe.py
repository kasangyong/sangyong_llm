"""바이트 단위 BPE 토크나이저. 직접 구현.

바이트 단위로 가는 이유: 기본 어휘가 0~255 전체 바이트라서 어떤 입력이
들어와도 미등록 토큰(OOV)이 원리적으로 없다. 유니코드든 이모지든 깨진
바이트든 왕복이 무손실로 보장된다.

학습은 사전 분할(pre-tokenization)된 조각들의 빈도 사전 위에서 돌린다.
같은 조각이 수백만 번 나와도 한 번만 다루므로 순수 파이썬으로도 감당된다.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# 사전 분할 패턴. 모든 문자가 반드시 어느 한 갈래에는 걸리도록 짰다
# (조각들을 이어붙이면 원문과 정확히 같아야 한다 — 테스트로 강제한다).
# 코드에 맞춰 식별자/숫자/기호/들여쓰기를 각각 따로 끊는다.
PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"  # 영어 축약형
    r"| ?[^\W\d_]\w*"  # (앞 공백 1개까지) 글자로 시작하는 식별자
    r"| ?_\w*"  # (앞 공백 1개까지) 밑줄로 시작하는 식별자
    r"| ?\d{1,3}"  # (앞 공백 1개까지) 숫자 최대 3자리
    r"| ?[^\s\w]+[\r\n]*"  # (앞 공백 1개까지) 기호 뭉치 + 뒤따르는 개행
    r"|\s*[\r\n]"  # 개행
    r"|\s+(?!\S)"  # 줄 끝 공백
    r"|\s+",  # 들여쓰기 등 나머지 공백
    re.UNICODE,
)

END_OF_TEXT = "<|endoftext|>"
DEFAULT_SPECIALS = [END_OF_TEXT]


def pre_tokenize(text: str) -> list[str]:
    """텍스트를 조각으로 끊는다. 이어붙이면 원문과 같아야 한다."""
    return PAT.findall(text)


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        specials: list[str] | None = None,
        n_reserved: int = 0,
    ):
        self.merges: list[tuple[int, int]] = merges or []
        self.specials: list[str] = specials if specials is not None else list(DEFAULT_SPECIALS)
        # 코퍼스가 작아 병합이 일찍 소진되면 남는 자리를 예약 토큰으로 채운다.
        # vocab_size가 요청값과 달라지면 모델 임베딩 크기와 어긋나 조용히
        # 망가지므로, 크기를 불변으로 유지하는 쪽을 택했다.
        self.n_reserved: int = n_reserved
        self._rebuild()

    # ------------------------------------------------------------------ 내부

    def _rebuild(self) -> None:
        """merges 목록으로부터 순위표·바이트표·특수토큰 id를 다시 만든다."""
        # 병합 쌍 -> 순위(작을수록 먼저 적용)
        self.ranks: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self.merges)
        }
        # id -> 실제 바이트열
        self.token_bytes: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.token_bytes.append(self.token_bytes[a] + self.token_bytes[b])
        # 병합 쌍 -> 새 id
        self.pair_to_id: dict[tuple[int, int], int] = {
            pair: 256 + i for i, pair in enumerate(self.merges)
        }
        # 특수 토큰은 바이트 어휘 뒤에 붙인다
        base = 256 + len(self.merges)
        self.special_to_id: dict[str, int] = {
            s: base + i for i, s in enumerate(self.specials)
        }
        self.id_to_special: dict[int, str] = {
            v: k for k, v in self.special_to_id.items()
        }
        self._special_pat = (
            re.compile("|".join(re.escape(s) for s in self.specials))
            if self.specials
            else None
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials) + self.n_reserved

    @property
    def reserved_start(self) -> int:
        """이 id부터는 예약 토큰. 학습 데이터에 절대 나타나지 않는다."""
        return 256 + len(self.merges) + len(self.specials)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """조각 하나를 BPE로 병합한다. 순위가 가장 낮은(먼저 배운) 쌍부터 적용."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached

        parts = list(chunk.encode("utf-8"))
        while len(parts) >= 2:
            best_rank = None
            best_i = -1
            for i in range(len(parts) - 1):
                r = self.ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            new_id = self.pair_to_id[(parts[best_i], parts[best_i + 1])]
            parts[best_i : best_i + 2] = [new_id]

        # 캐시가 무한정 커지는 것을 막는다
        if len(self._cache) < 500_000:
            self._cache[chunk] = parts
        return parts

    # ------------------------------------------------------------------ 공개 API

    def encode(self, text: str, allow_special: bool = True) -> list[int]:
        """문자열 -> 토큰 id 목록."""
        if not text:
            return []

        if allow_special and self._special_pat is not None:
            out: list[int] = []
            pos = 0
            for m in self._special_pat.finditer(text):
                if m.start() > pos:
                    out.extend(self._encode_ordinary(text[pos : m.start()]))
                out.append(self.special_to_id[m.group()])
                pos = m.end()
            if pos < len(text):
                out.extend(self._encode_ordinary(text[pos:]))
            return out

        return self._encode_ordinary(text)

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for chunk in pre_tokenize(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """토큰 id 목록 -> 문자열. 특수 토큰은 원래 문자열로 되돌린다."""
        buf = bytearray()
        out: list[str] = []
        for i in ids:
            if i >= self.reserved_start:
                # 예약 토큰. 학습 데이터엔 없지만 샘플링이 뽑을 수는 있다.
                # 내용이 없으므로 건너뛴다.
                continue
            special = self.id_to_special.get(i)
            if special is not None:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(special)
            else:
                buf.extend(self.token_bytes[i])
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    # ------------------------------------------------------------------ 학습

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        specials: list[str] | None = None,
        verbose: bool = True,
    ) -> "BPETokenizer":
        """코퍼스에서 병합 규칙을 학습한다.

        빈도 사전 위에서 돌리고, 병합할 때마다 영향받은 단어의 쌍 카운트만
        갱신한다(전체 재집계 없음).
        """
        specials = specials if specials is not None else list(DEFAULT_SPECIALS)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size가 너무 작다: 최소 {256 + len(specials)} 필요"
            )

        # 1) 조각 빈도 집계
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(pre_tokenize(text))
        if verbose:
            print(f"[bpe] 고유 조각 {len(counter):,}개")

        # 2) 조각을 바이트 열로 펼친다
        words: list[list[int]] = []
        freqs: list[int] = []
        for chunk, f in counter.items():
            b = list(chunk.encode("utf-8"))
            if len(b) >= 2:  # 1바이트짜리는 병합에 기여하지 못한다
                words.append(b)
                freqs.append(f)

        # 3) 초기 쌍 카운트와 역인덱스
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_words: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            f = freqs[wi]
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                pair_counts[p] += f
                pair_words.setdefault(p, set()).add(wi)

        merges: list[tuple[int, int]] = []
        next_id = 256

        for step in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"[bpe] 병합할 쌍이 소진됨 (step {step})")
                break
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] <= 0:
                break

            merges.append(best)
            new_id = next_id
            next_id += 1

            affected = list(pair_words.get(best, ()))
            for wi in affected:
                w = words[wi]
                f = freqs[wi]

                # 이 단어가 기여하던 쌍 카운트를 전부 뺀다
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        pair_words.pop(p, None)
                    else:
                        s = pair_words.get(p)
                        if s is not None:
                            s.discard(wi)

                # 병합 수행
                out: list[int] = []
                i = 0
                a, b = best
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out

                # 바뀐 단어의 쌍 카운트를 다시 더한다
                for i in range(len(out) - 1):
                    p = (out[i], out[i + 1])
                    pair_counts[p] += f
                    pair_words.setdefault(p, set()).add(wi)

            pair_counts.pop(best, None)
            pair_words.pop(best, None)

            if verbose and (step + 1) % 1000 == 0:
                print(f"[bpe] 병합 {step + 1:,}/{n_merges:,}")

        n_reserved = n_merges - len(merges)
        if n_reserved > 0:
            # 조용히 넘어가면 모델 임베딩 크기와 어긋난다. 반드시 알린다.
            print(
                f"[bpe] 경고: 병합할 쌍이 소진되어 {len(merges):,}개만 학습했다. "
                f"vocab_size를 맞추기 위해 예약 토큰 {n_reserved:,}개를 채운다. "
                "코퍼스가 더 크면 이 경고는 사라진다."
            )
        if verbose:
            print(f"[bpe] 병합 {len(merges):,}개 학습 완료")
        return cls(merges=merges, specials=specials, n_reserved=n_reserved)

    # ------------------------------------------------------------------ 저장/불러오기

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "specials": self.specials,
            "n_reserved": self.n_reserved,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("version") != 1:
            raise ValueError(f"모르는 토크나이저 형식: {payload.get('version')}")
        merges = [(a, b) for a, b in payload["merges"]]
        return cls(
            merges=merges,
            specials=payload["specials"],
            n_reserved=payload.get("n_reserved", 0),
        )
