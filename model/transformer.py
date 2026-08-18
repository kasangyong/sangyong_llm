"""디코더 전용 트랜스포머. 직접 구현.

구성은 현재 표준(Llama 계열)을 따른다:
  - RoPE (회전 위치 임베딩)
  - RMSNorm, pre-norm 배치
  - GQA (질의 헤드는 많고 KV 헤드는 적게 -> KV 캐시 메모리 절약)
  - SwiGLU 피드포워드
  - 입출력 임베딩 공유

PyTorch에서 가져다 쓰는 것은 텐서 연산, autograd, 그리고
scaled_dot_product_attention(융합 커널)뿐이다. 어텐션 수식, 위치 인코딩,
정규화, 블록 배선은 전부 아래에 직접 썼다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 640
    n_layers: int = 10
    n_heads: int = 10
    n_kv_heads: int = 2
    d_ff: int = 1728
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model({self.d_model})이 n_heads({self.n_heads})로 나뉘지 않는다")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads({self.n_heads})가 n_kv_heads({self.n_kv_heads})의 배수가 아니다"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


# 스모크 테스트용 소형 설정. 파이프라인 검증에만 쓴다.
SMOKE_CONFIG = ModelConfig(
    d_model=320, n_layers=6, n_heads=5, n_kv_heads=1, d_ff=864, max_seq_len=512
)


class RMSNorm(nn.Module):
    """LayerNorm에서 평균 빼기를 없앤 버전. 더 싸고 성능은 같다."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 정규화는 float32에서 한다. bf16으로 하면 제곱합에서 정밀도가 무너진다.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32
):
    """위치별 회전각의 cos/sin을 미리 계산한다. 형상: (seq_len, head_dim/2)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x에 회전 위치 임베딩을 적용한다.

    x: (B, n_heads, T, head_dim), cos/sin: (T, head_dim/2)
    앞 절반과 뒤 절반을 복소수의 실수부/허수부처럼 보고 회전시킨다.
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """GQA 인과 어텐션."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:
                # 캐시가 있는데 질의가 여러 개면 인과 마스크 정렬이 어긋난다.
                # 이 경로는 지원하지 않는다 (조용히 틀리느니 죽는 편이 낫다).
                if T > 1:
                    raise ValueError(
                        f"KV 캐시가 있는 상태에서 T={T} 질의는 지원하지 않는다. "
                        "프리필은 캐시 없이 한 번에, 이후는 T=1로 진행할 것."
                    )
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        # T==1이면 캐시된 과거 전체를 봐야 하므로 인과 마스크를 걸지 않는다.
        # T>1이면 캐시가 비어 있는 프리필이므로 정사각 인과 마스크가 맞다.
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=(T > 1), enable_gqa=(self.n_kv_heads != self.n_heads)
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache


class FeedForward(nn.Module):
    """SwiGLU. 게이트 한 갈래와 값 한 갈래를 곱한 뒤 되돌린다."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # 잔차 경로로 나가는 투영은 층 수에 맞춰 줄여 초기화한다.
        # 안 그러면 층을 쌓을수록 잔차 분산이 누적되어 학습 초반이 불안정하다.
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, device="cpu"
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """파라미터 수. 임베딩은 공유되므로 중복해서 세지 않는다."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.tok_emb.weight.numel()
        return total

    def forward(self, idx, targets=None, caches=None, start_pos=0):
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(
                f"위치 {start_pos + T}가 max_seq_len({self.cfg.max_seq_len})을 넘는다"
            )

        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]

        x = self.tok_emb(idx)
        new_caches = [] if caches is not None else None
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches is not None else None
            x, nc = block(x, cos, sin, cache)
            if new_caches is not None:
                new_caches.append(nc)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ):
        """KV 캐시를 쓴 자기회귀 생성."""
        self.eval()
        B, T = idx.shape

        # 프리필: 프롬프트 전체를 한 번에 넣고 캐시를 만든다.
        empty = [(None, None)] * self.cfg.n_layers
        logits, _, caches = self.forward(idx, caches=empty, start_pos=0)
        pos = T

        out = idx
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :]
            if temperature <= 0:
                nxt = logits_last.argmax(dim=-1, keepdim=True)
            else:
                logits_last = logits_last / temperature
                if top_k is not None:
                    k = min(top_k, logits_last.size(-1))
                    thresh = torch.topk(logits_last, k, dim=-1).values[:, -1:]
                    logits_last = logits_last.masked_fill(
                        logits_last < thresh, float("-inf")
                    )
                probs = F.softmax(logits_last, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
            if pos >= self.cfg.max_seq_len:
                break
            logits, _, caches = self.forward(nxt, caches=caches, start_pos=pos)
            pos += 1
        return out
