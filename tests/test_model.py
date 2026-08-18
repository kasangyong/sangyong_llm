"""트랜스포머를 깨뜨리기 위한 테스트.

핵심은 인과성 검증이다. 인과 마스크가 새면 모델은 미래를 보고 답을 맞히므로
학습 손실은 아름답게 떨어지지만 생성은 전혀 안 된다. 손실 곡선만 봐서는
절대 못 잡는 종류의 버그라 여기서 잡아야 한다.
"""

import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.transformer import (
    ModelConfig,
    Transformer,
    apply_rope,
    build_rope_cache,
)

RESULTS = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


TEST_CFG = ModelConfig(
    vocab_size=512, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2,
    d_ff=256, max_seq_len=64,
)


def _model(cfg=TEST_CFG, seed=0):
    torch.manual_seed(seed)
    m = Transformer(cfg).to(DEVICE)
    m.eval()
    return m


# --------------------------------------------------------------- 인과성

def c_causal_no_future_leak():
    """미래 토큰을 바꿔도 과거 위치의 출력은 변하면 안 된다."""
    m = _model()
    torch.manual_seed(1)
    T = 32
    idx = torch.randint(0, TEST_CFG.vocab_size, (2, T), device=DEVICE)

    with torch.no_grad():
        base, _, _ = m(idx)

    worst = 0.0
    for cut in (8, 16, 24):
        tampered = idx.clone()
        # cut 이후를 전부 다른 토큰으로 바꾼다
        tampered[:, cut:] = (tampered[:, cut:] + 137) % TEST_CFG.vocab_size
        with torch.no_grad():
            other, _, _ = m(tampered)
        diff = (base[:, :cut, :] - other[:, :cut, :]).abs().max().item()
        worst = max(worst, diff)

    assert worst < 1e-5, f"미래 정보가 과거로 샌다. 최대 차이={worst:.3e}"
    return f"미래 토큰 변조에도 과거 출력 불변 (최대차 {worst:.2e})"


def c_causal_prefix_invariance():
    """길이 T로 넣었을 때의 앞부분 출력 == 길이 t로 잘라 넣었을 때의 출력."""
    m = _model()
    torch.manual_seed(2)
    idx = torch.randint(0, TEST_CFG.vocab_size, (1, 40), device=DEVICE)
    with torch.no_grad():
        full, _, _ = m(idx)
    worst = 0.0
    for t in (1, 7, 23, 40):
        with torch.no_grad():
            part, _, _ = m(idx[:, :t])
        worst = max(worst, (full[:, :t, :] - part).abs().max().item())
    assert worst < 1e-5, f"접두사 불변성 위반: {worst:.3e}"
    return f"접두사 불변성 유지 (최대차 {worst:.2e})"


# --------------------------------------------------------------- RoPE

def c_rope_relative_position():
    """RoPE의 존재 이유: q·k 내적이 절대 위치가 아니라 상대 거리에만 의존해야 한다."""
    head_dim = 32
    cos, sin = build_rope_cache(64, head_dim, 10000.0, device="cpu")
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def score(i, j):
        qi = apply_rope(q, cos[i : i + 1], sin[i : i + 1])
        kj = apply_rope(k, cos[j : j + 1], sin[j : j + 1])
        return (qi * kj).sum().item()

    # 거리 3인 쌍들의 점수가 전부 같아야 한다
    vals = [score(i, i - 3) for i in (3, 10, 25, 50)]
    spread = max(vals) - min(vals)
    assert spread < 1e-4, f"같은 상대거리인데 점수가 다르다: {vals}"
    # 거리가 다르면 점수도 달라야 한다 (아무것도 안 하는 게 아님을 확인)
    assert abs(score(10, 7) - score(10, 2)) > 1e-4, "상대거리가 달라도 점수가 같다"
    return f"거리 3 고정 시 점수 편차 {spread:.2e}, 거리 다르면 점수 변함"


def c_rope_norm_preserved():
    """회전이므로 벡터 길이를 바꾸면 안 된다."""
    head_dim = 64
    cos, sin = build_rope_cache(32, head_dim, 10000.0, device="cpu")
    torch.manual_seed(4)
    x = torch.randn(2, 3, 32, head_dim)
    y = apply_rope(x, cos, sin)
    diff = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
    assert diff < 1e-4, f"RoPE가 길이를 바꾼다: {diff:.3e}"
    return f"길이 보존 (최대차 {diff:.2e})"


# --------------------------------------------------------------- KV 캐시

def c_kv_cache_matches_full():
    """캐시로 한 토큰씩 넣은 결과가 통짜 forward와 같아야 한다.

    캐시 버그는 생성 품질만 망가뜨리고 학습 지표에는 안 나타나서 놓치기 쉽다.
    """
    m = _model()
    torch.manual_seed(5)
    idx = torch.randint(0, TEST_CFG.vocab_size, (1, 20), device=DEVICE)
    with torch.no_grad():
        full, _, _ = m(idx)

        # 프리필 8개 + 한 토큰씩 12개
        caches = [(None, None)] * TEST_CFG.n_layers
        logits, _, caches = m(idx[:, :8], caches=caches, start_pos=0)
        got = [logits]
        for t in range(8, 20):
            logits, _, caches = m(idx[:, t : t + 1], caches=caches, start_pos=t)
            got.append(logits)
        stitched = torch.cat(got, dim=1)

    diff = (full - stitched).abs().max().item()
    assert diff < 1e-4, f"KV 캐시 결과가 통짜 forward와 다르다: {diff:.3e}"
    return f"캐시 경로 == 통짜 경로 (최대차 {diff:.2e})"


def c_kv_cache_rejects_bad_use():
    """캐시가 찬 상태에서 여러 토큰을 넣으면 조용히 틀리지 말고 죽어야 한다."""
    m = _model()
    idx = torch.randint(0, TEST_CFG.vocab_size, (1, 8), device=DEVICE)
    caches = [(None, None)] * TEST_CFG.n_layers
    with torch.no_grad():
        _, _, caches = m(idx, caches=caches, start_pos=0)
        try:
            m(idx[:, :3], caches=caches, start_pos=8)
        except ValueError:
            return "지원 안 하는 캐시 사용은 ValueError (의도된 동작)"
    raise AssertionError("잘못된 캐시 사용인데 조용히 통과했다")


# --------------------------------------------------------------- 구조/학습

def c_param_count():
    """실제 파라미터 수가 손계산과 맞는가."""
    cfg = ModelConfig()  # 목표 설정
    m = Transformer(cfg)
    n = m.num_params()

    emb = cfg.vocab_size * cfg.d_model
    hd = cfg.head_dim
    attn = cfg.d_model * cfg.n_heads * hd * 2 + cfg.d_model * cfg.n_kv_heads * hd * 2
    ffn = 3 * cfg.d_model * cfg.d_ff
    norms = 2 * cfg.d_model
    expect = emb + cfg.n_layers * (attn + ffn + norms) + cfg.d_model

    assert n == expect, f"파라미터 수 불일치: 실제 {n:,} vs 계산 {expect:,}"
    assert 50e6 < n < 57e6, f"설계 목표(약 53M)에서 벗어남: {n:,}"
    return f"{n:,} 파라미터 (손계산과 일치, 비임베딩 {m.num_params(True):,})"


def c_weight_tying():
    """입출력 임베딩이 실제로 같은 텐서여야 한다."""
    m = _model()
    assert m.lm_head.weight is m.tok_emb.weight, "가중치 공유가 안 됨"
    n_shared = Transformer(ModelConfig(tie_embeddings=True)).num_params()
    n_split = Transformer(ModelConfig(tie_embeddings=False)).num_params()
    gap = n_split - n_shared
    expect = ModelConfig().vocab_size * ModelConfig().d_model
    assert gap == expect, f"공유 여부 차이가 이상: {gap:,} != {expect:,}"
    return f"동일 텐서 확인, 미공유 대비 {gap:,} 절약"


def c_init_loss():
    """초기 손실이 균등분포 기대값 ln(V)에 가까워야 한다. 초기화가 망가지면 벗어난다."""
    m = _model(seed=7)
    torch.manual_seed(7)
    idx = torch.randint(0, TEST_CFG.vocab_size, (4, 32), device=DEVICE)
    tgt = torch.randint(0, TEST_CFG.vocab_size, (4, 32), device=DEVICE)
    with torch.no_grad():
        _, loss, _ = m(idx, targets=tgt)
    expect = math.log(TEST_CFG.vocab_size)
    assert abs(loss.item() - expect) < 0.3, f"초기 손실 {loss.item():.3f} vs 기대 {expect:.3f}"
    return f"loss={loss.item():.4f}, ln(V)={expect:.4f}"


def c_gradients_flow():
    """모든 파라미터에 기울기가 흘러야 한다. 안 쓰이는 층이 있으면 여기서 걸린다."""
    torch.manual_seed(8)
    m = Transformer(TEST_CFG).to(DEVICE)
    m.train()
    idx = torch.randint(0, TEST_CFG.vocab_size, (2, 16), device=DEVICE)
    tgt = torch.randint(0, TEST_CFG.vocab_size, (2, 16), device=DEVICE)
    _, loss, _ = m(idx, targets=tgt)
    loss.backward()

    dead, nan = [], []
    for name, p in m.named_parameters():
        if p.grad is None:
            dead.append(name)
        elif not torch.isfinite(p.grad).all():
            nan.append(name)
        elif p.grad.abs().max().item() == 0.0:
            dead.append(name + "(0)")
    assert not dead, f"기울기가 안 흐르는 파라미터: {dead}"
    assert not nan, f"기울기에 NaN/Inf: {nan}"
    n = sum(1 for _ in m.named_parameters())
    return f"{n}개 파라미터 전부 유한한 0 아닌 기울기"


def c_gqa_variants():
    """GQA 설정을 바꿔도 동작해야 한다 (MHA/GQA/MQA)."""
    outs = []
    for n_kv in (4, 2, 1):
        cfg = ModelConfig(
            vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=n_kv,
            d_ff=128, max_seq_len=32,
        )
        m = Transformer(cfg).to(DEVICE).eval()
        idx = torch.randint(0, 256, (1, 16), device=DEVICE)
        with torch.no_grad():
            logits, _, _ = m(idx)
        assert logits.shape == (1, 16, 256), f"형상 이상: {logits.shape}"
        assert torch.isfinite(logits).all(), "출력에 NaN/Inf"
        outs.append(f"kv={n_kv}")
    return "MHA/GQA/MQA 전부 정상 (" + ", ".join(outs) + ")"


def c_config_validation():
    """말이 안 되는 설정은 생성 시점에 죽어야 한다."""
    bad = 0
    try:
        ModelConfig(d_model=100, n_heads=7)
    except ValueError:
        bad += 1
    try:
        ModelConfig(d_model=64, n_heads=4, n_kv_heads=3)
    except ValueError:
        bad += 1
    assert bad == 2, f"잘못된 설정 {2 - bad}개가 통과했다"
    return "나눠떨어지지 않는 설정 2종 모두 거부"


def c_seq_len_overflow():
    """max_seq_len을 넘기면 조용히 잘리지 말고 죽어야 한다."""
    m = _model()
    idx = torch.randint(0, TEST_CFG.vocab_size, (1, TEST_CFG.max_seq_len + 1), device=DEVICE)
    try:
        with torch.no_grad():
            m(idx)
    except ValueError:
        return f"max_seq_len({TEST_CFG.max_seq_len}) 초과 시 ValueError (의도된 동작)"
    raise AssertionError("길이 초과인데 조용히 통과했다")


def c_generate_works():
    """생성이 실제로 돌고, 어휘 범위 안의 토큰만 나와야 한다."""
    m = _model()
    idx = torch.randint(0, TEST_CFG.vocab_size, (2, 5), device=DEVICE)
    out = m.generate(idx, max_new_tokens=20, temperature=0.8, top_k=10)
    assert out.shape == (2, 25), f"생성 형상 이상: {out.shape}"
    assert (out >= 0).all() and (out < TEST_CFG.vocab_size).all(), "어휘 범위 밖 토큰"
    assert torch.equal(out[:, :5], idx), "프롬프트가 보존되지 않았다"
    greedy1 = m.generate(idx, max_new_tokens=10, temperature=0.0)
    greedy2 = m.generate(idx, max_new_tokens=10, temperature=0.0)
    assert torch.equal(greedy1, greedy2), "greedy 생성이 결정적이지 않다"
    return "생성 정상, greedy 재현성 확인"


def main():
    print("=" * 60)
    print(f"2단계: 트랜스포머 적대적 검증 (device={DEVICE})")
    print("=" * 60)

    check("인과성: 미래 누설 없음", c_causal_no_future_leak)
    check("인과성: 접두사 불변", c_causal_prefix_invariance)
    check("RoPE 상대위치 성질", c_rope_relative_position)
    check("RoPE 길이 보존", c_rope_norm_preserved)
    check("KV 캐시 == 통짜 forward", c_kv_cache_matches_full)
    check("KV 캐시 오용 거부", c_kv_cache_rejects_bad_use)
    check("파라미터 수 손계산 일치", c_param_count)
    check("가중치 공유", c_weight_tying)
    check("초기 손실 = ln(V)", c_init_loss)
    check("기울기 흐름", c_gradients_flow)
    check("GQA 변형 동작", c_gqa_variants)
    check("잘못된 설정 거부", c_config_validation)
    check("길이 초과 거부", c_seq_len_overflow)
    check("생성 동작", c_generate_works)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 2단계 통과 불가")
        return 1
    print("\n판정: 통과 - 다음 단계 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
