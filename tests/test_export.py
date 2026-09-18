"""추론용 내보내기를 깨뜨리기 위한 테스트.

여기서 잡아야 하는 것:
  - dtype 변환이 입출력 임베딩의 공유를 끊어 같은 값이 파일에 두 번 들어가는 것
  - 정규화 가중치까지 bf16으로 내려 수치가 불안정해지는 것
  - 내보낸 파일로 로드했을 때 원본과 다른 출력이 나오는 것

공유가 끊기는 버그는 정확도에 영향이 없다. 두 사본의 값이 같고 로드할 때
묶인 파라미터에 둘 다 쓰이기 때문이다. 파일 크기를 재보기 전에는 드러나지
않으므로 여기서 크기와 저장소 개수를 함께 본다.
"""

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.transformer import ModelConfig, Transformer
from scripts.export_weights import cast_state

RESULTS = []

TINY = ModelConfig(
    vocab_size=256, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2,
    d_ff=256, max_seq_len=64,
)


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def _storage_groups(state):
    g = {}
    for k, v in state.items():
        g.setdefault(v.untyped_storage().data_ptr(), []).append(k)
    return g


def _model():
    torch.manual_seed(0)
    return Transformer(TINY)


def c_tying_survives_cast():
    """bf16으로 내려도 tok_emb와 lm_head가 같은 텐서를 가리켜야 한다."""
    sd = _model().state_dict()
    before = [ks for ks in _storage_groups(sd).values() if len(ks) > 1]
    assert before, "원본에서 이미 공유가 없다. 테스트 전제가 깨졌다."

    half = cast_state(sd, half=True)
    after = [ks for ks in _storage_groups(half).values() if len(ks) > 1]
    assert sorted(map(sorted, after)) == sorted(map(sorted, before)), (
        f"변환 후 공유가 달라졌다: {before} -> {after}. "
        "키마다 .to()를 부르면 새 저장소가 생겨 임베딩이 두 번 저장된다."
    )
    assert half["tok_emb.weight"] is half["lm_head.weight"], "같은 객체가 아니다"
    return f"공유 {before} 유지, 저장소 {len(_storage_groups(half))}개"


def c_only_2d_is_cast():
    """정규화 가중치(1차원)는 fp32로 남아야 한다."""
    sd = _model().state_dict()
    half = cast_state(sd, half=True)
    bad_1d = [k for k, v in half.items() if v.dim() < 2 and v.dtype is not torch.float32]
    bad_2d = [k for k, v in half.items() if v.dim() >= 2 and v.dtype is not torch.bfloat16]
    assert not bad_1d, f"1차원인데 bf16으로 내려갔다: {bad_1d}"
    assert not bad_2d, f"2차원인데 안 내려갔다: {bad_2d}"
    n1 = sum(1 for v in half.values() if v.dim() < 2)
    return f"1차원 {n1}개 fp32 유지, 나머지 {len(half) - n1}개 bf16"


def c_half_is_not_passthrough():
    """half=False면 원본을 그대로 돌려줘야 한다(불필요한 복사 금지)."""
    sd = _model().state_dict()
    assert cast_state(sd, half=False) is sd, "half=False인데 새 dict를 만들었다"
    return "half=False는 원본 그대로"


def c_file_size_halves():
    """실제 파일 크기가 절반 근처여야 한다. 공유가 끊기면 여기서 걸린다."""
    sd = _model().state_dict()
    with tempfile.TemporaryDirectory() as d:
        f32 = Path(d) / "f32.pt"
        f16 = Path(d) / "bf16.pt"
        torch.save({"model": cast_state(sd, half=False)}, f32)
        torch.save({"model": cast_state(sd, half=True)}, f16)
        a, b = f32.stat().st_size, f16.stat().st_size
    ratio = b / a
    # 1차원만 fp32로 남으므로 0.50을 약간 넘는 선이 정상이다(실측 0.502).
    # 공유가 끊기면 임베딩이 중복돼 올라간다: 이 소형 모델 0.537, 53M 0.60.
    # 소형 모델은 임베딩 비중이 7%뿐이라 임계값을 넉넉히 두면 못 잡는다.
    assert ratio < 0.51, (
        f"bf16 파일이 fp32의 {ratio:.3f}배다. 0.50 근처여야 한다 - "
        "임베딩이 중복 저장되고 있을 가능성이 높다."
    )
    return f"{a:,} -> {b:,} 바이트 ({ratio:.3f}배)"


def c_round_trip_outputs_match():
    """내보낸 가중치를 로드해서 원본과 같은 출력이 나오는가."""
    model = _model().eval()
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    with torch.no_grad():
        ref, _, _ = model(idx)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "w.pt"
        torch.save({"model": cast_state(model.state_dict(), half=True)}, path)
        loaded = Transformer(TINY)
        loaded.load_state_dict(torch.load(path, map_location="cpu")["model"])
    loaded.eval()
    with torch.no_grad():
        got, _, _ = loaded(idx)

    diff = (ref - got).abs().max().item()
    scale = ref.abs().max().item()
    # bf16은 가수 8비트라 상대오차 0.4% 안팎이 층마다 누적된다
    assert diff / scale < 0.05, f"출력 차이가 크다: {diff:.4e} (스케일 {scale:.4e})"
    return f"최대차 {diff:.2e} / 스케일 {scale:.2e} = {diff / scale:.2%}"


def main():
    print("=" * 60)
    print("추론용 내보내기 적대적 검증")
    print("=" * 60)

    check("bf16 변환 후 가중치 공유 유지", c_tying_survives_cast)
    check("2차원만 bf16, 정규화는 fp32", c_only_2d_is_cast)
    check("half=False는 원본 그대로", c_half_is_not_passthrough)
    check("파일 크기가 절반 근처", c_file_size_halves)
    check("내보내기 왕복 출력 일치", c_round_trip_outputs_match)

    n_pass = sum(1 for ok, _, _ in RESULTS if ok)
    print("=" * 60)
    print(f"결과: {n_pass}/{len(RESULTS)} 통과")
    print()
    if n_pass == len(RESULTS):
        print("판정: 통과 - 내보낸 파일을 그대로 써도 된다")
        sys.exit(0)
    print("판정: 위험 - 아래 항목이 깨져 있다")
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  - {name}: {detail}")
    sys.exit(1)


if __name__ == "__main__":
    main()
