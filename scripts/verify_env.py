"""0단계 환경 검증. 통과해야만 다음 단계로 넘어간다.

단순히 cuda.is_available()을 믿지 않는다. 실제로 연산을 돌려서
결과가 CPU와 일치하는지, 고른 정밀도가 fp32보다 실제로 빠른지, 학습에 쓸
메모리가 실제로 확보되는지까지 확인한다.
"""

import sys
import time
import traceback
from pathlib import Path

# Windows 콘솔 기본 코드페이지(cp949)에서 한글/기호가 깨지거나 죽는 것을 막는다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from model.precision import label
from model.precision import resolve as resolve_precision

RESULTS = []


def check(name, fn):
    """검증 하나를 실행하고 결과를 기록한다. 예외는 실패로 처리한다."""
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def c_versions():
    return f"torch {torch.__version__}, cuda build {torch.version.cuda}"


def c_available():
    assert torch.cuda.is_available(), "CUDA를 못 본다"
    return f"device_count={torch.cuda.device_count()}"


def c_device_info():
    p = torch.cuda.get_device_properties(0)
    total_gb = p.total_memory / 1024**3
    assert total_gb > 4.0, f"VRAM이 너무 작다: {total_gb:.1f}GB"
    return f"{p.name}, sm_{p.major}{p.minor}, VRAM {total_gb:.2f}GB"


def c_matmul_correct():
    """GPU 행렬곱이 CPU와 같은 답을 내는가. 드라이버가 깨져 있으면 여기서 걸린다."""
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    cpu = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    diff = (cpu - gpu).abs().max().item()
    assert diff < 1e-3, f"CPU/GPU 결과 불일치: {diff}"
    return f"max_abs_diff={diff:.2e}"


def _tflops(dtype, n=2048, warmup=5, iters=20):
    """행렬곱 실효 성능. 이론값이 아니라 실측이다."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(warmup):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return iters * 2 * n**3 / (time.perf_counter() - t0) / 1e12


def c_precision():
    """고른 정밀도가 fp32보다 실제로 빠른지 잰다.

    is_bf16_supported()를 믿으면 안 된다. V100(sm_70)에서도 True를 반환하는데,
    실측하면 bf16 10.0 / fp32 13.2 / fp16 88.8 TFLOPS로 bf16이 fp32보다도
    느리다(에뮬레이션). 검증은 초록불인데 학습만 8분의 1 속도로 도는 상황을
    여기서 막는다. 지원 여부가 아니라 속도를 판정 근거로 쓴다.
    """
    dtype, why = resolve_precision("cuda")
    fp32 = _tflops(torch.float32)
    chosen = _tflops(dtype) if dtype is not None else fp32

    x = torch.randn(256, 256, device="cuda", dtype=dtype or torch.float32)
    assert torch.isfinite((x @ x).float()).all(), f"{label(dtype)} 연산에 NaN/Inf"

    if dtype is not None:
        assert chosen > fp32 * 1.3, (
            f"{label(dtype)} {chosen:.1f} TFLOPS가 fp32 {fp32:.1f} TFLOPS보다 "
            "빠르지 않다. 텐서코어를 못 쓰고 있다 - 정밀도 선택이 틀렸다."
        )
    bf16_claim = torch.cuda.is_bf16_supported()
    return (
        f"{label(dtype)} {chosen:.1f} vs fp32 {fp32:.1f} TFLOPS "
        f"(is_bf16_supported={bf16_claim}) | {why}"
    )


def c_sdpa():
    """어텐션에 F.scaled_dot_product_attention을 쓴다. GQA 형상으로 확인."""
    B, T, H, KV, D = 2, 128, 10, 2, 64
    # 학습에서 실제로 쓸 dtype으로 확인한다. sm_70에는 flash 백엔드가 없어
    # mem_efficient/math로 내려가는데, enable_gqa를 그 백엔드가 받는지는
    # 실제로 돌려봐야 안다.
    dtype, _ = resolve_precision("cuda")
    dtype = dtype or torch.float32
    q = torch.randn(B, H, T, D, device="cuda", dtype=dtype)
    k = torch.randn(B, KV, T, D, device="cuda", dtype=dtype)
    v = torch.randn(B, KV, T, D, device="cuda", dtype=dtype)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert out.shape == (B, H, T, D), f"형상 이상: {out.shape}"
    assert torch.isfinite(out.float()).all(), "SDPA 결과에 NaN/Inf"
    return f"GQA causal SDPA OK, out={tuple(out.shape)}"


def c_autograd():
    """역전파가 실제로 흐르는가."""
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    w = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (x @ w).pow(2).mean()
    loss.backward()
    assert x.grad is not None and w.grad is not None, "grad가 None"
    assert torch.isfinite(x.grad).all(), "grad에 NaN/Inf"
    return f"grad_norm={w.grad.norm().item():.4f}"


def c_free_memory():
    """실제로 쓸 수 있는 VRAM을 잰다. 다른 프로세스가 점유 중이면 여기서 드러난다."""
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb = free_b / 1024**3
    total_gb = total_b / 1024**3
    if free_gb < 3.0:
        raise AssertionError(
            f"가용 VRAM {free_gb:.2f}GB / {total_gb:.2f}GB. "
            "다른 프로세스가 GPU를 점유 중이다. 정리 후 재실행할 것."
        )
    return f"free={free_gb:.2f}GB / total={total_gb:.2f}GB"


def c_alloc_stress():
    """1GB 텐서를 잡았다 놓아본다. 실제 학습 시 OOM 여유를 본다."""
    n = 256 * 1024 * 1024  # float32 1GiB
    t = torch.empty(n, dtype=torch.float32, device="cuda")
    t.fill_(1.0)
    s = t[::1024].sum().item()
    del t
    torch.cuda.empty_cache()
    assert s > 0, "할당한 메모리에 쓰기 실패"
    return "1GiB 할당/해제 OK"


def main():
    print("=" * 60)
    print("0단계: 학습 환경 검증")
    print("=" * 60)

    check("torch/cuda 버전", c_versions)
    check("CUDA 사용 가능", c_available)
    check("GPU 정보", c_device_info)
    check("GPU 행렬곱 정확도", c_matmul_correct)
    check("정밀도 선택 (실측)", c_precision)
    check("GQA causal SDPA", c_sdpa)
    check("autograd 역전파", c_autograd)
    check("가용 VRAM", c_free_memory)
    check("1GiB 할당 스트레스", c_alloc_stress)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 — 0단계 통과 불가")
        return 1
    print("\n판정: 통과 — 1단계 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
