"""학습에 쓸 혼합정밀도 dtype을 고른다.

`torch.cuda.is_bf16_supported()`를 판단 근거로 쓰면 안 된다. V100(sm_70)에서도
True를 반환한다 — 하드웨어 텐서코어 대신 소프트웨어 에뮬레이션 경로가 있기
때문이다. 그런데 그 에뮬레이션은 fp32보다도 느리다.

  Tesla V100S-PCIE-32GB, 4096x4096 matmul (2026-08-31 실측)
    fp32  13.2 TFLOPS
    fp16  88.8 TFLOPS   <- 텐서코어. fp32의 6.7배
    bf16  10.0 TFLOPS   <- 에뮬레이션. fp32의 0.76배

즉 is_bf16_supported()를 믿으면 환경 검증은 초록불을 켜고 학습만 8분의 1
속도로 도는 상태가 된다. bf16 텐서코어는 Ampere(sm_80)부터이므로 여기서는
compute capability로 판단한다.

fp16을 고르면 GradScaler가 반드시 따라와야 한다. bf16은 지수부가 fp32와 같아
그냥 되지만, fp16은 지수부가 좁아 작은 기울기가 언더플로로 0이 된다.
"""

from __future__ import annotations

import contextlib

import torch

CHOICES = ("auto", "bf16", "fp16", "fp32")

# bf16 텐서코어가 들어온 세대. 이 미만은 에뮬레이션이라 쓰면 안 된다.
BF16_TENSOR_CORE_MAJOR = 8


def resolve(device: str = "cuda", override: str = "auto") -> tuple[torch.dtype | None, str]:
    """(autocast dtype, 선택 근거)를 돌려준다. dtype이 None이면 autocast를 끈다."""
    if override not in CHOICES:
        raise ValueError(f"모르는 정밀도: {override!r} (가능: {CHOICES})")

    if device != "cuda" or not torch.cuda.is_available():
        return None, "CPU — autocast 없음"

    major, minor = torch.cuda.get_device_capability()
    sm = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name()

    if override == "fp32":
        return None, f"명시 지정: fp32 ({name}, {sm})"
    if override == "bf16":
        return torch.bfloat16, f"명시 지정: bf16 ({name}, {sm})"
    if override == "fp16":
        return torch.float16, f"명시 지정: fp16 ({name}, {sm})"

    if major >= BF16_TENSOR_CORE_MAJOR:
        return torch.bfloat16, f"{name} {sm} — bf16 텐서코어 있음"
    return torch.float16, (
        f"{name} {sm} — bf16 텐서코어 없음(Ampere 미만). "
        "bf16 에뮬레이션은 fp32보다도 느려서 fp16을 쓴다"
    )


def needs_scaler(dtype: torch.dtype | None) -> bool:
    """fp16만 GradScaler가 필요하다. bf16은 지수부가 fp32와 같아 안 쓴다."""
    return dtype is torch.float16


def autocast(dtype: torch.dtype | None):
    """dtype이 None이면 아무것도 하지 않는 컨텍스트를 준다.

    `torch.autocast(..., enabled=False)`로 껐다 켰다 하는 것보다 호출부가
    한 줄로 정리되고, dtype=None을 autocast에 넘기는 경로가 아예 없어진다.
    """
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def label(dtype: torch.dtype | None) -> str:
    return {torch.bfloat16: "bf16", torch.float16: "fp16", None: "fp32"}[dtype]
