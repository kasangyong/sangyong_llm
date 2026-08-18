"""학습 루프를 깨뜨리기 위한 테스트.

여기서 잡아야 하는 버그들:
  - 기울기 누적을 잘못 나눠서 유효 학습률이 달라지는 것
  - 체크포인트에 옵티마이저 상태를 빠뜨려 재개 후 궤적이 튀는 것
  - 데이터 로더가 x/y 정렬을 틀려서 모델이 자기 입력을 베끼는 것
  - LR 스케줄이 워밍업/감쇠를 잘못 계산하는 것

단일 배치 과적합 테스트가 핵심이다. 모델+옵티마이저+손실이 실제로
연결돼 있다면 손실이 0으로 내려가야 한다. 안 내려가면 배선이 끊긴 것이다.
"""

import math
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from model.transformer import ModelConfig, Transformer
from train.train import (
    BinDataset,
    TrainConfig,
    load_checkpoint,
    lr_at,
    make_optimizer,
    save_checkpoint,
)

RESULTS = []
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


# --------------------------------------------------------------- LR 스케줄

def c_lr_schedule():
    cfg = TrainConfig(lr=1e-3, warmup_iters=100, max_iters=1000, min_lr_frac=0.1)
    assert abs(lr_at(0, cfg) - 1e-5) < 1e-9, f"첫 스텝 lr 이상: {lr_at(0, cfg)}"
    assert abs(lr_at(99, cfg) - 1e-3) < 1e-9, f"워밍업 끝 lr 이상: {lr_at(99, cfg)}"
    # 워밍업 구간은 단조 증가
    warm = [lr_at(i, cfg) for i in range(100)]
    assert all(b > a for a, b in zip(warm, warm[1:])), "워밍업이 단조 증가가 아니다"
    # 감쇠 구간은 단조 감소
    decay = [lr_at(i, cfg) for i in range(100, 1000, 50)]
    assert all(b < a for a, b in zip(decay, decay[1:])), "감쇠가 단조 감소가 아니다"
    # 끝에서 min_lr에 도달
    end = lr_at(1000, cfg)
    assert abs(end - 1e-4) < 1e-9, f"최종 lr 이상: {end}"
    return f"워밍업 1e-05->1e-03, 감쇠 후 {end:.1e} (min_lr_frac 준수)"


# --------------------------------------------------------------- 기울기 누적

def c_grad_accum_equivalence():
    """누적 4스텝 x 배치2 == 통짜 배치8. 안 맞으면 유효 학습률이 달라진다."""
    torch.manual_seed(11)
    model = Transformer(TINY).to(DEVICE)
    torch.manual_seed(12)
    x = torch.randint(0, TINY.vocab_size, (8, 32), device=DEVICE)
    y = torch.randint(0, TINY.vocab_size, (8, 32), device=DEVICE)

    # 통짜
    model.zero_grad(set_to_none=True)
    _, loss, _ = model(x, targets=y)
    loss.backward()
    full = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    full_loss = loss.item()

    # 누적 (배치 2씩 4번, accum으로 나눔)
    model.zero_grad(set_to_none=True)
    accum = 4
    acc_loss = 0.0
    for i in range(accum):
        xb = x[i * 2 : (i + 1) * 2]
        yb = y[i * 2 : (i + 1) * 2]
        _, l, _ = model(xb, targets=yb)
        (l / accum).backward()
        acc_loss += l.item() / accum
    accd = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]

    gdiff = max((a - b).abs().max().item() for a, b in zip(full, accd))
    ldiff = abs(full_loss - acc_loss)
    assert gdiff < 1e-4, f"기울기 불일치: {gdiff:.3e}"
    assert ldiff < 1e-4, f"손실 불일치: {ldiff:.3e}"
    return f"기울기 최대차 {gdiff:.2e}, 손실차 {ldiff:.2e}"


# --------------------------------------------------------------- 과적합

def c_overfit_single_batch():
    """배치 하나를 외우게 만든다. 손실이 안 떨어지면 어딘가 끊긴 것이다."""
    torch.manual_seed(13)
    model = Transformer(TINY).to(DEVICE)
    cfg = TrainConfig(lr=3e-3, weight_decay=0.0)
    opt = make_optimizer(model, cfg)

    torch.manual_seed(14)
    x = torch.randint(0, TINY.vocab_size, (4, 32), device=DEVICE)
    y = torch.randint(0, TINY.vocab_size, (4, 32), device=DEVICE)

    model.train()
    first = None
    for step in range(400):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = loss.item()
    last = loss.item()

    assert last < 0.05, f"과적합 실패: {first:.3f} -> {last:.3f} (0.05 미만이어야 함)"
    return f"loss {first:.3f} -> {last:.5f} (400스텝)"


# --------------------------------------------------------------- 체크포인트

def c_checkpoint_roundtrip():
    """저장/불러오기 후 가중치와 옵티마이저 상태가 완전히 같아야 한다."""
    torch.manual_seed(15)
    model = Transformer(TINY).to(DEVICE)
    cfg = TrainConfig()
    opt = make_optimizer(model, cfg)

    # 옵티마이저 상태(모멘텀)를 실제로 만들어 놓는다
    x = torch.randint(0, TINY.vocab_size, (2, 32), device=DEVICE)
    y = torch.randint(0, TINY.vocab_size, (2, 32), device=DEVICE)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ck.pt"
        save_checkpoint(p, model, opt, TINY, cfg, it=42, best_val=1.23)
        model2, ck = load_checkpoint(p, DEVICE)
        opt2 = make_optimizer(model2, cfg)
        opt2.load_state_dict(ck["optimizer"])

        assert ck["iter"] == 42, f"iter 복원 실패: {ck['iter']}"
        assert abs(ck["best_val"] - 1.23) < 1e-9, "best_val 복원 실패"

        wdiff = max(
            (a - b).abs().max().item()
            for a, b in zip(model.state_dict().values(), model2.state_dict().values())
        )
        assert wdiff == 0.0, f"가중치 불일치: {wdiff}"

        # 옵티마이저 exp_avg(1차 모멘텀)가 실제로 복원됐는지
        s1 = list(opt.state.values())[0]
        s2 = list(opt2.state.values())[0]
        mdiff = (s1["exp_avg"] - s2["exp_avg"]).abs().max().item()
        assert mdiff == 0.0, f"옵티마이저 모멘텀 불일치: {mdiff}"
        assert int(s1["step"]) == int(s2["step"]) == 3, "옵티마이저 step 수 불일치"

    return "가중치/모멘텀/step/iter 전부 정확히 복원"


def c_resume_trajectory():
    """중단 후 재개한 궤적이 끊김 없이 이어진 궤적과 같아야 한다.

    옵티마이저 상태를 빠뜨리는 버그는 이 테스트만이 잡는다.
    """
    def run(steps, ckpt_at=None, resume_from=None, tmpdir=None):
        torch.manual_seed(16)
        if resume_from is None:
            model = Transformer(TINY).to(DEVICE)
            cfg = TrainConfig(lr=1e-3, weight_decay=0.0)
            opt = make_optimizer(model, cfg)
            start = 0
        else:
            model, ck = load_checkpoint(resume_from, DEVICE)
            cfg = TrainConfig(lr=1e-3, weight_decay=0.0)
            opt = make_optimizer(model, cfg)
            opt.load_state_dict(ck["optimizer"])
            start = ck["iter"] + 1

        g = torch.Generator().manual_seed(99)
        losses = []
        for it in range(steps):
            # 데이터는 스텝 번호로 결정론적으로 만든다
            gg = torch.Generator(device="cpu").manual_seed(1000 + it)
            x = torch.randint(0, TINY.vocab_size, (2, 32), generator=gg).to(DEVICE)
            y = torch.randint(0, TINY.vocab_size, (2, 32), generator=gg).to(DEVICE)
            if it < start:
                continue
            opt.zero_grad(set_to_none=True)
            _, loss, _ = model(x, targets=y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if ckpt_at is not None and it == ckpt_at:
                save_checkpoint(tmpdir / "mid.pt", model, opt, TINY, cfg, it, 0.0)
        return losses

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        straight = run(20)
        _ = run(20, ckpt_at=9, tmpdir=tmp)
        resumed = run(20, resume_from=tmp / "mid.pt", tmpdir=tmp)

    tail = straight[10:]
    assert len(tail) == len(resumed), f"길이 불일치: {len(tail)} vs {len(resumed)}"
    worst = max(abs(a - b) for a, b in zip(tail, resumed))
    assert worst < 1e-4, f"재개 궤적이 어긋남: 최대차 {worst:.3e}"
    return f"10스텝에서 중단/재개 후 남은 10스텝 손실 최대차 {worst:.2e}"


# --------------------------------------------------------------- 옵티마이저 그룹

def c_weight_decay_groups():
    """decay는 2차원 이상에만 걸려야 한다 (정규화/편향 제외)."""
    model = Transformer(TINY).to(DEVICE)
    opt = make_optimizer(model, TrainConfig(weight_decay=0.1))
    g_decay, g_nodecay = opt.param_groups
    assert g_decay["weight_decay"] == 0.1, "decay 그룹 설정 이상"
    assert g_nodecay["weight_decay"] == 0.0, "no-decay 그룹 설정 이상"
    assert all(p.dim() >= 2 for p in g_decay["params"]), "decay 그룹에 1차원 파라미터"
    assert all(p.dim() < 2 for p in g_nodecay["params"]), "no-decay 그룹에 2차원 파라미터"
    # 가중치 공유 때문에 파라미터를 두 번 세면 안 된다
    total = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert total == model.num_params(), f"옵티마이저 파라미터 수 불일치: {total:,}"
    return (
        f"decay {len(g_decay['params'])}개 / no-decay {len(g_nodecay['params'])}개, "
        f"중복 없음 ({total:,})"
    )


# --------------------------------------------------------------- 데이터 로더

def c_bin_dataset():
    """x와 y가 정확히 한 칸 어긋나야 한다. 여기가 틀리면 모델은 복사만 배운다."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.bin"
        arr = np.arange(5000, dtype=np.uint16)
        p.write_bytes(arr.tobytes())

        with BinDataset(p, block_size=64) as ds:
            assert len(ds) == 5000, f"길이 이상: {len(ds)}"
            x, y = ds.batch(8, "cpu")
            assert x.shape == (8, 64) and y.shape == (8, 64), f"형상 이상: {x.shape} {y.shape}"
            # y는 x를 한 칸 민 것
            assert torch.equal(y[:, :-1], x[:, 1:]), "y가 x보다 한 칸 앞이 아니다"
            # 데이터가 arange라서 값이 연속이어야 한다
            assert torch.equal(x[0], torch.arange(x[0, 0], x[0, 0] + 64)), "구간이 연속이 아니다"
            assert x.dtype == torch.int64, f"dtype 이상: {x.dtype}"

        # close() 후에는 파일 잠금이 풀려 삭제/교체가 가능해야 한다 (Windows)
        p.unlink()
        assert not p.exists(), "close() 후에도 파일이 잠겨 있다"
        p.write_bytes(arr.tobytes())

        # 너무 짧은 파일은 거부해야 한다
        q = Path(d) / "s.bin"
        q.write_bytes(np.arange(10, dtype=np.uint16).tobytes())
        try:
            BinDataset(q, block_size=64)
        except ValueError:
            pass
        else:
            raise AssertionError("토큰이 부족한데 조용히 통과했다")
    return "x/y 한 칸 정렬 확인, 짧은 파일 거부"


def c_bf16_training_step():
    """실제 학습에 쓰는 bf16 autocast 경로가 도는가."""
    if DEVICE != "cuda":
        raise AssertionError("CUDA가 없어 bf16 경로를 검증할 수 없다")
    torch.manual_seed(17)
    model = Transformer(TINY).to(DEVICE)
    opt = make_optimizer(model, TrainConfig(lr=1e-3))
    x = torch.randint(0, TINY.vocab_size, (4, 32), device=DEVICE)
    y = torch.randint(0, TINY.vocab_size, (4, 32), device=DEVICE)
    losses = []
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss, _ = model(x, targets=y)
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ), "bf16 경로에서 기울기에 NaN/Inf"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"bf16으로 손실이 안 줄었다: {losses[0]:.3f} -> {losses[-1]:.3f}"
    return f"bf16 30스텝, loss {losses[0]:.3f} -> {losses[-1]:.3f}, NaN 없음"


def main():
    print("=" * 60)
    print(f"3단계: 학습 루프 적대적 검증 (device={DEVICE})")
    print("=" * 60)

    check("LR 스케줄", c_lr_schedule)
    check("기울기 누적 == 통짜 배치", c_grad_accum_equivalence)
    check("단일 배치 과적합", c_overfit_single_batch)
    check("체크포인트 왕복", c_checkpoint_roundtrip)
    check("중단/재개 궤적 일치", c_resume_trajectory)
    check("weight decay 그룹 분리", c_weight_decay_groups)
    check("데이터 로더 x/y 정렬", c_bin_dataset)
    check("bf16 학습 경로", c_bf16_training_step)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - 3단계 통과 불가")
        return 1
    print("\n판정: 통과 - 본 학습 진행 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
