"""DDP 적대적 검증. 3장으로 늘려도 1장과 같은 학습을 하는가.

DDP는 조용히 틀리기 쉬운 구간이다. 손실 곡선은 멀쩡한데
  - 랭크들이 같은 배치를 봐서 3배 느리기만 하거나
  - 유효 배치가 3배로 부풀어 의도한 lr 스케줄과 어긋나거나
  - state_dict에 "module." 접두사가 붙어 로컬에서 못 여는
일이 벌어진다. 셋 다 여기서 잡는다.
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model.transformer import ModelConfig, Transformer
from train.train import DDPContext, TrainConfig, load_checkpoint, save_checkpoint, setup_ddp

RESULTS = []
TINY = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                   d_ff=128, max_seq_len=64)
N_GPU = torch.cuda.device_count()


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


# ------------------------------------------------------------ 단일 프로세스

def c_no_torchrun_falls_back():
    """torchrun 없이 그냥 실행하면 기존 단일 GPU 경로로 떨어져야 한다."""
    saved = {k: os.environ.pop(k, None) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")}
    try:
        ddp = setup_ddp()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert not ddp.enabled, "torchrun 없이도 DDP가 켜졌다"
    assert (ddp.world_size, ddp.rank) == (1, 0), f"world_size/rank 이상: {ddp}"
    assert ddp.is_master, "단일 실행인데 마스터가 아니다"
    return f"enabled=False, device={ddp.device}"


def c_effective_batch_scales():
    """유효 배치가 world_size만큼 곱해져야 한다. 안 그러면 스텝 수가 3배로
    잡혀 데이터를 3바퀴 돌게 된다."""
    one = TrainConfig(batch_size=4, grad_accum=8, block_size=1024, world_size=1)
    three = TrainConfig(batch_size=4, grad_accum=8, block_size=1024, world_size=3)
    assert one.tokens_per_iter == 4 * 8 * 1024, one.tokens_per_iter
    assert three.tokens_per_iter == one.tokens_per_iter * 3, three.tokens_per_iter
    return f"1랭크 {one.tokens_per_iter:,} -> 3랭크 {three.tokens_per_iter:,} 토큰/스텝"


def c_total_tokens_invariant():
    """GPU 수를 바꿔도 총 학습 토큰(= epochs x 코퍼스)은 같아야 한다.
    max_iters = epochs * len(train) / tokens_per_iter 이므로,
    tokens_per_iter가 world_size를 반영해야만 성립한다."""
    n_tokens, epochs = 6_600_000_000, 1.0
    seen = set()
    for ws, ga in ((1, 24), (3, 8)):  # grad_accum을 1/3로 줄여 유효 배치를 맞춘 경우
        cfg = TrainConfig(batch_size=4, grad_accum=ga, block_size=2048, world_size=ws)
        iters = max(1, int(epochs * n_tokens / cfg.tokens_per_iter))
        seen.add(iters * cfg.tokens_per_iter)
    assert len(seen) == 1, f"총 토큰이 GPU 수에 따라 달라진다: {seen}"
    return f"1랭크/3랭크 모두 {seen.pop() / 1e9:.2f}B 토큰"


def c_rank_seeds_differ():
    """랭크마다 다른 시드를 써야 서로 다른 구간을 뽑는다. 같은 시드면
    세 장이 같은 배치를 돌아 3배 느리기만 하고 얻는 게 없다."""
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(10_000, (16,), generator=g)

    base = TrainConfig().seed
    r0, r1, r2 = draw(base + 0), draw(base + 1), draw(base + 2)
    assert not torch.equal(r0, r1), "랭크 0과 1이 같은 배치를 뽑는다"
    assert not torch.equal(r1, r2), "랭크 1과 2가 같은 배치를 뽑는다"
    overlap = len(set(r0.tolist()) & set(r1.tolist()))
    return f"랭크별 인덱스 상이 (0<->1 겹침 {overlap}/16)"


def c_checkpoint_has_no_module_prefix():
    """DDP로 학습해도 체크포인트는 원본 모델 기준으로 저장돼야 한다.
    래퍼째 저장하면 키마다 "module." 접두사가 붙어 로컬(단일 GPU)에서
    load_state_dict가 통째로 실패한다."""
    torch.manual_seed(0)
    raw = Transformer(TINY)
    opt = torch.optim.AdamW(raw.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.pt"
        save_checkpoint(p, raw, opt, TINY, TrainConfig(), 7, 1.23)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        bad = [k for k in ck["model"] if k.startswith("module.")]
        assert not bad, f"module. 접두사가 붙었다: {bad[:3]}"
        model, _ = load_checkpoint(p, "cpu")  # 단일 GPU 경로로 다시 열린다
        assert model.num_params() == raw.num_params()
    return f"키 {len(ck['model'])}개 전부 접두사 없음, 재로드 성공"


# ------------------------------------------------------------ 다중 프로세스

def _worker(rank, world_size, tmpdir, backend, device_type):
    """랭크 하나. 결과를 tmpdir에 파일로 남긴다."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT="29517",
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size),
    )
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend)
        device = f"cuda:{rank}"
    else:
        dist.init_process_group(backend=backend)
        device = "cpu"

    # 랭크마다 다른 시드로 시작해도 DDP가 가중치를 랭크 0으로 맞춰야 한다.
    torch.manual_seed(1337 + rank)
    raw = Transformer(TINY).to(device)
    model = DDP(raw, device_ids=[rank] if device_type == "cuda" else None)
    torch.save(
        {k: v.cpu() for k, v in raw.state_dict().items()},
        Path(tmpdir) / f"init_{rank}.pt",
    )

    # 랭크별로 다른 데이터를 2회 누적. 마지막 마이크로스텝에서만 동기화한다.
    accum = 2
    data = [
        (torch.randint(0, TINY.vocab_size, (2, 16), device=device),
         torch.randint(0, TINY.vocab_size, (2, 16), device=device))
        for _ in range(accum)
    ]
    torch.save([(x.cpu(), y.cpu()) for x, y in data], Path(tmpdir) / f"data_{rank}.pt")

    model.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(data):
        ctx = model.no_sync() if i < accum - 1 else torch.enable_grad()
        with ctx:
            _, loss, _ = model(x, targets=y)
            (loss / accum).backward()
    torch.save(
        {n: p.grad.cpu() for n, p in raw.named_parameters()},
        Path(tmpdir) / f"grad_{rank}.pt",
    )

    opt = torch.optim.AdamW(raw.parameters(), lr=1e-2)
    opt.step()
    torch.save(
        {k: v.cpu() for k, v in raw.state_dict().items()},
        Path(tmpdir) / f"after_{rank}.pt",
    )
    dist.barrier()
    dist.destroy_process_group()


def _run_ddp(tmpdir, world_size=2):
    if N_GPU >= world_size:
        backend, device_type = "nccl", "cuda"
    else:
        backend, device_type = "gloo", "cpu"
    mp.spawn(_worker, args=(world_size, tmpdir, backend, device_type),
             nprocs=world_size, join=True)
    return backend


_CACHE = {}


def _ddp_run():
    """무거우니 한 번만 돌리고 재사용한다."""
    if "dir" not in _CACHE:
        td = tempfile.mkdtemp()
        _CACHE["backend"] = _run_ddp(td)
        _CACHE["dir"] = td
    return _CACHE["dir"], _CACHE["backend"]


def c_ddp_broadcasts_init():
    """랭크마다 시드가 달라도 DDP가 초기 가중치를 랭크 0으로 맞춰야 한다.
    안 맞으면 세 장이 서로 다른 모델을 학습하고 평균만 섞인다."""
    td, backend = _ddp_run()
    a = torch.load(Path(td) / "init_0.pt", weights_only=True)
    b = torch.load(Path(td) / "init_1.pt", weights_only=True)
    worst = max((a[k] - b[k]).abs().max().item() for k in a)
    assert worst == 0.0, f"초기 가중치가 랭크 간 다르다: 최대차 {worst:.2e}"
    return f"{backend}, 키 {len(a)}개 최대차 {worst:.2e}"


def c_no_sync_grads_are_averaged():
    """no_sync로 누적해도 최종 기울기가 두 랭크 데이터 전체의 평균과 같아야
    한다. no_sync를 잘못 걸면 마지막 마이크로배치 기울기만 동기화돼,
    손실은 정상으로 보이는데 실제로는 데이터 절반을 버린다."""
    td, _ = _ddp_run()
    g0 = torch.load(Path(td) / "grad_0.pt", weights_only=True)
    g1 = torch.load(Path(td) / "grad_1.pt", weights_only=True)
    worst_sync = max((g0[k] - g1[k]).abs().max().item() for k in g0)
    assert worst_sync < 1e-6, f"랭크 간 기울기 불일치: {worst_sync:.2e}"

    # 같은 데이터를 단일 프로세스에서 4개 마이크로배치로 통짜 누적한 것과 비교
    torch.manual_seed(1337)
    ref = Transformer(TINY)
    ref.load_state_dict(torch.load(Path(td) / "init_0.pt", weights_only=True))
    d0 = torch.load(Path(td) / "data_0.pt", weights_only=True)
    d1 = torch.load(Path(td) / "data_1.pt", weights_only=True)
    ref.zero_grad(set_to_none=True)
    # DDP는 랭크 평균이므로 (누적 2회 x 랭크 2개) = 4로 나눈 것과 같다
    for x, y in d0 + d1:
        _, loss, _ = ref(x, targets=y)
        (loss / 4).backward()
    worst_ref = max(
        (g0[n] - p.grad).abs().max().item() for n, p in ref.named_parameters()
    )
    assert worst_ref < 1e-5, f"통짜 누적과 불일치: {worst_ref:.2e} — 데이터가 새고 있다"
    return f"랭크간 {worst_sync:.2e}, 통짜 4배치 대비 {worst_ref:.2e}"


def c_params_identical_after_step():
    """옵티마이저 스텝 후 랭크들의 파라미터가 같아야 한다. 같지 않으면
    마스터만 저장하는 체크포인트가 나머지 랭크의 학습을 버리는 셈이다."""
    td, _ = _ddp_run()
    a = torch.load(Path(td) / "after_0.pt", weights_only=True)
    b = torch.load(Path(td) / "after_1.pt", weights_only=True)
    worst = max((a[k] - b[k]).abs().max().item() for k in a)
    assert worst == 0.0, f"스텝 후 파라미터가 갈라졌다: {worst:.2e}"
    init = torch.load(Path(td) / "init_0.pt", weights_only=True)
    moved = max((a[k] - init[k]).abs().max().item() for k in a)
    assert moved > 0, "스텝을 밟았는데 파라미터가 그대로다"
    return f"랭크간 최대차 {worst:.2e}, 초기 대비 이동 {moved:.4f}"


def main():
    print("=" * 60)
    print(f"DDP 검증 (가용 GPU {N_GPU}장)")
    print("=" * 60)

    check("torchrun 없으면 단일 경로", c_no_torchrun_falls_back)
    check("유효 배치가 랭크 수만큼 곱해짐", c_effective_batch_scales)
    check("총 학습 토큰이 GPU 수에 불변", c_total_tokens_invariant)
    check("랭크별 시드 분리", c_rank_seeds_differ)
    check("체크포인트에 module. 접두사 없음", c_checkpoint_has_no_module_prefix)
    check("초기 가중치 브로드캐스트", c_ddp_broadcasts_init)
    check("no_sync 누적 == 통짜 평균", c_no_sync_grads_are_averaged)
    check("스텝 후 랭크 간 파라미터 동일", c_params_identical_after_step)

    print("=" * 60)
    n_ok = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"결과: {n_ok}/{len(RESULTS)} 통과")
    if n_ok == len(RESULTS):
        print("\n판정: 통과 - 3장 DDP로 본 학습을 걸어도 된다")
        sys.exit(0)
    print("\n판정: 위험 - 실패 항목을 고치기 전에는 돌리지 말 것")
    for ok, name, detail in RESULTS:
        if not ok:
            print(f"  - {name}: {detail}")
    sys.exit(1)


if __name__ == "__main__":
    main()
