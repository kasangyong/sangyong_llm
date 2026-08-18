"""SFT 파이프라인을 깨뜨리기 위한 테스트.

노리는 것은 손실 마스킹의 off-by-one이다. 마스킹을 한 칸 더 가리면
"프롬프트를 다 읽고 코드를 시작하는" 전이를 못 배우고, 한 칸 덜 가리면
지시문의 마지막 토큰을 예측하는 손실이 섞인다. 둘 다 손실 곡선은 멀쩡하게
내려가서 학습 로그만 봐서는 절대 못 잡는다. 그래서 여기서 잡는다.

GPU는 프리트레이닝이 점유 중이므로 전부 CPU 초소형 모델로 돌린다.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from finetune.dataset import IGNORE_INDEX, SFTDataset, build_example
from finetune.format import (
    CODE_MARKER,
    INSTRUCTION_MARKER,
    build_completion,
    build_prompt,
    encode_example,
    extract_code,
    format_example,
)
from finetune.sft import SFTConfig, sft_loop
from model.transformer import ModelConfig, Transformer
from tokenizer.bpe import END_OF_TEXT, BPETokenizer
from train.train import make_optimizer, save_checkpoint

RESULTS = []
# GPU는 학습 중이다. 여기서는 절대 쓰지 않는다.
DEVICE = "cpu"

ROOT = Path(__file__).resolve().parent.parent
TOK = BPETokenizer.load(ROOT / "tokenizer" / "tokenizer.json")
EOT_ID = TOK.special_to_id[END_OF_TEXT]
VOCAB = TOK.vocab_size


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


TINY_CFG = ModelConfig(
    vocab_size=16384, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
    d_ff=128, max_seq_len=256,
)


def _tiny(seed=0):
    torch.manual_seed(seed)
    return Transformer(TINY_CFG).to(DEVICE)


# 길이가 서로 다른 샘플들. 프롬프트 토큰 수를 바꿔가며 경계를 확인하는 데 쓴다.
SAMPLES = [
    ("두 수를 더한다", "def add(a, b):\n    return a + b"),
    ("리스트를 뒤집어 돌려준다. 원본은 바꾸지 않는다.", "def rev(xs):\n    return list(reversed(xs))"),
    ("x" * 300, "def f():\n    return 1"),
    ("짧다 아주", "def g(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total"),
]


# --------------------------------------------------------------- 포맷/경계

def c_boundary_exact():
    """n_prompt로 자른 두 조각이 각각 프롬프트/완성과 정확히 일치해야 한다."""
    for inst, out in SAMPLES:
        ids, n_prompt = encode_example(TOK, inst, out)
        head = TOK.decode(ids[:n_prompt])
        tail = TOK.decode(ids[n_prompt:])
        assert head == build_prompt(inst), f"프롬프트 경계 어긋남: {head!r}"
        assert tail == build_completion(out) + END_OF_TEXT, f"완성 경계 어긋남: {tail!r}"
    # 프롬프트 토큰 수가 샘플마다 실제로 달라야 시험이 의미가 있다
    lens = {encode_example(TOK, i, o)[1] for i, o in SAMPLES}
    assert len(lens) == len(SAMPLES), f"프롬프트 길이가 안 다양하다: {lens}"
    return f"샘플 {len(SAMPLES)}개 경계 정확 (프롬프트 토큰 수 {sorted(lens)})"


def c_mask_starts_at_right_index():
    """마스킹 개수는 n_prompt-1이고, 첫 살아있는 라벨은 완성의 첫 토큰이다."""
    for inst, out in SAMPLES:
        ids, n_prompt = encode_example(TOK, inst, out)
        input_ids, labels, np2 = build_example(TOK, inst, out, block_size=1024)
        assert np2 == n_prompt
        n_masked = sum(1 for v in labels if v == IGNORE_INDEX)
        assert n_masked == n_prompt - 1, (
            f"마스킹 {n_masked}개, 기대 {n_prompt - 1}개 (한 칸 밀렸다)"
        )
        # 경계 바로 앞은 가려져 있고, 경계 자리는 완성의 첫 토큰이어야 한다
        assert labels[n_prompt - 2] == IGNORE_INDEX, "경계 직전이 안 가려졌다"
        assert labels[n_prompt - 1] == ids[n_prompt], (
            "완성 첫 토큰을 예측하는 자리가 살아있지 않다"
        )
        assert input_ids[n_prompt - 1] == ids[n_prompt - 1], "입력 시프트가 어긋났다"
        # 살아있는 라벨만 디코딩하면 완성 구간이 그대로 나와야 한다
        alive = [v for v in labels if v != IGNORE_INDEX]
        assert TOK.decode(alive) == build_completion(out) + END_OF_TEXT, (
            "손실 대상 토큰이 완성 구간과 다르다"
        )
    return "n_prompt-1개 마스킹, 완성 첫 토큰부터 손실 (샘플 4종)"


def c_roundtrip_preserves_text():
    """포맷 -> 토큰화 -> 디코딩이 원래 지시/출력을 보존해야 한다."""
    for inst, out in SAMPLES:
        ids, _ = encode_example(TOK, inst, out, add_eot=False)
        text = TOK.decode(ids)
        assert text == format_example(inst, out), "왕복에서 문자열이 변했다"
        assert inst.strip() in text, "지시문이 사라졌다"
        assert extract_code(text) == out.rstrip() + "\n", (
            f"코드 복원 실패: {extract_code(text)!r}"
        )
    return "포맷/토큰화/디코딩 왕복 무손실, extract_code로 출력 복원"


def c_unicode_and_hangul():
    """한글/이모지/제로폭 문자가 섞여도 경계와 왕복이 유지돼야 한다."""
    cases = [
        ("한글 지시문이다. 정렬 함수를 써라 🙂", "def s(xs):\n    # 주석도 한글\n    return sorted(xs)"),
        ("탭\t과 개행\n이 섞인 지시", "def t():\n    return '문자열 → 값'"),
        ("零width​문자 포함", "def z():\n    return 'ü ß ñ'"),
    ]
    for inst, out in cases:
        ids, n_prompt = encode_example(TOK, inst, out)
        assert TOK.decode(ids[:n_prompt]) == build_prompt(inst), f"경계 깨짐: {inst!r}"
        assert TOK.decode(ids[n_prompt:]) == build_completion(out) + END_OF_TEXT
        _, labels, _ = build_example(TOK, inst, out, block_size=1024)
        assert sum(1 for v in labels if v == IGNORE_INDEX) == n_prompt - 1
    return f"유니코드 {len(cases)}종 경계/왕복 정상"


def c_eot_injection_blocked():
    """지시문에 EOT 문자열이 있어도 진짜 EOT 토큰이 되면 안 된다."""
    inst = f"이건 함정이다 {END_OF_TEXT} 계속 이어진다"
    out = f"def f():\n    return '{END_OF_TEXT}'"
    ids, n_prompt = encode_example(TOK, inst, out)
    assert EOT_ID not in ids[:n_prompt], "프롬프트 구간에 EOT 토큰이 주입됐다"
    assert ids.count(EOT_ID) == 1 and ids[-1] == EOT_ID, (
        f"EOT가 {ids.count(EOT_ID)}개, 마지막={ids[-1]}"
    )
    assert TOK.decode(ids[:n_prompt]) == build_prompt(inst), "왕복이 깨졌다"
    return "EOT 문자열은 일반 텍스트로 인코딩됨 (문서 조기 종료 없음)"


def c_marker_in_instruction():
    """지시문이 마커를 흉내내도 토큰 인덱스 경계는 안 흔들려야 한다."""
    inst = f"다음을 보라\n{CODE_MARKER}\nprint(1)"
    out = "def f():\n    return 2"
    ids, n_prompt = encode_example(TOK, inst, out)
    assert TOK.decode(ids[n_prompt:]) == build_completion(out) + END_OF_TEXT, (
        "마커가 섞이자 경계가 밀렸다"
    )
    # 문자열 기반 파싱은 이 입력에서 실제로 틀린다. 그래서 인덱스를 쓴다.
    text = format_example(inst, out)
    assert extract_code(text) != out.rstrip() + "\n", (
        "문자열 파싱이 안 틀렸다면 이 테스트의 전제가 바뀐 것"
    )
    assert INSTRUCTION_MARKER in text
    return "마커 위장 입력에도 토큰 경계 정확 (문자열 파싱은 예상대로 실패)"


# --------------------------------------------------------------- 손실 마스킹

def _masked_loss(model, x, y):
    logits, loss, _ = model(x, targets=y)
    return logits, loss


def c_masked_positions_excluded_from_loss():
    """마스킹된 자리가 실제로 손실 계산에서 빠지는가 (수동 계산과 대조)."""
    m = _tiny().eval()
    inst, out = SAMPLES[1]
    input_ids, labels, n_prompt = build_example(TOK, inst, out, block_size=200)
    x = torch.tensor([input_ids])
    y = torch.tensor([labels])
    with torch.no_grad():
        logits, loss = _masked_loss(m, x, y)
        # 완성 구간만 직접 골라 손실을 계산한다
        sel = torch.tensor(labels) != IGNORE_INDEX
        manual = F.cross_entropy(logits[0][sel], torch.tensor(labels)[sel])
    diff = abs(loss.item() - manual.item())
    assert diff < 1e-5, f"마스킹된 자리가 손실에 섞였다: 차이 {diff:.3e}"
    assert int(sel.sum()) < len(labels), "가려진 자리가 하나도 없다"
    return (
        f"손실 {loss.item():.5f} == 완성 구간만 계산한 값 (차이 {diff:.1e}, "
        f"{int(sel.sum())}/{len(labels)} 토큰만 기여)"
    )


def c_prompt_edit_vs_completion_edit():
    """프롬프트 구간을 건드리면 손실 불변, 완성 구간을 건드리면 손실 변동.

    같은 조작(라벨 토큰을 다른 토큰으로 치환)을 마스킹한 라벨과 마스킹하지
    않은 라벨에 각각 걸어 비교한다. 마스킹이 없으면 프롬프트 조작에도 손실이
    움직인다 — 그게 우리가 없애려는 낭비다.
    """
    m = _tiny(seed=1).eval()
    inst, out = SAMPLES[3]
    ids, n_prompt = encode_example(TOK, inst, out)
    input_ids, masked, np2 = build_example(TOK, inst, out, block_size=200)
    naive = list(ids[1:])  # 마스킹을 안 한 라벨 (비교용)
    x = torch.tensor([input_ids])

    def shift(labels, lo, hi):
        out_ = list(labels)
        for j in range(lo, hi):
            if out_[j] != IGNORE_INDEX:
                out_[j] = (out_[j] + 137) % VOCAB
        return out_

    def loss_of(labels):
        with torch.no_grad():
            return m(x, targets=torch.tensor([labels]))[1].item()

    end = len(masked)
    base_masked, base_naive = loss_of(masked), loss_of(naive)
    masked_prompt_edit = loss_of(shift(masked, 0, n_prompt - 1))
    naive_prompt_edit = loss_of(shift(naive, 0, n_prompt - 1))
    masked_completion_edit = loss_of(shift(masked, n_prompt - 1, end))

    d_mask_prompt = abs(masked_prompt_edit - base_masked)
    d_naive_prompt = abs(naive_prompt_edit - base_naive)
    d_mask_completion = abs(masked_completion_edit - base_masked)

    assert d_mask_prompt == 0.0, (
        f"프롬프트 구간을 바꿨는데 손실이 움직였다: {d_mask_prompt:.3e} (마스킹 누락)"
    )
    assert d_naive_prompt > 1e-3, (
        "마스킹 없는 라벨조차 프롬프트 조작에 반응하지 않는다 (비교 자체가 무의미)"
    )
    assert d_mask_completion > 1e-3, (
        f"완성 구간을 바꿨는데 손실이 그대로다: {d_mask_completion:.3e}"
    )
    return (
        f"프롬프트 조작 -> 마스킹 {d_mask_prompt:.1e} / 마스킹 없으면 "
        f"{d_naive_prompt:.2e}, 완성 조작 -> {d_mask_completion:.2e}"
    )


def c_no_gradient_on_masked_positions():
    """가려진 자리의 로짓에는 기울기가 정확히 0이어야 한다."""
    m = _tiny(seed=2)
    inst, out = SAMPLES[0]
    input_ids, labels, n_prompt = build_example(TOK, inst, out, block_size=200)
    x = torch.tensor([input_ids])
    logits, _, _ = m(x)
    logits.retain_grad()
    loss = F.cross_entropy(
        logits.view(-1, VOCAB), torch.tensor(labels), ignore_index=IGNORE_INDEX
    )
    loss.backward()
    g = logits.grad[0]
    masked = g[: n_prompt - 1].abs().max().item()
    alive = g[n_prompt - 1 :].abs().max().item()
    assert masked == 0.0, f"가려진 자리에 기울기가 흐른다: {masked:.3e}"
    assert alive > 0.0, "살아있는 자리에 기울기가 안 흐른다"
    return f"마스킹 구간 기울기 최대 {masked:.1e}, 손실 구간 {alive:.2e}"


# --------------------------------------------------------------- 데이터셋

def _jsonl(rows) -> Path:
    f = tempfile.NamedTemporaryFile(
        "w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    f.close()
    return Path(f.name)


def c_empty_instruction_and_output():
    """빈 지시는 살리되, 빈 출력은 버려야 한다."""
    ids, n_prompt = encode_example(TOK, "", "def f():\n    return 1")
    assert n_prompt > 0 and TOK.decode(ids[:n_prompt]) == build_prompt("")
    built = build_example(TOK, "", "def f():\n    return 1", block_size=200)
    assert built is not None, "빈 지시 샘플이 통째로 사라졌다"

    # 빈 출력은 "지시를 보면 곧바로 EOT"를 가르친다. 버려야 한다.
    ids2, np2 = encode_example(TOK, "무언가 해라", "")
    assert ids2[np2:] == [EOT_ID], f"빈 출력의 완성 구간이 이상: {ids2[np2:]}"
    path = _jsonl([
        {"instruction": "무언가 해라", "output": ""},
        {"instruction": "공백만", "output": "   \n  "},
        {"instruction": "정상", "output": "def f():\n    return 1"},
        {"instruction": "타입오류", "output": 123},
    ])
    ds = SFTDataset.from_jsonl(path, TOK, block_size=200)
    path.unlink()
    assert len(ds) == 1, f"남은 샘플 {len(ds)}개, 기대 1개"
    assert ds.stats.dropped_empty_output == 2, ds.stats.report()
    assert ds.stats.dropped_bad_record == 1, ds.stats.report()
    return f"빈 지시 유지 / 빈 출력 2개·형식오류 1개 탈락 ({ds.stats.report()})"


def c_too_long_dropped_not_truncated():
    """block_size 초과 샘플은 조용히 잘리지 말고 버려지고 집계돼야 한다."""
    long_out = "def f():\n" + "\n".join(f"    x{i} = {i}" for i in range(400))
    path = _jsonl([
        {"instruction": "짧은 것", "output": "def f():\n    return 1"},
        {"instruction": "긴 것", "output": long_out},
    ])
    ds = SFTDataset.from_jsonl(path, TOK, block_size=128)
    path.unlink()
    assert len(ds) == 1, f"길이 초과가 안 버려졌다: {len(ds)}개"
    assert ds.stats.dropped_too_long == 1, ds.stats.report()
    assert ds.stats.max_len_seen > 128, "길이 집계가 안 된다"
    # 남은 샘플 중 block_size를 넘는 것이 하나도 없어야 한다
    assert all(len(x) <= 128 for x, _ in ds.examples), "block_size 초과가 남았다"
    # 경계값: 정확히 block_size+1 토큰인 샘플은 살아야 한다
    inst, out = "경계", "def f():\n    return 1"
    ids, _ = encode_example(TOK, inst, out)
    assert build_example(TOK, inst, out, block_size=len(ids) - 1) is not None
    assert build_example(TOK, inst, out, block_size=len(ids) - 2) is None
    return f"초과 1개 탈락 (최장 {ds.stats.max_len_seen} 토큰), 경계값 정확"


def c_padding_is_masked_and_neutral():
    """패딩 자리는 라벨 -1이고, 배치 손실이 개별 손실 합과 같아야 한다."""
    rows = [
        {"instruction": i, "output": o}
        for i, o in [SAMPLES[0], SAMPLES[3], SAMPLES[1]]
    ]
    path = _jsonl(rows)
    ds = SFTDataset.from_jsonl(path, TOK, block_size=256)
    path.unlink()
    x, y = ds.collate([0, 1, 2])
    widths = [len(a) for a, _ in ds.examples]
    assert x.shape[1] == max(widths), f"배치 폭 {x.shape[1]}, 기대 {max(widths)}"
    for r, w in enumerate(widths):
        assert (y[r, w:] == IGNORE_INDEX).all(), "패딩 자리 라벨이 -1이 아니다"
        assert (x[r, w:] == ds.pad_id).all(), "패딩 입력이 pad_id가 아니다"

    m = _tiny(seed=3).eval()
    with torch.no_grad():
        _, batch_loss, _ = m(x, targets=y)
        total, count = 0.0, 0
        for i in range(len(ds)):
            xi, yi = ds[i]
            logits, _, _ = m(xi[None])
            total += F.cross_entropy(
                logits.view(-1, VOCAB), yi, ignore_index=IGNORE_INDEX, reduction="sum"
            ).item()
            count += int((yi != IGNORE_INDEX).sum())
    diff = abs(batch_loss.item() - total / count)
    assert diff < 1e-4, f"패딩이 손실을 왜곡한다: 차이 {diff:.3e}"
    return f"패딩 라벨 -1, 배치 손실 == 개별 합/{count} (차이 {diff:.1e})"


def c_real_dataset_usable():
    """make_dataset.py가 만든 실제 데이터가 200개 이상 살아남아야 한다."""
    path = ROOT / "finetune" / "data" / "sft_train.jsonl"
    assert path.exists(), f"{path}가 없다. make_dataset.py build를 먼저 돌릴 것."
    ds = SFTDataset.from_jsonl(path, TOK, block_size=1024)
    assert len(ds) >= 200, f"쓸 수 있는 샘플이 {len(ds)}개뿐이다"
    frac = ds.stats.supervised_tokens / ds.stats.kept_tokens
    # 손실 대상이 너무 적으면 프롬프트가 데이터의 대부분이라는 뜻이다
    assert 0.2 < frac < 0.95, f"손실 대상 비율이 이상: {frac:.2f}"
    return ds.stats.report()


# --------------------------------------------------------------- 학습/불변식

def c_sft_step_reduces_loss():
    """초소형 모델 CPU SFT로 손실이 실제로 내려가는가."""
    rows = [
        {"instruction": f"{i}번 함수를 만들어라", "output": f"def f{i}(x):\n    return x + {i}"}
        for i in range(6)
    ]
    path = _jsonl(rows)
    ds = SFTDataset.from_jsonl(path, TOK, block_size=128)
    path.unlink()
    assert len(ds) == 6

    cfg = SFTConfig(
        lr=3e-3, warmup_iters=5, max_iters=40, batch_size=3, grad_accum=1,
        block_size=128, log_interval=1000,
    )
    m = _tiny(seed=4)
    opt = make_optimizer(m, cfg)
    hist = sft_loop(m, opt, {"train": ds}, cfg, DEVICE, verbose=False)

    first = sum(h["loss"] for h in hist[:5]) / 5
    last = sum(h["loss"] for h in hist[-5:]) / 5
    assert last < first - 0.5, f"손실이 안 내려간다: {first:.4f} -> {last:.4f}"
    assert all(math.isfinite(h["loss"]) for h in hist), "손실에 NaN/Inf"
    return f"40스텝 CPU SFT: loss {first:.4f} -> {last:.4f}"


def c_vocab_unchanged():
    """어휘 16,384가 그대로여야 한다. 특수 토큰이 늘면 임베딩이 깨진다."""
    assert TOK.vocab_size == 16384, f"어휘가 바뀌었다: {TOK.vocab_size}"
    assert TOK.specials == [END_OF_TEXT], f"특수 토큰이 늘었다: {TOK.specials}"
    assert ModelConfig().vocab_size == 16384, "모델 기본 어휘가 바뀌었다"
    # 포맷터가 만드는 토큰이 전부 기존 어휘 안에 있어야 한다
    for inst, out in SAMPLES:
        ids, _ = encode_example(TOK, inst, out)
        assert max(ids) < 16384 and min(ids) >= 0, "어휘 범위 밖 토큰"
    assert EOT_ID == 16383, f"EOT id가 {EOT_ID}로 밀렸다"
    return f"vocab=16,384, specials={TOK.specials}, EOT id={EOT_ID}"


def c_masking_ratio_sane():
    """지시문이 길어질수록 마스킹 비율이 커져야 한다 (마스킹이 살아있는지 확인)."""
    ratios = []
    for n in (1, 10, 100):
        inst = "설명 " * n
        _, labels, _ = build_example(TOK, inst, "def f():\n    return 1", block_size=1024)
        ratios.append(sum(1 for v in labels if v == IGNORE_INDEX) / len(labels))
    assert ratios[0] < ratios[1] < ratios[2], f"마스킹 비율이 단조롭지 않다: {ratios}"
    assert ratios[2] > 0.8, f"긴 지시문인데 마스킹 비율이 낮다: {ratios[2]:.2f}"
    return "지시문 길이별 마스킹 비율 " + ", ".join(f"{r:.2f}" for r in ratios)


# --------------------------------------------------- 학습 루프/CLI 배선

def c_eval_does_not_consume_epoch():
    """평가가 학습 에포크 순회를 갉아먹지 않아야 한다.

    estimate_loss는 train 스플릿에서도 batch()를 부른다. SFTDataset은
    BinDataset과 달리 커서를 가진 순차 순회라, 평가가 뽑아간 샘플은 그
    에포크의 학습에서 그대로 빠진다. 손실 곡선에는 전혀 안 나타난다.
    """
    rows = [
        {"instruction": f"{i}번 함수를 만들어라", "output": f"def f{i}(x):\n    return x + {i}"}
        for i in range(16)
    ]
    path = _jsonl(rows)
    ds = SFTDataset.from_jsonl(path, TOK, block_size=128)
    vds = SFTDataset.from_jsonl(path, TOK, block_size=128)
    path.unlink()
    assert len(ds) == 16

    # 학습 draw만 기록한다. estimate_loss는 no_grad 안이라 이걸로 갈린다.
    seen = []
    orig_collate = ds.collate

    def spy(indices):
        if torch.is_grad_enabled():
            seen.extend(indices)
        return orig_collate(indices)

    ds.collate = spy

    # 정확히 1에포크(16샘플)만큼 돌리고, 매 스텝 평가를 끼운다
    cfg = SFTConfig(
        lr=1e-4, warmup_iters=1, max_iters=4, batch_size=4, grad_accum=1,
        block_size=128, eval_interval=1, eval_iters=2, log_interval=1000,
    )
    m = _tiny(seed=7)
    opt = make_optimizer(m, cfg)
    sft_loop(m, opt, {"train": ds, "val": vds}, cfg, DEVICE, verbose=False)

    assert len(seen) == 16, f"학습 draw가 {len(seen)}개, 기대 16개"
    assert len(set(seen)) == 16, (
        f"1에포크에 {len(set(seen))}/16개만 학습됐다 - 평가가 커서를 소모한다"
    )
    return f"1에포크 학습 draw {len(seen)}개 전부 서로 다름 (평가 4회 끼운 상태)"


def c_epoch_state_roundtrip():
    """epoch_state/restore_epoch_state가 순회 상태를 정확히 되돌려야 한다."""
    rows = [{"instruction": f"{i}번", "output": f"def f{i}():\n    return {i}"} for i in range(10)]
    path = _jsonl(rows)
    ds = SFTDataset.from_jsonl(path, TOK, block_size=128)
    path.unlink()
    ds.batch(3, DEVICE)
    saved = ds.epoch_state()
    ds.batch(4, DEVICE)
    ds.batch(4, DEVICE)  # 에포크를 넘겨 _order 자체가 바뀌게 한다
    ds.restore_epoch_state(saved)
    assert (ds._order, ds._cursor) == (saved[0], saved[1]), "복원이 안 됐다"
    # 스냅샷은 얕은 참조면 안 된다 (뒤에서 _order가 바뀌면 같이 오염된다)
    saved2 = ds.epoch_state()
    ds._order[0] = -999
    assert saved2[0][0] != -999, "스냅샷이 내부 리스트를 그대로 참조한다"
    return "커서/순서 복원 정확, 스냅샷은 복사본"


def _fake_base(tmpdir: Path) -> Path:
    """CPU 초소형 기반 체크포인트. GPU는 학습 중이라 절대 안 쓴다."""
    m = _tiny(seed=11)
    save_checkpoint(
        tmpdir / "base.pt", m, make_optimizer(m, SFTConfig()), TINY_CFG, SFTConfig(), 9, 4.2
    )
    return tmpdir / "base.pt"


def _args(**kw):
    import argparse
    d = dict(base=None, train=None, val=None, lr=None, epochs=None, batch_size=2,
             max_iters=2, device="cpu", out_dir=None)
    d.update(kw)
    return argparse.Namespace(**d)


def c_out_dir_cannot_clobber_base():
    """--out-dir이 기반 체크포인트 폴더면 거부해야 한다.

    sft_loop는 out_dir/latest.pt를 쓴다. 그 경로가 checkpoints/면
    train.py --resume이 읽는 파일을 SFT 가중치로 덮어써서 프리트레이닝이
    통째로 날아간다. 실제 checkpoints/를 위험에 두지 않으려고
    BASE_CKPT_DIR을 임시 폴더로 바꿔 같은 코드 경로를 시험한다.
    """
    import finetune.sft as sft_mod

    tmp = Path(tempfile.mkdtemp())
    base = _fake_base(tmp)
    rows = [{"instruction": f"{i}번", "output": f"def f{i}():\n    return {i}"} for i in range(4)]
    tp = _jsonl(rows)

    original = sft_mod.BASE_CKPT_DIR
    sft_mod.BASE_CKPT_DIR = tmp
    try:
        try:
            sft_mod.run(_args(base=str(base), train=str(tp), val=str(tmp / "nope.jsonl"),
                              out_dir=str(tmp)))
        except SystemExit as e:
            msg = str(e)
        else:
            msg = None
    finally:
        sft_mod.BASE_CKPT_DIR = original
        tp.unlink()

    assert msg is not None, "기반 체크포인트 폴더에 쓰는 것을 막지 않았다"
    assert "덮어써" in msg or "out-dir" in msg, f"메시지가 이상하다: {msg}"
    assert not (tmp / "latest.pt").exists(), "거부했는데 latest.pt가 생겼다"
    return "기반 폴더 --out-dir 거부, latest.pt 미생성"


def c_empty_val_does_not_crash_midrun():
    """val 샘플이 전부 탈락해도 첫 평가에서 죽으면 안 된다.

    기본 설정이면 50스텝째에 죽고 ckpt_interval이 100이라 그때까지 저장된
    체크포인트도 없다. 시작 시점에 걸러야 한다.
    """
    import finetune.sft as sft_mod

    tmp = Path(tempfile.mkdtemp())
    base = _fake_base(tmp)
    tp = _jsonl([
        {"instruction": f"{i}번", "output": f"def f{i}():\n    return {i}"} for i in range(4)
    ])
    # block_size(=TINY_CFG.max_seq_len=256)를 확실히 넘는 val 샘플만 넣는다
    long_out = "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(400))
    vp = _jsonl([{"instruction": "아주 긴 것", "output": long_out}])

    vds = SFTDataset.from_jsonl(vp, TOK, block_size=256)
    assert len(vds) == 0, "val이 안 비었다 - 시험 전제가 깨졌다"

    sft_mod.run(_args(base=str(base), train=str(tp), val=str(vp),
                      out_dir=str(tmp / "out")))
    tp.unlink()
    vp.unlink()
    assert (tmp / "out" / "latest.pt").exists(), "체크포인트가 안 남았다"
    return "빈 val에서 평가를 건너뛰고 정상 종료, latest.pt 저장됨"


def c_drop_stats_add_up():
    """탈락 집계가 전체와 맞아떨어져야 한다 (사유 하나가 다른 걸 흡수하면 안 됨)."""
    path = ROOT / "finetune" / "data" / "sft_train.jsonl"
    ds = SFTDataset.from_jsonl(path, TOK, block_size=256)
    s = ds.stats
    total = s.kept + s.dropped_too_long + s.dropped_empty_output + s.dropped_bad_record
    assert total == s.total, f"집계 합 {total} != 전체 {s.total}"
    assert s.dropped_too_long > 0, "block_size 256인데 탈락이 하나도 없다"
    return f"합계 일치 ({s.total:,} = 유지 {s.kept:,} + 탈락 {s.total - s.kept:,})"


def c_split_encoding_matches_whole_doc():
    """따로 인코딩해 이어붙인 토큰열이 통짜 인코딩과 완전히 같아야 한다.

    다르면 학습 문서가 프리트레이닝 코퍼스와 미묘하게 다른 토큰 분포를
    갖게 된다. 왕복 무손실만으로는 이걸 못 잡는다(디코딩은 같아도 토큰이
    다를 수 있다).
    """
    rows = [json.loads(l) for l in open(
        ROOT / "finetune" / "data" / "sft_train.jsonl", encoding="utf-8"
    )][:300]
    bad = 0
    for d in rows:
        whole = TOK.encode(
            build_prompt(d["instruction"]) + build_completion(d["output"]),
            allow_special=False,
        )
        split, _ = encode_example(TOK, d["instruction"], d["output"], add_eot=False)
        if whole != split:
            bad += 1
    assert bad == 0, f"{bad}/{len(rows)}개에서 토큰열이 다르다"
    return f"실제 샘플 {len(rows)}개 토큰열 동일 (경계 병합 없음)"


def c_train_val_no_leak():
    """train과 val에 같은 지시문이 있으면 val loss가 낙관적으로 나온다.

    best.pt 선택이 val loss에 걸려 있어서, 누수는 '더 나쁜 모델을 고르는'
    형태로 조용히 나타난다.
    """
    d = ROOT / "finetune" / "data"
    tr = [json.loads(l) for l in open(d / "sft_train.jsonl", encoding="utf-8")]
    va = [json.loads(l) for l in open(d / "sft_val.jsonl", encoding="utf-8")]
    assert len(va) >= 20, f"val이 {len(va)}개뿐이다"
    inst_overlap = {x["instruction"] for x in tr} & {x["instruction"] for x in va}
    assert not inst_overlap, (
        f"지시문 {len(inst_overlap)}개가 train/val 양쪽에 있다 (예: "
        f"{sorted(inst_overlap)[0][:40]!r})"
    )
    return f"train {len(tr):,} / val {len(va):,}, 지시문 누수 0"


def main():
    print("=" * 60)
    print(f"SFT 파이프라인 적대적 검증 (device={DEVICE})")
    print("=" * 60)

    check("포맷: 프롬프트/완성 경계 정확", c_boundary_exact)
    check("마스킹 시작 인덱스", c_mask_starts_at_right_index)
    check("포맷 왕복 무손실", c_roundtrip_preserves_text)
    check("유니코드/한글 처리", c_unicode_and_hangul)
    check("EOT 문자열 주입 차단", c_eot_injection_blocked)
    check("마커 위장 지시문", c_marker_in_instruction)
    check("마스킹 자리 손실 제외", c_masked_positions_excluded_from_loss)
    check("프롬프트/완성 변경 시 손실 반응", c_prompt_edit_vs_completion_edit)
    check("마스킹 자리 기울기 0", c_no_gradient_on_masked_positions)
    check("빈 지시/빈 출력", c_empty_instruction_and_output)
    check("길이 초과는 절단 아닌 폐기", c_too_long_dropped_not_truncated)
    check("패딩 마스킹/무해성", c_padding_is_masked_and_neutral)
    check("실제 데이터셋 사용 가능", c_real_dataset_usable)
    check("CPU SFT 손실 감소", c_sft_step_reduces_loss)
    check("어휘 16,384 불변", c_vocab_unchanged)
    check("마스킹 비율 단조성", c_masking_ratio_sane)
    check("평가가 에포크를 갉아먹지 않음", c_eval_does_not_consume_epoch)
    check("순회 상태 저장/복원", c_epoch_state_roundtrip)
    check("--out-dir이 기반 체크포인트를 못 덮음", c_out_dir_cannot_clobber_base)
    check("빈 val에서 중도 사망 없음", c_empty_val_does_not_crash_midrun)
    check("탈락 사유 집계 정합", c_drop_stats_add_up)
    check("분할 인코딩 == 통짜 인코딩", c_split_encoding_matches_whole_doc)
    check("train/val 누수 없음", c_train_val_no_leak)

    print("=" * 60)
    failed = [r for r in RESULTS if not r[0]]
    print(f"결과: {len(RESULTS) - len(failed)}/{len(RESULTS)} 통과")
    if failed:
        print("\n실패 항목:")
        for _, name, detail in failed:
            print(f"  - {name}: {detail}")
        print("\n판정: 위험 - SFT 파이프라인을 믿을 수 없다")
        return 1
    print("\n판정: 통과 - SFT 파이프라인 사용 가능")
    return 0


if __name__ == "__main__":
    sys.exit(main())
