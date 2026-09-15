"""README에 넣을 그림을 그린다.

학습 곡선·학습률·타임라인은 checkpoints/trainlog.jsonl에서, 평가 격자는
eval/results.json에서 나온다. 구조도·파라미터 구성·데이터 깔때기·정밀도는
측정해 둔 값을 이 파일 상단 상수에 적어 두고 쓴다(출처는 각 상수의 주석).

matplotlib은 이 스크립트에서만 쓴다. 학습/추론 경로의 의존성이 아니므로
.venv에 넣지 말고 문서를 다시 그릴 때만 따로 설치할 것.

  uv pip install matplotlib
  python scripts/plot_training.py --font NotoSansKR-Regular.ttf NotoSansKR-Bold.ttf

라이트/다크 두 벌을 만든다. GitHub README가 <picture>로 테마에 맞춰 고른다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "checkpoints" / "trainlog.jsonl"
EVAL = ROOT / "eval" / "results.json"
OUT = ROOT / "docs" / "assets"

# 검증된 기본 팔레트. 슬롯 순서가 색각 이상 분리를 보장하므로 바꾸지 않는다.
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
        grid="#e3e2de", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
    ),
    "dark": dict(
        surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7",
        grid="#34332f", s1="#3987e5", s2="#d95926", s3="#199e70",
    ),
}

# 학습 이력. 시작 시각은 train_stdout.log의 [detached] 헤더, 사망 시각은
# 그때 사람이 확인해 적어둔 값이다(로그에 스텝별 시각이 없어 추정하지 않는다).
#   (시작 시각, 시작 iter, 끝 iter, 끝난 시각)
RUNS = [
    (datetime(2026, 8, 31, 16, 10, 35), 0, 5670, datetime(2026, 9, 5, 22, 57)),
    (datetime(2026, 9, 6, 22, 21, 5), 5501, 11860, datetime(2026, 9, 12, 20, 45)),
    (datetime(2026, 9, 14, 11, 30, 18), 11751, 13120, datetime(2026, 9, 15, 18, 15, 19)),
]


def load():
    train, ev = [], []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "eval" in d:
            ev.append(d)
        else:
            train.append(d)
    return train, ev


def style(ax, t, xlabel, ylabel, title=None):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.grid(True, color=t["grid"], linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5)
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=11.5, fontweight="bold",
                     loc="left", pad=10)


def thousands(x, _):
    return f"{int(x):,}"


def fig_loss(train, ev, t, path):
    """손실 곡선. 왼쪽은 전 구간, 오른쪽은 1,000스텝 이후 확대."""
    it = [d["iter"] for d in train]
    ls = [d["loss"] for d in train]
    ei = [d["iter"] for d in ev]
    el = [d["eval"]["val"] for d in ev]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=t["surface"])
    for ax, lo in zip(axes, (0, 1000)):
        xs = [a for a in it if a >= lo]
        ys = [b for a, b in zip(it, ls) if a >= lo]
        ax.plot(xs, ys, color=t["s1"], linewidth=1.0, alpha=0.75,
                label="학습 손실", solid_capstyle="round")
        ax.plot(ei, el, color=t["s2"], linewidth=2.0, marker="o",
                markersize=6, markeredgecolor=t["surface"],
                markeredgewidth=1.2, label="검증 손실", zorder=3)
        ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    axes[0].set_ylim(0, 10.3)
    axes[1].set_xlim(800, 13600)
    axes[1].set_ylim(0.68, 1.65)

    style(axes[0], t, "스텝", "손실", "전 구간 — 13,121스텝 / 6.88B 토큰 (교차 엔트로피)")
    style(axes[1], t, "스텝", "", "확대 — 1,000스텝 이후")

    # 최저 검증 손실은 숫자로 박아 둔다. 곡선에서 눈으로 읽을 값이 아니다.
    bi = min(range(len(el)), key=lambda i: el[i])
    axes[1].annotate(
        f"최저 검증 손실 {el[bi]:.4f}\n(ppl {ev[bi]['val_ppl']:.2f})",
        xy=(ei[bi], el[bi]), xytext=(-18, 34), textcoords="offset points",
        color=t["ink"], fontsize=9, ha="right", linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=1),
    )
    leg = axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_schedule(train, t, path):
    """학습률 스케줄과 기울기 노름. 학습이 건강했는지 보는 두 지표다."""
    it = [d["iter"] for d in train]
    lr = [d["lr"] for d in train]
    gn = [d["gnorm"] for d in train]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.4), sharex=True,
                             facecolor=t["surface"])
    axes[0].plot(it, lr, color=t["s1"], linewidth=2.0, solid_capstyle="round")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    style(axes[0], t, "", "학습률", "학습률 — 500스텝 워밍업 뒤 코사인 감쇠")
    axes[0].annotate("최고 6.0e-04", xy=(500, 6e-4), xytext=(14, -4),
                     textcoords="offset points", color=t["ink"], fontsize=9)
    axes[0].annotate("바닥 6.0e-05 (최고의 10%)", xy=(13120, 6e-5),
                     xytext=(-8, 16), textcoords="offset points",
                     color=t["ink"], fontsize=9, ha="right")

    axes[1].plot(it, gn, color=t["s3"], linewidth=1.0, alpha=0.85)
    axes[1].set_yscale("log")
    style(axes[1], t, "스텝", "기울기 노름 (로그)",
          "기울기 노름 — 워밍업 끝 스파이크를 클리핑이 흡수했다")
    axes[1].annotate(
        "iter 480: 노름 292.6 / 손실 5.74\n클리핑이 잘라내고 1,500스텝 안에 회복",
        xy=(480, 292.59), xytext=(62, -34), textcoords="offset points",
        color=t["ink"], fontsize=9, linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=1))
    axes[1].axhline(1.0, color=t["s2"], linewidth=1.5, linestyle=(0, (4, 3)))
    axes[1].annotate("클리핑 한계 1.0", xy=(13120, 1.0), xytext=(-8, 8),
                     textcoords="offset points", color=t["ink2"], fontsize=9,
                     ha="right")
    axes[1].xaxis.set_major_formatter(FuncFormatter(thousands))

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def fig_timeline(t, path):
    """벽시계 대비 진행 스텝. 학습이 두 번 죽었고 얼마를 잃었는지 보여준다.

    구간의 양끝은 기록된 시각이고 그 사이는 직선으로 잇는다. 처리량이 전
    구간 6,505~6,524 tok/s로 평평했으므로 직선 근사가 실제와 거의 같다.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.0), facecolor=t["surface"])

    wall = (RUNS[-1][3] - RUNS[0][0]).total_seconds() / 86400
    computed = sum(i1 - i0 for _, i0, i1, _ in RUNS)  # 되감긴 스텝은 두 번 센다
    compute = sum((e - s).total_seconds() for s, _, _, e in RUNS) / 86400
    ends = [end for _, _, _, end in RUNS]
    for i, (start, i0, i1, end) in enumerate(RUNS):
        ax.plot([start, end], [i0, i1], color=t["s1"], linewidth=2.6,
                solid_capstyle="round", zorder=3,
                label="학습 진행" if i == 0 else None)

    for (end, (nstart, ni0, _, _)) in zip(ends, RUNS[1:]):
        # 죽은 지점에서 재개 지점까지. 세로 낙차가 되돌아간 스텝이다.
        ax.plot([end, nstart], [0, 0], color=t["surface"], linewidth=0)
        ax.axvspan(end, nstart, color=t["s2"], alpha=0.13, zorder=1)
        ax.annotate(
            f"{(nstart - end).total_seconds() / 3600:.0f}시간 정지",
            xy=(end + (nstart - end) / 2, 1200), color=t["ink2"], fontsize=9,
            ha="center", rotation=90, va="bottom",
        )

    for end, (_, ni0, _, _), (_, _, pi1, _) in zip(ends, RUNS[1:], RUNS[:-1]):
        ax.annotate(
            f"{pi1:,}에서 죽고\n{ni0:,}로 되감김 (−{pi1 - ni0})",
            xy=(end, pi1), xytext=(-10, 26), textcoords="offset points",
            color=t["ink"], fontsize=8.5, ha="right", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=1),
        )

    ax.set_ylim(0, 15400)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    style(ax, t, "", "진행 스텝",
          f"벽시계 {wall:.1f}일, 실제 계산 {compute:.1f}일 — 두 번 죽었다")
    print(f"  벽시계 {wall:.2f}일 / 계산 {compute:.2f}일 / "
          f"중복 계산 {computed - RUNS[-1][2]:,}스텝")
    for end, (nstart, _, _, _) in zip(ends, RUNS[1:]):
        print(f"  정지 {end:%m-%d %H:%M} -> {nstart:%m-%d %H:%M} "
              f"({(nstart - end).total_seconds() / 3600:.1f}시간)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"],
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


# 평가 격자 색. 통과만 또렷하게 두고 실패 네 종류는 채도를 낮춰, 격자를 보면
# "통과/실패"가 먼저 읽히고 실패 사유가 그다음에 읽히게 한다.
OUTCOME = {
    "light": dict(**{"pass": "#1baf7a"}, assert_="#eb6834", error="#7a5cc4",
                  syntax="#8a8983", timeout="#3f3e3b"),
    "dark": dict(**{"pass": "#199e70"}, assert_="#d95926", error="#8b6fd4",
                 syntax="#6f6e68", timeout="#9b9a93"),
}
REASON_KO = {"pass": "통과", "assert": "단언 불일치", "error": "실행 오류",
             "syntax": "문법 오류", "timeout": "타임아웃"}

# 파라미터 구성. 손으로 센 게 아니라 Transformer를 실제로 만들어 센 값이다.
# 가중치 공유(tie_embeddings) 때문에 lm_head는 임베딩과 같은 텐서라 한 번만 센다.
#   .venv/bin/python -c "from model.transformer import make_config, Transformer; ..."
#   (레이어 1장) 어텐션 2,621,440 + FFN 8,454,144 + 노름 2,048 = 11,077,632
PARAMS_TOTAL = [("FFN (SwiGLU)", 202_899_456), ("어텐션 (GQA)", 62_914_560),
                ("임베딩 = lm_head", 16_777_216), ("RMSNorm", 50_176)]
PARAMS_LAYER = [("FFN (SwiGLU)", 8_454_144), ("어텐션 (GQA)", 2_621_440),
                ("RMSNorm", 2_048)]

# scripts/verify_env.py가 이 V100S에서 직접 잰 값. 카탈로그 수치가 아니다.
TFLOPS = [("bf16", 10.0), ("fp32", 13.2), ("fp16", 88.8)]

# data/prepare.py filter 단계의 출력. 합이 맞는다:
#   5,361,373 − 2,020,123 − 466,624 − 6,287 = 2,868,339
FUNNEL = [("라이선스 탈락", 2_020_123, "비허용·불명 라이선스"),
          ("문법 탈락", 466_624, "ast.parse 실패"),
          ("크기 탈락", 6_287, "너무 짧거나 너무 긺")]
FUNNEL_START, FUNNEL_END = 5_361_373, 2_868_339


def _box(ax, x, y, w, h, text, face, edge, t, fs=9, weight="normal", tc=None):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                               linewidth=1.3, joinstyle="round", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", zorder=3,
            color=tc or t["ink"], fontsize=fs, fontweight=weight,
            linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, t, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle=style, color=t["ink2"], linewidth=1.2))


def fig_arch(t, path):
    """구조도. 표로 적어둔 숫자가 실제로 어떻게 쌓여 있는지 보여준다."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.9), facecolor=t["surface"],
                             gridspec_kw=dict(width_ratios=[1, 1.35, 1.15]))
    for ax in axes:
        ax.set_facecolor(t["surface"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # 왼쪽 — 전체 스택
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("전체 — 282,641,408 파라미터", color=t["ink"], fontsize=11,
                 fontweight="bold", loc="left", pad=12)
    stack = [("토큰 임베딩\n16,384 × 1,024", t["s1"], 8.55),
             ("트랜스포머 블록 × 24\n(오른쪽에서 펼침)", t["s2"], 5.75),
             ("RMSNorm", t["grid"], 3.15),
             ("lm_head\n임베딩과 가중치 공유", t["s1"], 1.75)]
    for i, (label, col, y) in enumerate(stack):
        h = 1.9 if i == 1 else 1.0
        _box(ax, 0.7, y, 8.6, h, label, t["surface"], col, t,
             fs=9, weight="bold" if i == 1 else "normal")
    _arrow(ax, 5, 8.45, 5, 7.75, t)
    _arrow(ax, 5, 5.65, 5, 4.25, t)
    _arrow(ax, 5, 3.05, 5, 2.85, t)
    ax.text(5, 0.9, "logits 16,384", ha="center", color=t["ink2"], fontsize=9)
    _arrow(ax, 5, 1.65, 5, 1.15, t)
    ax.text(5, 9.75, "토큰 id", ha="center", color=t["ink2"], fontsize=9)
    _arrow(ax, 5, 9.6, 5, 9.65, t, style="-")

    # 가운데 — 블록 하나
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("블록 1장 — pre-norm, 잔차 2개", color=t["ink"], fontsize=11,
                 fontweight="bold", loc="left", pad=12)
    inner = [("RMSNorm", t["grid"], 7.9), ("GQA 어텐션 + RoPE", t["s2"], 6.3),
             ("RMSNorm", t["grid"], 4.0), ("SwiGLU FFN", t["s3"], 2.4)]
    for label, col, y in inner:
        _box(ax, 2.2, y, 5.4, 1.1, label, t["surface"], col, t)
    for y0, y1 in ((7.8, 7.45), (6.2, 5.75), (3.9, 3.55), (2.3, 1.85)):
        _arrow(ax, 4.9, y0, 4.9, y1, t)
    # 잔차 연결
    for y_top, y_join, lbl in ((9.05, 5.5, "⊕"), (5.2, 1.6, "⊕")):
        ax.plot([1.2, 1.2], [y_join, y_top], color=t["s1"], linewidth=1.4,
                linestyle=(0, (4, 3)), zorder=1)
        ax.plot([1.2, 4.9], [y_join, y_join], color=t["s1"], linewidth=1.4,
                linestyle=(0, (4, 3)), zorder=1)
        ax.text(4.9, y_join, lbl, ha="center", va="center", fontsize=13,
                color=t["s1"], zorder=4,
                bbox=dict(boxstyle="circle,pad=0.12", fc=t["surface"],
                          ec=t["s1"], linewidth=1.3))
    ax.text(9.0, 6.85, "2.62M", ha="right", color=t["ink2"], fontsize=8.5)
    ax.text(9.0, 2.95, "8.45M", ha="right", color=t["ink2"], fontsize=8.5)
    ax.text(5, 0.55, "레이어 1장 11.08M × 24층", ha="center",
            color=t["ink2"], fontsize=9)
    ax.text(5, 9.5, "x", ha="center", color=t["ink2"], fontsize=9)

    # 오른쪽 — GQA 헤드 공유
    ax = axes[2]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_title("GQA — Q 16헤드가 KV 4헤드를 공유", color=t["ink"],
                 fontsize=11, fontweight="bold", loc="left", pad=12)
    for g in range(4):
        gy = 8.15 - g * 1.95
        for h in range(4):
            _box(ax, 0.6 + h * 1.15, gy, 0.95, 0.72, "", t["s2"], t["s2"], t)
        _box(ax, 6.9, gy, 1.5, 0.72, "KV", t["surface"], t["s1"], t, fs=8.5)
        _arrow(ax, 5.35, gy + 0.36, 6.8, gy + 0.36, t)
    ax.text(2.9, 9.35, "Q 16헤드", ha="center", color=t["ink2"], fontsize=9)
    ax.text(7.65, 9.35, "KV 4헤드", ha="center", color=t["ink2"], fontsize=9)
    ax.text(5, 0.62, "KV 캐시가 1/4로 줄어 2,048토큰 생성이 32GB에 들어간다",
            ha="center", color=t["ink"], fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def fig_params(t, path):
    """파라미터가 어디에 있는지. 어텐션이 아니라 FFN이 대부분이다."""
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 3.9), facecolor=t["surface"],
                             gridspec_kw=dict(height_ratios=[1, 1]))
    cols = {"FFN (SwiGLU)": t["s3"], "어텐션 (GQA)": t["s2"],
            "임베딩 = lm_head": t["s1"], "RMSNorm": t["ink2"]}
    for ax, rows, total, title in (
            (axes[0], PARAMS_TOTAL, 282_641_408, "모델 전체 282.6M"),
            (axes[1], PARAMS_LAYER, 11_077_632, "레이어 1장 11.08M")):
        ax.set_facecolor(t["surface"])
        left = 0
        for name, v in rows:
            ax.barh(0, v, left=left, height=0.52, color=cols[name],
                    edgecolor=t["surface"], linewidth=1.4)
            pct = v / total * 100
            if pct > 12:
                ax.text(left + v / 2, 0, f"{name}\n{pct:.1f}%", ha="center",
                        va="center", color=t["surface"], fontsize=9,
                        fontweight="bold", linespacing=1.3)
            elif pct > 1:
                # 조각이 좁아 글자가 안 들어간다. 지시선으로 아래에 뺀다.
                ax.annotate(f"{name} {pct:.1f}%", xy=(left + v / 2, -0.27),
                            xytext=(left + v / 2, -0.66), ha="center",
                            va="top", color=t["ink"], fontsize=8.5,
                            arrowprops=dict(arrowstyle="-", color=t["ink2"],
                                            linewidth=0.9))
            left += v
        ax.set_xlim(0, total); ax.set_ylim(-1.05, 0.45)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, color=t["ink"], fontsize=10.5, fontweight="bold",
                     loc="left", pad=7)
    axes[1].text(11_077_632, -0.95, "RMSNorm 50,176개(0.02%)는 눈에 보이지 않는다",
                 ha="right", color=t["ink2"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def fig_precision(t, path):
    """실측 TFLOPS. bf16이 '지원'되지만 fp32보다 느리다는 것이 요점이다."""
    fig, ax = plt.subplots(figsize=(8.0, 2.9), facecolor=t["surface"])
    names = [n for n, _ in TFLOPS]
    vals = [v for _, v in TFLOPS]
    cols = [t["ink2"], t["s2"], t["s3"]]
    bars = ax.barh(names, vals, height=0.58, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v + 1.6, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", color=t["ink"], fontsize=10, fontweight="bold")
    ax.set_xlim(0, 104)
    style(ax, t, "실측 TFLOPS", "",
          "정밀도별 실측 성능 — V100S, scripts/verify_env.py")
    ax.grid(False)
    ax.annotate("bf16은 텐서코어가 아니라 에뮬레이션이다.\n"
                "'지원함'만 믿었다면 8.9배 느리게 돌았다",
                xy=(8.4, 0.26), xytext=(30, 0.62), color=t["ink"], fontsize=9,
                linespacing=1.4,
                arrowprops=dict(arrowstyle="-", color=t["ink2"], linewidth=1))
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def fig_funnel(t, path):
    """문서 깔때기. 버린 절반의 대부분이 라이선스 때문이라는 것이 요점이다."""
    fig, ax = plt.subplots(figsize=(9.4, 3.6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    rows = [("원본 문서", FUNNEL_START, t["s1"], "")]
    run = FUNNEL_START
    for name, drop, why in FUNNEL:
        run -= drop
        rows.append((name, drop, t["s2"], why))
    rows.append(("유지", FUNNEL_END, t["s3"], "53.5% / 24.24 GB"))

    y = len(rows)
    left_run = FUNNEL_START
    for i, (name, v, col, why) in enumerate(rows):
        y -= 1
        if i == 0:
            x0 = 0
        elif i == len(rows) - 1:
            x0 = 0
        else:
            left_run -= v
            x0 = left_run
        ax.barh(y, max(v, 26_000), left=x0, height=0.6, color=col,
                edgecolor=t["surface"], linewidth=1.2)
        ax.text(-90_000, y, name, ha="right", va="center", color=t["ink"],
                fontsize=9.5)
        lbl = f"{v:,}" if i in (0, len(rows) - 1) else f"−{v:,}"
        ax.text(x0 + max(v, 26_000) + 70_000, y, lbl, ha="left", va="center",
                color=t["ink"], fontsize=9,
                fontweight="bold" if i in (0, len(rows) - 1) else "normal")
        if why:
            ax.text(x0 + max(v, 26_000) + 900_000, y, why, ha="left", va="center",
                    color=t["ink2"], fontsize=8.5)
    ax.set_xlim(0, 6_900_000); ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("codeparrot-clean 54샤드 → 학습 코퍼스",
                 color=t["ink"], fontsize=11, fontweight="bold", loc="left",
                 pad=10)
    ax.text(0, -0.62, "중복 제거는 0건이었다 — 원본이 이미 해시 중복을 걷어낸 뒤였다",
            color=t["ink2"], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def fig_eval(res, t, path):
    """문제 10개 × 5회 생성의 결과 격자. pass@5가 왜 잡음인지 눈으로 보인다."""
    oc = OUTCOME["light" if t is THEMES["light"] else "dark"]
    key = lambda r: oc["assert_"] if r == "assert" else oc[r]
    probs = res["per_problem"]
    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    counts = {}
    for i, p in enumerate(probs):
        y = len(probs) - 1 - i
        for j, r in enumerate(p["reasons"]):
            counts[r] = counts.get(r, 0) + 1
            ax.add_patch(plt.Rectangle((j, y - 0.38), 0.86, 0.76,
                                       facecolor=key(r), edgecolor=t["surface"],
                                       linewidth=1.5))
        ax.text(-0.25, y, p["name"], ha="right", va="center", color=t["ink"],
                fontsize=9.5, family="monospace")
        ok = p["solved"]
        ax.text(5.35, y, "통과" if ok else "실패", ha="left", va="center",
                fontsize=9.5, fontweight="bold" if ok else "normal",
                color=oc["pass"] if ok else t["ink2"])
    ax.set_xlim(-2.6, 7.2); ax.set_ylim(-1.5, len(probs) - 0.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for j in range(5):
        ax.text(j + 0.43, len(probs) - 0.42, f"{j + 1}회", ha="center",
                color=t["ink2"], fontsize=8.5)
    ax.text(5.35, len(probs) - 0.42, "pass@5", ha="left", color=t["ink2"],
            fontsize=8.5)
    ax.set_title(f"pass@5 {res['pass_at_k'] * 100:.0f}% — 통과 칸은 "
                 f"{counts.get('pass', 0)}/50뿐이다",
                 color=t["ink"], fontsize=11, fontweight="bold", loc="left",
                 pad=10)
    order = ["pass", "assert", "error", "syntax", "timeout"]
    x = -2.55
    for r in order:
        n = counts.get(r, 0)
        ax.add_patch(plt.Rectangle((x, -1.32), 0.42, 0.42, facecolor=key(r),
                                   edgecolor=t["surface"], linewidth=1))
        ax.text(x + 0.56, -1.11, f"{REASON_KO[r]} {n}", va="center",
                color=t["ink2"], fontsize=8.5)
        x += 0.56 + len(REASON_KO[r]) * 0.19 + 0.5
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"], bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", nargs="+", default=[],
                    help="한글 TTF 경로. 없으면 라벨이 전부 두부가 된다. "
                         "가변 폰트는 matplotlib이 굵기를 못 골라 100으로 "
                         "떨어지므로, wght를 고정한 정적 TTF를 넣을 것")
    args = ap.parse_args()

    for path in args.font:
        font_manager.fontManager.addfont(path)
    if args.font:
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=args.font[0]).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    train, ev = load()
    # 평가 격자는 eval/results.json이 있어야 그린다. 체크포인트가 있어야 다시
    # 만들 수 있는 파일이라, 없으면 그 그림만 건너뛰고 나머지는 그린다.
    res = json.loads(EVAL.read_text(encoding="utf-8")) if EVAL.exists() else None
    if res is None:
        print(f"  건너뜀: {EVAL} 이 없어 평가 격자는 그리지 않는다")
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, t in THEMES.items():
        fig_loss(train, ev, t, OUT / f"training-curve-{name}.png")
        fig_schedule(train, t, OUT / f"lr-gnorm-{name}.png")
        fig_timeline(t, OUT / f"timeline-{name}.png")
        fig_arch(t, OUT / f"architecture-{name}.png")
        fig_params(t, OUT / f"params-{name}.png")
        fig_precision(t, OUT / f"precision-{name}.png")
        fig_funnel(t, OUT / f"data-funnel-{name}.png")
        n += 7
        if res:
            fig_eval(res, t, OUT / f"eval-grid-{name}.png")
            n += 1
    print(f"완료: {OUT} 에 {n}개 파일")


if __name__ == "__main__":
    main()
