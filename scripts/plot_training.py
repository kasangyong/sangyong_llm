"""checkpoints/trainlog.jsonl로 README에 넣을 학습 그래프를 그린다.

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
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in THEMES.items():
        fig_loss(train, ev, t, OUT / f"training-curve-{name}.png")
        fig_schedule(train, t, OUT / f"lr-gnorm-{name}.png")
        fig_timeline(t, OUT / f"timeline-{name}.png")
    print(f"완료: {OUT} 에 6개 파일")


if __name__ == "__main__":
    main()
