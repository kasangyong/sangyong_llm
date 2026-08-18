"""평가 하네스. 생성 코드를 실제로 실행해서 채점한다.

두 가지를 잰다:
  1) 문법 유효율 - 생성 결과가 ast.parse를 통과하는 비율
  2) pass@k      - 문제별로 k개 생성해서 하나라도 테스트를 통과하는 비율

실행은 별도 프로세스 + 타임아웃으로 격리한다. 진짜 샌드박스(컨테이너/seccomp)는
아니다. 우리가 학습시킨 모델이 우리 기계에서 만든 코드라는 전제 아래
무한루프와 예외로부터 채점기를 지키는 수준이다.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.problems import PROBLEMS, Problem

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_SEC = 10
# 테스트가 실제로 끝까지 돌았다는 표시
SENTINEL = "__HARNESS_TESTS_COMPLETED__"


@dataclass
class RunResult:
    ok: bool
    reason: str  # "pass" | "syntax" | "assert" | "error" | "timeout"
    detail: str = ""


def is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False


def run_candidate(code: str, test: str, timeout: int = TIMEOUT_SEC) -> RunResult:
    """후보 코드 + 테스트를 별도 프로세스에서 실행한다.

    종료코드 0만 보면 안 된다. 생성 코드가 sys.exit(0)이나 os._exit(0)를
    부르면 테스트를 건너뛰고도 통과로 잡힌다. 테스트 뒤에 센티넬을 출력하게
    하고 그게 실제로 찍혔는지까지 확인한다.
    """
    full = code + "\n\n" + test + f"\nprint({SENTINEL!r})\n"
    if not is_valid_python(full):
        return RunResult(False, "syntax")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "cand.py"
        path.write_text(full, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(path)],
                capture_output=True,
                timeout=timeout,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, "timeout")

    if proc.returncode == 0:
        if SENTINEL in (proc.stdout or ""):
            return RunResult(True, "pass")
        # 종료코드는 0인데 센티넬이 없다 = 테스트에 도달하지 못했다
        return RunResult(False, "early_exit")
    err = (proc.stderr or "").strip()
    reason = "assert" if "AssertionError" in err else "error"
    return RunResult(False, reason, err.splitlines()[-1] if err else "")


def truncate_completion(prompt: str, generated: str) -> str:
    """생성된 함수 본문만 남긴다.

    모델은 함수를 끝낸 뒤에도 계속 쓴다. 들여쓰기가 풀리는 첫 최상위 줄에서
    자른다.
    """
    body = generated[len(prompt):] if generated.startswith(prompt) else generated
    lines = []
    for line in body.split("\n"):
        if line.strip() and not line[0].isspace():
            # 최상위 레벨로 돌아왔다 -> 함수 본문 끝
            break
        lines.append(line)
    return prompt + "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------- 모델 기반 평가


def load_model(ckpt_path: Path, device: str):
    import torch

    from model.transformer import ModelConfig, Transformer

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("iter", -1)


def evaluate_model(
    ckpt_path: Path,
    tokenizer_path: Path,
    k: int = 5,
    temperature: float = 0.6,
    max_new_tokens: int = 128,
    problems: list[Problem] | None = None,
):
    import torch

    from tokenizer.bpe import BPETokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(tokenizer_path)
    model, cfg, it = load_model(ckpt_path, device)
    problems = problems or PROBLEMS

    print(f"[eval] 체크포인트 iter={it}, device={device}, k={k}, T={temperature}")

    n_valid = 0
    n_gen = 0
    solved = 0
    per_problem = []

    for prob in problems:
        ids = tok.encode(prob.prompt, allow_special=False)
        if len(ids) >= cfg.max_seq_len - max_new_tokens:
            ids = ids[-(cfg.max_seq_len - max_new_tokens):]
        idx = torch.tensor([ids] * k, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                eos_id=tok.special_to_id.get("<|endoftext|>"),
            )

        any_pass = False
        reasons = []
        for row in out:
            text = tok.decode(row.tolist())
            cand = truncate_completion(prob.prompt, text)
            n_gen += 1
            if is_valid_python(cand):
                n_valid += 1
            r = run_candidate(cand, prob.test)
            reasons.append(r.reason)
            if r.ok:
                any_pass = True
        if any_pass:
            solved += 1
        per_problem.append({"name": prob.name, "solved": any_pass, "reasons": reasons})
        print(f"  {prob.name:16s} {'PASS' if any_pass else 'fail':5s}  {reasons}")

    result = {
        "checkpoint_iter": it,
        "k": k,
        "temperature": temperature,
        "n_problems": len(problems),
        "syntax_valid_rate": n_valid / max(n_gen, 1),
        "pass_at_k": solved / max(len(problems), 1),
        "per_problem": per_problem,
    }
    print(f"\n[eval] 문법 유효율 : {result['syntax_valid_rate'] * 100:.1f}% ({n_valid}/{n_gen})")
    print(f"[eval] pass@{k}      : {result['pass_at_k'] * 100:.1f}% ({solved}/{len(problems)})")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없다: {ckpt}")

    res = evaluate_model(
        ckpt, Path(args.tokenizer), k=args.k, temperature=args.temperature
    )
    Path(args.out).write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[eval] 결과 저장: {args.out}")


if __name__ == "__main__":
    main()
