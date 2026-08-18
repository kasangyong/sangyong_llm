"""학습된 모델로 코드를 생성해 본다.

  python train/sample.py --prompt "def quicksort(xs):"
  python train/sample.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.transformer import ModelConfig, Transformer
from tokenizer.bpe import END_OF_TEXT, BPETokenizer

ROOT = Path(__file__).resolve().parent.parent


def load(ckpt_path: Path, tok_path: Path, device: str):
    tok = BPETokenizer.load(tok_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_config"])
    if cfg.vocab_size != tok.vocab_size:
        raise SystemExit(
            f"어휘 크기 불일치: 체크포인트 {cfg.vocab_size} vs 토크나이저 {tok.vocab_size}"
        )
    model = Transformer(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(
        f"[sample] iter={ck.get('iter', -1)}, val_loss={ck.get('best_val', float('nan')):.4f}, "
        f"params={model.num_params():,}"
    )
    return model, tok, cfg


def generate(model, tok, cfg, prompt: str, n: int, temperature: float, top_k: int, device):
    ids = tok.encode(prompt, allow_special=False)
    budget = cfg.max_seq_len - n
    if budget <= 0:
        raise SystemExit(f"max_new_tokens({n})가 context({cfg.max_seq_len})보다 크다")
    if len(ids) > budget:
        ids = ids[-budget:]
        print(f"[sample] 프롬프트가 길어 뒤 {budget}토큰만 사용")
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=n,
        temperature=temperature,
        top_k=top_k,
        eos_id=tok.special_to_id.get(END_OF_TEXT),
    )
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "best.pt"))
    ap.add_argument("--tokenizer", default=str(ROOT / "tokenizer" / "tokenizer.json"))
    ap.add_argument("--prompt", default="def quicksort(xs):\n")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"체크포인트가 없다: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load(ckpt, Path(args.tokenizer), device)

    if args.interactive:
        print("프롬프트를 입력하세요. 빈 줄이면 종료.")
        while True:
            try:
                line = input("\n>>> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                break
            text = generate(
                model, tok, cfg, line + "\n", args.max_new_tokens,
                args.temperature, args.top_k, device,
            )
            print("-" * 60)
            print(text)
        return

    for i in range(args.num_samples):
        text = generate(
            model, tok, cfg, args.prompt, args.max_new_tokens,
            args.temperature, args.top_k, device,
        )
        print(f"\n{'=' * 60}\n샘플 {i + 1}/{args.num_samples}\n{'=' * 60}")
        print(text)


if __name__ == "__main__":
    main()
