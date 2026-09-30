"""FLORES+ devtest chrF, French to Moore and back, for one model.

    python scripts/measure/flores.py --name qwen3-4b                        # on GPU
    python scripts/measure/flores.py --name P2+sft --start P2@checkpoint-6120 --sft sft-P2-6120-20260907-0700/checkpoint-1191
    python scripts/measure/flores.py --table                                # no GPU

FLORES+ is the one public benchmark in the paper; the other numbers come from
our own held-out texts, facts and probes. It is the only published set with
`mos_Latn`, and none of its 2,009 Moore sentences is in the training mix, so
anyone can check these numbers.

The two prompts are wordings the SFT set contains. The base model gets the same
prompts and gives the floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))
sys.path.insert(0, str(ROOT / "scripts" / "measure"))

OUT = ROOT / "evaluation" / "flores"
REPO = "openlanguagedata/flores_plus"
PROMPTS = {
    "french_to_moore": "Traduis cette phrase en mooré.\n\n{}",
    "moore_to_french": "Traduis cette phrase en français.\n\n{}",
}
SFT_URI = "s3://burkimbia-store/text/moore-assistant/sft"
RESULTS_URI = "s3://burkimbia-store/text/moore-assistant/flores"


def pairs(french: list[dict], moore: list[dict]) -> list[tuple[str, str]]:
    """(french, moore) pairs aligned by FLORES id rather than by line order."""
    by_id = {row["id"]: row["text"] for row in moore}
    missing = [row["id"] for row in french if row["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} French ids have no Moore line")
    return [(row["text"], by_id[row["id"]]) for row in french]


def first_line(reply: str) -> str:
    """The translation, without what a chat model adds after it."""
    # FLORES sentences are one line each, so anything after the first line is
    # commentary. A model that puts a preamble first loses here.
    for line in reply.strip().splitlines():
        if line.strip():
            return line.strip().strip('"«» ')
    return ""


def load_devtest() -> list[tuple[str, str]]:
    from huggingface_hub import hf_hub_download

    def rows(lang: str) -> list[dict]:
        path = hf_hub_download(REPO, f"devtest/{lang}.jsonl", repo_type="dataset")
        return [json.loads(line) for line in open(path, encoding="utf-8")]

    return pairs(rows("fra_Latn"), rows("mos_Latn"))


def translate(model, tokenizer, sources: list[str], direction: str,
              batch: int, budget: int) -> list[str]:
    """Greedy, batched, left-padded: the same decoding for every model."""
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    for start in range(0, len(sources), batch):
        texts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[direction].format(s)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for s in sources[start:start + batch]]
        encoded = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            produced = model.generate(**encoded, max_new_tokens=budget,
                                      do_sample=False,
                                      pad_token_id=tokenizer.pad_token_id)
        width = encoded.input_ids.size(1)
        outputs += [first_line(tokenizer.decode(p[width:], skip_special_tokens=True))
                    for p in produced]
        print(f"  {direction} {len(outputs)}/{len(sources)}", flush=True)
    return outputs


def score(hypotheses: list[str], references: list[str]) -> dict:
    from sacrebleu.metrics import CHRF

    return {"chrf": round(CHRF().corpus_score(hypotheses, [references]).score, 2),
            "chrf++": round(CHRF(word_order=2).corpus_score(
                hypotheses, [references]).score, 2)}


def run(name: str, start: str, sft: str | None, batch: int, budget: int) -> Path:
    import train_sft
    from evaluate_checkpoint import load

    from yaaba import s3

    work = Path("/content/flores-work") if Path("/content").exists() else OUT / "work"
    _, socle = train_sft.start_from(start, work)
    checkpoint = "base"
    if sft:
        checkpoint = str(work / "sft" / sft.replace("/", "-"))
        if not Path(checkpoint, train_sft.ADAPTER).exists():
            s3.download_dir(f"{SFT_URI}/{sft}", Path(checkpoint))
    # `load` recognises an SFT point by its folder name, and this folder is
    # renamed, so the socle check is done here.
    if sft and start != "base" and not socle:
        raise SystemExit("an SFT point on an arm needs its CPT socle")
    model, tokenizer, _ = load(checkpoint, socle)

    data = load_devtest()
    french, moore = [f for f, _ in data], [m for _, m in data]
    result = {"name": name, "start": start, "sft": sft, "sentences": len(data),
              "budget": budget, "prompts": PROMPTS}
    for direction, sources, references in (("french_to_moore", french, moore),
                                           ("moore_to_french", moore, french)):
        hypotheses = translate(model, tokenizer, sources, direction, batch, budget)
        result[direction] = {**score(hypotheses, references),
                             "hypotheses": hypotheses}
        print(f"{name} {direction}: {result[direction]['chrf']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name.replace('+', '_').replace('/', '_')}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    # Upload right away: a Colab restart erases the runtime's disk.
    s3.upload_file(path, f"{RESULTS_URI}/{path.name}")
    return path


def in_target_language(hypothesis: str, direction: str) -> bool:
    """Whether the output is in the requested language rather than left in the source one."""
    from yaaba.moore import written_in_moore

    return written_in_moore(hypothesis) == (direction == "french_to_moore")


def table() -> None:
    """chrF per direction, and the share of outputs in the language asked for."""
    print(f"{'model':16} {'fr->mos':>8} {'in mos':>7} {'mos->fr':>8} {'in fr':>7}")
    for path in sorted(OUT.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        cells = []
        for direction in ("french_to_moore", "moore_to_french"):
            hyps = r[direction]["hypotheses"]
            share = sum(in_target_language(h, direction) for h in hyps) / len(hyps)
            cells.append(f"{r[direction]['chrf']:8} {share:7.1%}")
        print(f"{r['name']:16} {' '.join(cells)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name")
    parser.add_argument("--start", default="base",
                        help="base, or a CPT arm and point: P2@checkpoint-6120")
    parser.add_argument("--sft", help="SFT run/checkpoint under the SFT prefix")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.table:
        table()
        return 0
    if not args.name:
        parser.error("--name is required to measure")
    print(run(args.name, args.start, args.sft, args.batch, args.budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
