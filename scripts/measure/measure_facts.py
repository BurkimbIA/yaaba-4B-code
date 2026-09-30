"""An automatically scored fact probe, for what perplexity cannot see.

Between `Qwen3-4B` and an earlier Moore fine-tune of it, French bits per
character moved by 0.5 %, yet the fine-tune placed the capital of Burkina Faso
in Bobo-Dioulasso, in both languages. Perplexity averages surprise over a text
and cannot see one fact flip. A person reading outputs saw it; this probe does
that reading automatically.

Each item has a checkable answer and, apart from it, the known wrong answer:

    {"id": "...", "question": "...", "attendu": ["<right answer>"],
     "piege": ["<known wrong answer>"]}

`piege` is its own field because a model can be wrong in two ways: it does not
know, or it learned something else. The second is a regression. A plausible
confusion, such as a country's second city given as its capital, is what
training can install.

Scoring (`yaaba.facts.score`) handles three cases found by reading the first
outputs:

- An answer holding both the expected string and the trap ("X, no, Y", or "X
  covers more than Y") is scored `ambigu` and left to a human, since a
  substring match cannot tell a rebuttal from a retraction.
- Strings match on word boundaries, so "36" does not match "360".
- The token budget is 320, because 120 tokens cut step-by-step arithmetic
  before the result.

The score detects flips and is not a knowledge grade: a right answer phrased
differently counts as wrong, and a negated mention counts as right. It is
computed the same way before and after training, on the same items. The items
are simple and checkable, since the probe only watches that the model keeps
what it knew.

    python scripts/measure/measure_facts.py    # prints the Colab cell

The cell writes `evaluation/faits_resultats.json` on the runtime.
"""

CELL = r'''
import json
import re
import unicodedata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3_4b_base":  "Qwen/Qwen3-4B",
    "tengsoaba_4b":   "burkimbia/tengsoaba-4b",
    "tengsoaba_1.7b": "burkimbia/tengsoaba-1.7b",
    "mistral_7b":     "burkimbia/BIA-MISTRAL-7B-SACHI_merged",
}

MAX_TOKENS = __BUDGET__   # substituted from `yaaba.facts.BUDGET` at print time
facts = [json.loads(line) for line in open("evaluation/faits.jsonl", encoding="utf-8")]
print(f"{len(facts)} factual items")


def flatten(text):
    """No accents, no case: "Ouagadougou" and "ouagadougou" are the same answer."""
    text = unicodedata.normalize("NFD", str(text).casefold())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def present(needle, haystack):
    """Word boundary on numbers: "36" is not inside "360"."""
    flat = flatten(needle)
    if flat.replace(" ", "").replace(".", "").isdigit():
        return re.search(rf"(?<!\d){re.escape(flat)}(?!\d)", haystack) is not None
    return flat in haystack


def judge(answer, item):
    """juste / piege / ambigu / faux, the same four `src/yaaba/facts.py` uses.

    No order of tests is right: testing the trap first scores "X covers more
    than Y" as a regression, and testing the expected answer first scores "X,
    no, Y" as right. An answer holding both strings is scored `ambigu`.

    (The cell cannot import the package, it runs on a bare runtime, so this
    mirrors `yaaba.facts.score` with the numeric word boundary added.)
    """
    flat = flatten(answer)
    right = any(present(x, flat) for x in item["attendu"])
    trapped = any(present(x, flat) for x in item["piege"])
    if right and trapped:
        return "ambigu"       # a matcher cannot tell; a human must read it
    if right:
        return "juste"
    if trapped:
        return "piege"        # it learned something else
    return "faux"             # it does not know


everything = {}
for role, name in MODELS.items():
    print(f"\n{'='*70}\n{role}\n{'='*70}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()

    rows, counts = [], {"juste": 0, "piege": 0, "ambigu": 0, "faux": 0}
    for item in facts:
        message = [{"role": "user", "content": item["question"]}]
        # Qwen3 needs the chat template with thinking off; other model families do
        # not accept `enable_thinking`, hence the fallbacks.
        try:
            prompt = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except (TypeError, ValueError):
            try:
                prompt = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = item["question"]
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **encoded, max_new_tokens=MAX_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id)
        answer = tokenizer.decode(out[0][encoded.input_ids.size(1):],
                                  skip_special_tokens=True).strip()
        verdict = judge(answer, item)
        counts[verdict] += 1
        rows.append({**item, "reponse": answer, "verdict": verdict})
        mark = {"juste": "  ", "faux": "? ", "piege": "!!", "ambigu": "~ "}[verdict]
        print(f"{mark} {item['id']:6s} {item['question'][:52]:54s} -> {answer[:60]}",
              flush=True)

    everything[role] = {"modele": name, "compte": counts, "lignes": rows}
    print(f"\n  right {counts['juste']}/{len(facts)}   wrong {counts['faux']}   "
          f"TRAP {counts['piege']}   unclear {counts['ambigu']}", flush=True)
    del model
    torch.cuda.empty_cache()

json.dump(everything, open("evaluation/faits_resultats.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# --- the table, then the flips -------------------------------------------
print(f"\n\n{'model':18s} {'right':>7s} {'wrong':>7s} {'trap':>7s} {'unclear':>8s}")
for role, data in everything.items():
    counts = data["compte"]
    print(f"{role:18s} {counts['juste']:>7} {counts['faux']:>7} "
          f"{counts['piege']:>7} {counts['ambigu']:>8}")

# Items the base model gets right and a trained model gets wrong: the
# regressions.
base = {row["id"]: row["verdict"] for row in everything["qwen3_4b_base"]["lignes"]}
for role, data in everything.items():
    if role == "qwen3_4b_base":
        continue
    lost = [row for row in data["lignes"]
            if base.get(row["id"]) == "juste" and row["verdict"] != "juste"]
    gained = [row for row in data["lignes"]
              if base.get(row["id"]) != "juste" and row["verdict"] == "juste"]
    print(f"\n--- {role} against base: {len(lost)} lost, {len(gained)} gained")
    for row in lost:
        print(f"  LOST [{row['verdict']}] {row['question'][:56]}")
        print(f"       -> {row['reponse'][:90]}")
    for row in gained:
        print(f"  gained {row['question'][:56]}")

print("\n-> evaluation/faits_resultats.json")
'''

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from yaaba.facts import BUDGET

    print(__doc__)
    print("--- cell to paste into Colab ---")
    # The cell is a string literal and cannot import the budget, so the value is
    # substituted here and stays defined in one place.
    print(CELL.replace("__BUDGET__", str(BUDGET)))
