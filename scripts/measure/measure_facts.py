"""An automatically-scored factual probe, for what perplexity cannot see.

Between `Qwen3-4B` and an earlier Moore fine-tune of it, French loses
**0.5 %** of bits per character, which reads as "nothing". And the capital of
Burkina Faso moves from Ouagadougou to Bobo-Dioulasso, in both languages.

**Perplexity measures average surprise over a text; it cannot see that one
precise fact flipped.** No threshold on the seven bits-per-character numbers
would have caught that, and it was a human who saw it while reading outputs.

This file makes that reading automatic. Twenty items with a checkable answer,
each carrying what counts as right **and what counts as the known trap**:

    {"question": "Quelle est la capitale du Burkina Faso ?",
     "attendu": ["ouagadougou"],
     "piege":   ["bobo-dioulasso"]}

## Why `piege` is a field of its own, separate from wrong

A wrong answer is wrong in one of two ways: the model does not know, or the
model learned something else. Telling them apart matters, because the second is
a regression and the first is a gap. `bobo-dioulasso` instead of `ouagadougou`
is not noise: it is the country's second city, so a *plausible* confusion that
training can install.

## Three defects found by reading its first output

None would have been visible from the total alone. It took going item by item.

**The trap was tested before the expected answer, and then after, and neither
is right.** Qwen3 base answers "the largest ocean is the Pacific Ocean. It
covers more than the Atlantic...": a right answer, marked as a trap because it
mentions the Atlantic in order to rule it out. Flipping the order fixes that
one and breaks the other: "Ouagadougou, no, Bobo-Dioulasso" would score right.
A substring matcher cannot tell a rebuttal from a retraction, so an answer
holding both strings is now scored `ambigu` and read by a human. Picking an
order silently is how the two scorers of this repo disagreed for weeks about
the same answer, each with a written justification.

**Substring search had no word boundary.** tengsoaba-1.7b answers "24 x 15 =
360", which is right, and "36" is a substring of "360". Numeric traps are now
compared with a boundary.

**120 tokens cut long-form arithmetic short.** "Pour calculer 347 + 285, on
commence par les unites : 7 + 5 = 12..." and end of output: counted wrong, while
nothing said the model had erred. Raised to 320.

These are defects of measurement, not of models, and they skewed the first table
in both directions.

## What this probe does not do

It looks for a string in the answer. That is coarse, and it errs both ways: a
right answer phrased differently counts as wrong, and an answer quoting the
right word inside a sentence that negates it counts as right. The score is
therefore not a knowledge grade, **it is a flip detector**: what matters is that
it is computed identically before and after, on the same items, and that an item
going from right to wrong is visible.

The items are deliberately simple and checkable. None calls for specialist
knowledge, and that is the point: we are not measuring the model's erudition, we
are watching that it does not lose what it knew.

    python scripts/measure/measure_facts.py    # prints the Colab cell

Writes `evaluation/faits_resultats.json` from the runtime.
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

    **No ordering is right, and picking one silently is the bug.** Testing the
    trap first scored "the Pacific Ocean covers more than the Atlantic" as a
    regression; testing the expected value first would score "Ouagadougou, no,
    Bobo-Dioulasso" as correct. A substring matcher cannot tell the two apart,
    so when both strings are present it says so.

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
        # Qwen3 needs the chat template and thinking off; other families do not
        # take `enable_thinking`, hence the ladder rather than one call.
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

# --- the table, and above all the flips ----------------------------------
print(f"\n\n{'model':18s} {'right':>7s} {'wrong':>7s} {'trap':>7s} {'unclear':>8s}")
for role, data in everything.items():
    counts = data["compte"]
    print(f"{role:18s} {counts['juste']:>7} {counts['faux']:>7} "
          f"{counts['piege']:>7} {counts['ambigu']:>8}")

# What actually matters: which items the base model gets right and a trained
# model gets wrong. That is what a regression is.
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
    # The cell is a literal, so the budget cannot be imported inside it. It is
    # substituted here instead, so the number lives in one place even though
    # it ends up running in another runtime.
    print(CELL.replace("__BUDGET__", str(BUDGET)))
