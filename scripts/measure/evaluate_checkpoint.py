"""Measure one checkpoint, and decide which of the five to keep.

    python scripts/measure/evaluate_checkpoint.py --checkpoint base      # on GPU
    python scripts/measure/evaluate_checkpoint.py --checkpoint /path/checkpoint-1094
    python scripts/measure/evaluate_checkpoint.py --compare              # here

Three metric families, because one does not suffice: a perplexity once dropped
70 % while the country's capital moved from Ouagadougou to Bobo-Dioulasso.
Bits per character on seven held-out capabilities, twenty
automatically scored facts, twenty-six generative probes.

Two stop signals: Moore stops improving, or a fact the base model knew
is lost. The best checkpoint minimises Moore without crossing either, and past
the first signal a better number no longer measures the same thing.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from yaaba import moore, s3, uncertainty  # noqa: E402
from yaaba.facts import BUDGET, VERDICTS, budget_of, score  # noqa: E402

EVAL = ROOT / "evaluation"
POINTS = EVAL / "points"
CONFIG = json.loads((ROOT / "configs" / "cpt.json").read_text(encoding="utf-8"))
BASE, WINDOW = CONFIG["base"], CONFIG["fenetre"]
POINTS_URI = "s3://burkimbia-store/text/moore-assistant/cpt/points"
# The destination is a PARAMETER of `measure`, not only this constant: a second
# stage writing here would pollute the CPT curve.

SHOULD_DROP = ("moore_humain", "francais_parallele")
SHOULD_HOLD = ("anglais", "francais", "code", "maths")
# The CPT exit criterion: these four may lose at most this much against the
# base point. It was printed as a table and left to the eye; a criterion the
# code does not compute is not a criterion.
HOLD_TOLERANCE = 0.03
WATCH_APART = ("moore_whisper",)

# A note, not a stop signal. `keepable` cuts an arm's curve at the first
# mark, so leaving this one in the same list made "we refuse to compare"
# read as "the model regressed", and every arm came back with nothing
# keepable the day six points moved to another budget.
NOT_COMPARABLE = ("facts not comparable: another budget, "
                  "or another set of items")


def overlap(a: str, b: str) -> float:
    """Word overlap, to turn the model echoing the question into a number."""
    first, second = set(a.casefold().split()), set(b.casefold().split())
    return len(first & second) / max(len(first), 1)


# An SFT point trained on a CPT arm, by the name the notebook gives it:
# `C-4820-checkpoint-1191`. What makes it recognisable is the arm and its CPT
# step before `checkpoint-`, which a CPT point (`checkpoint-6005`) never has.
POINT_DE_BRAS = re.compile(r"^(?:A|B|C|P|P2)-\d+-checkpoint-\d+$")


def load(checkpoint: str, socle: str | None = None):
    """The base, then `socle`, then `checkpoint` merged in, and the name.

    Merging rather than keeping the adapter attached makes generation as fast as
    the base and is what every measurement here does.

    **`socle` is not optional for an SFT point trained on an arm.** The SFT
    stage merges the arm's CPT adapter into the base and trains a fresh adapter
    **on top of that**, so its weights are relative to `base + CPT`. Loading it
    onto the bare base applies an adapter to a model it was never trained
    against: that once gave `A + SFT` 4.7883 bits per character of Moore, worse than the base itself and four times the 1.2759 of `A` alone.
    It produced numbers, not an error, which is why the guard below exists.
    """
    if socle is None and POINT_DE_BRAS.match(Path(checkpoint).name):
        raise SystemExit(
            f"{Path(checkpoint).name} is an SFT point trained on a CPT arm, and "
            "no `socle` was given. Measuring it on the bare base applies the "
            "adapter to a model it was never trained against. Pass the arm's "
            "CPT adapter, which `train_sft.start_from` returns.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
                                                 device_map="auto")
    name = "base"
    if socle or checkpoint != "base":
        from peft import PeftModel

    if socle:
        model = PeftModel.from_pretrained(model, socle).merge_and_unload()
    if checkpoint != "base":
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
        name = Path(checkpoint).name
    model.eval()
    return model, tokenizer, name


def generate(model, tokenizer, prompt: str, budget: int,
             heat: float = 0.0, seed: int | None = None) -> tuple[str, bool]:
    """The reply, and whether it finished on its own rather than being cut.

    Module level, not a closure inside `measure`: the SFT stage asks a model the
    same kind of question outside a measurement, and a second copy of this would
    drift from the first exactly where it matters, on `enable_thinking`.

    `heat` is the one decoding knob, and it exists for one question. The
    opening check (`moore.speaks_to_each`) rejects a point whose answers all open the same way, measured greedily, and
    greedy takes the argmax at every step: two questions sharing the start of
    their distribution share their opening BY CONSTRUCTION. Re-asking the same
    twelve questions at the same point with sampling on tells which it is. If
    variety comes back, the check measures the decoder; if it stays at one
    opening out of twelve, it is in the weights.

    Bits per character are blind to generation because they are teacher-forced,
    and the opening check has the opposite blind spot.

    0.0 is neutral and keeps greedy, so no stored measurement moves. Above it,
    `seed` is required: a draw that cannot be replayed is not a measurement.

    **`seed` must differ from one question to the next**, and the caller owns
    that. This function seeds the global generator on every call, so passing the
    same seed to twelve questions makes them consume the same random stream:
    wherever their distributions are close, the same draw picks the same token,
    and the twelve answers open alike **because they were seeded alike**. That is
    a property of the seeding, not of the model, and it once produced a full
    collapse (`Yaa tɩ` twelve times at `T=0.7`) that had nothing to do with the
    model. Derive a per-question seed, `seed + index`.
    """
    import torch

    if heat and seed is None:
        raise ValueError("sampling without a seed cannot be replayed")
    if heat:
        torch.manual_seed(seed)

    message = [{"role": "user", "content": prompt}]
    try:
        # Qwen3's thinking mode is on by default and hurts: it reasons in
        # Chinese and spends the whole token budget.
        text = tokenizer.apply_chat_template(message, tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
    except (TypeError, ValueError):
        text = tokenizer.apply_chat_template(message, tokenize=False,
                                             add_generation_prompt=True)
    encoded = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=budget,
                                do_sample=bool(heat),
                                # transformers ignores `temperature` when
                                # `do_sample` is false, silently. Passing None
                                # keeps it from looking like it applies.
                                temperature=heat or None,
                                top_p=0.95 if heat else None,
                                pad_token_id=tokenizer.eos_token_id)
    produced = output[0][encoded.input_ids.size(1):]
    return (tokenizer.decode(produced, skip_special_tokens=True).strip(),
            len(produced) < budget)


def measure(checkpoint: str, out: Path, upload: bool,
            uri: str = POINTS_URI, socle: str | None = None) -> int:
    import torch  # noqa: F401  (kept for the nll loop below)

    model, tokenizer, name = load(checkpoint, socle)
    print(f"checkpoint: {name}  ({checkpoint})", flush=True)
    if socle:
        print(f"  socle: {socle}", flush=True)

    by_capability: dict[str, list[str]] = {}
    for line in (EVAL / "tenu_a_lecart.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        by_capability.setdefault(record["capacite"], []).append(record["text"])

    metrics = {}
    for capability, texts in by_capability.items():
        # Kept per text, and not only summed: an interval needs the sample, and
        # re-running the GPU to get one would cost more than storing 200 pairs.
        # The order is the order of the held-out file, which is what makes a
        # later comparison paired.
        per_text: list[tuple[float, int]] = []
        nll, n_tokens, n_chars = 0.0, 0, 0
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=WINDOW).input_ids.to(model.device)
            if ids.size(1) < 2:
                per_text.append((0.0, 0))     # placeholder: keeps texts aligned
                continue
            with torch.no_grad():
                loss = model(ids, labels=ids).loss
            predicted = ids.size(1) - 1        # the first token is not predicted
            chars = len(tokenizer.decode(ids[0], skip_special_tokens=True))
            per_text.append((loss.item() * predicted, chars))
            nll += loss.item() * predicted
            n_tokens += predicted
            n_chars += chars
        spread = uncertainty.interval(per_text)
        metrics[capability] = {
            # comparable across tokenizers, unlike perplexity
            "bits_par_caractere": round(nll / math.log(2) / n_chars, 4),
            "intervalle": [round(spread.low, 4), round(spread.high, 4)],
            "perplexite": round(math.exp(nll / n_tokens), 3),
            "tokens": n_tokens, "caracteres": n_chars,
            "par_texte": [[round(v, 4), c] for v, c in per_text],
        }
        current = metrics[capability]
        print(f"  {capability:20s} bpc {current['bits_par_caractere']:>7.4f}   "
              f"ppl {current['perplexite']:>9.2f}", flush=True)

    facts, tally = ask_facts(model, tokenizer)

    probes = []
    for line in (EVAL / "sondes.jsonl").open(encoding="utf-8"):
        probe = json.loads(line)
        probes.append({**probe,
                       "reponse": generate(model, tokenizer,
                                          probe["prompt"], 160)[0]})
    echoes = [overlap(p["prompt"], p["reponse"])
              for p in probes if p["famille"] == "conversation_moore"]
    echo = round(sum(echoes) / len(echoes), 3) if echoes else None
    print(f"  Moore question/answer echo: {echo:.0%}" if echo is not None
          else "  no conversation probe", flush=True)

    result = {"point": name, "chemin": checkpoint, "socle": socle,
              "fenetre": WINDOW,
              "decodage": "greedy", "mesures": metrics, "faits": facts,
              "faits_compte": tally, "budget_faits": BUDGET,
              "echo_conversation_moore": echo, "sondes": probes}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n-> {path}")

    if upload:
        s3.upload_file(path, f"{uri}/{name}.json")
        print(f"-> {uri}/{name}.json")
    return 0


def ask_facts(model, tokenizer) -> tuple[list[dict], dict]:
    """Every item of the probe, answered and scored. Shared by `measure` and
    `refacts`."""
    facts = []
    for line in (EVAL / "faits.jsonl").open(encoding="utf-8"):
        fact = json.loads(line)
        reply, complete = generate(model, tokenizer, fact["question"], BUDGET)
        facts.append({"id": fact["id"], "domaine": fact["domaine"],
                      "note": score(reply, fact, complete), "reponse": reply,
                      "complet": complete})
    tally = {kind: sum(f["note"] == kind for f in facts) for kind in VERDICTS}
    print(f"\n  facts: {tally['juste']}/{len(facts)} right, "
          f"{tally['piege']} trapped, "
          f"{tally['ambigu']} cut or unclear, {tally['faux']} not known",
          flush=True)
    return facts, tally


def refacts(checkpoint: str, out: Path, upload: bool,
            socle: str | None = None, point_name: str | None = None) -> int:
    """Re-answer the probe into an existing point, leaving the rest alone.

    The bits per character of a point cost minutes and do not depend on the
    generation budget; the answers cost seconds and do. Raising `BUDGET` from
    128 to 320 therefore does not justify re-running `measure`, which would
    recompute 200 held-out texts in order to change twenty strings.

    The point must already exist: this repairs a measurement, it does not make
    one. `budget_faits` is written next to the answers, because a tally is only
    comparable to another tally taken at the same budget.

    **`point_name` exists because the checkpoint directory is not the point.**
    The adapter downloaded from S3 sits in a folder called `checkpoint-218`,
    while the point that describes it is `P2-6120-checkpoint-218.json`. Deriving
    the file name from the folder made this look for a point that does not
    exist.

    **And `socle` is not optional for an SFT point trained on an arm**, for the
    same reason as in `measure`: without it the adapter is applied to a model it
    was never trained against and the run returns numbers instead of an error.
    The guard in `load` does not catch it here, because the name it
    recognises is the point's, not the folder's.
    """
    name = "base" if checkpoint == "base" else (point_name or Path(checkpoint).name)
    path = out / f"{name}.json"
    if not path.exists():
        print(f"{path} does not exist: measure the point first", file=sys.stderr)
        return 2
    point = json.loads(path.read_text(encoding="utf-8"))
    was, over = point["faits_compte"]["juste"], len(point["faits"])

    model, tokenizer, _ = load(checkpoint, socle)
    print(f"checkpoint: {name}  ({checkpoint})  facts only, budget {BUDGET}",
          flush=True)
    point["faits"], point["faits_compte"] = ask_facts(model, tokenizer)
    point["budget_faits"] = BUDGET
    path.write_text(json.dumps(point, ensure_ascii=False, indent=1), encoding="utf-8")
    # Two denominators on purpose: the probe grew from 20 items to 52, and
    # printing one of them would hide that the item set changed.
    print(f"\n  {was}/{over} -> "
          f"{point['faits_compte']['juste']}/{len(point['faits'])}")
    print(f"-> {path}")

    if upload:
        s3.upload_file(path, f"{POINTS_URI}/{path.stem}.json")
        print(f"-> {POINTS_URI}/{path.stem}.json")
    return 0


def order(name: str) -> tuple[int, int]:
    """Base first, then checkpoints by step. Numerically, never lexicographically.

    The step is what follows `checkpoint-`, not every digit in the name. A point
    measured from a directory carrying its run, `A-20260831-1024-checkpoint-219`,
    used to key on 202608311024219: the ordering became run timestamp first and
    step second, by accident of how many digits each step happened to have.
    """
    if name == "base":
        return (0, 0)
    _, marker, tail = name.rpartition("checkpoint-")
    digits = "".join(c for c in (tail if marker else name) if c.isdigit())
    return (1, int(digits) if digits else 0)


def arm_of(point: dict) -> str:
    """Which arm produced this point, read from the run directory in its path.

    `/content/cpt/A-20260831-1024/checkpoint-6005` -> `A`. Points from every arm
    land in one directory, so without this the curve of one arm and the curve of
    another get spliced into a single sequence.
    """
    if point.get("point") == "base":
        return "base"
    parts = Path(point.get("chemin", "")).parts
    for part in reversed(parts):
        if "-" in part and not part.startswith("checkpoint"):
            return part.split("-", 1)[0]
    return "?"


def asked_the_same(point: dict) -> tuple[int, frozenset[str]]:
    """What a fact tally has to share with another one to be its comparable."""
    return budget_of(point), frozenset(f["id"] for f in point["faits"])


def same_fact_budget(first: dict, second: dict) -> bool:
    """Whether two points scored their facts the same way.

    The test is agreement, not recency: two points both scored at 48 compare to
    each other perfectly well. Only a mixed pair measures the budget instead of
    the models.

    It used to test whether `complet` was *present*, which separates 48 from
    128 and **not 128 from 320** -- both carry the field. Once six points were
    re-answered at 320 against twenty-five still at 128, the guard passed them
    all and the arm ranking moved a whole arm's kept point by three epochs.

    **The item set is the second half of the same question**, and it was
    missing until the probe was widened from 20 items to 52. A point that answered 20 and a point that answered 52 put 13 and 38
    in the same column of the same table, and the second number looks like a
    gain. `lost` never had the defect, because it is keyed by item id; the
    tally did.
    """
    return asked_the_same(first) == asked_the_same(second)


def forgotten(known: set[str], verdicts: dict[str, str]) -> list[str]:
    """Items the start answered and this point no longer answers at all.

    Reported, never a rejection. It is the count that carries the
    conclusion, with the interval the 52-item probe makes readable;
    a per-item veto rejected every checkpoint on items that come back later in
    the same run.
    """
    return sorted(i for i in known if verdicts.get(i) == "faux")


def marks_for(base: dict, earlier: dict, point: dict, known: set[str]) -> list[str]:
    """Every stop signal `point` crosses, against its own predecessor and base.

    `earlier` is the previous point OF THE SAME ARM, never of the sorted list.
    Shared by the curve and the arm ranking so the two cannot drift.
    """
    marks = []
    current = point["mesures"]["moore_humain"]["bits_par_caractere"]
    previous = earlier["mesures"]["moore_humain"]["bits_par_caractere"]

    # A rise the sample supports, not any rise: comparing point estimates
    # makes the noise floor the stop criterion.
    rise = gap(earlier, point, "moore_humain")
    if rise is None:
        if current >= previous:
            marks.append(f"overfitting? (bpc {previous:.4f} -> {current:.4f}, "
                         "no interval: re-measure)")
    elif rise.low > 0:
        marks.append(f"overfitting (bpc {rise.value:+.4f} "
                     f"[{rise.low:+.4f}, {rise.high:+.4f}])")

    # An item the base got right and this point answers DIFFERENTLY: it learned
    # something else, which is the regression. `faux`, it no longer
    # answers, is counted by `forgotten` and reported beside the verdict, not
    # here: measured on five points it flips back and forth inside one run, so
    # as a per-item veto it rejected every checkpoint on the stability of a
    # binary item at greedy decoding. `ambigu` is kept apart too: a
    # substring matcher could not tell, so it is a question for a human.
    verdicts = {f["id"]: f["note"] for f in point["faits"]}
    if not same_fact_budget(base, point):
        marks.append(NOT_COMPARABLE)
        verdicts = {}
    lost = sorted(i for i in known if verdicts.get(i) == "piege")
    if lost:
        marks.append(f"facts learned otherwise: {', '.join(lost)}")

    # bpc measures cost: it going UP is the loss. Comparing the wrong way round
    # would silently pass every degraded checkpoint.
    for capability in SHOULD_HOLD:
        was = base["mesures"][capability]["bits_par_caractere"]
        now = point["mesures"][capability]["bits_par_caractere"]
        if (now - was) / was > HOLD_TOLERANCE:
            marks.append(f"{capability} +{100 * (now - was) / was:.1f} % "
                         f"(over {100 * HOLD_TOLERANCE:.0f} %)")
    return marks


def answer_key(arm: str, point_name: str) -> tuple[str, str]:
    """(arm, training step): what identifies a point across two file families.

    The name alone does NOT: the same checkpoint is on disk as
    `218-checkpoint-218` in the answers and `P2-6120-checkpoint-218` in the
    points, and matching on the name silently found no answer for exactly the
    points the gate is about. Both families record the arm (`depart` there,
    the run directory here) and the step, so the key is derived, not guessed.
    """
    return arm, point_name.rpartition("checkpoint-")[2] or point_name


def criterion_answers(directory: Path = EVAL) -> dict[tuple[str, str], list[str]]:
    """The twelve criterion answers of each point, by (arm, step).

    They are not in the point files: `measure` stores three conversation probes,
    the twelve held turns are generated by the SFT notebook and land in
    `evaluation/reponses-*.jsonl`. Reading them here is what lets the
    point SELECTION apply the collapse checks, and not only the notebook that
    prints them.

    **Only answers to the CURRENT twelve questions count.** Eight of the twelve
    files on disk answer earlier question sets, and taking them at the glob
    makes the gate read a point's old answers and clear it. The test
    is the question itself, not the file's date or name. The twelve come from
    `questions_tenues.jsonl` in the gated evaluation set.
    """
    held = {" ".join(sorted(moore._WORD.findall(q["question"].casefold())))
            for q in (json.loads(line) for line in
                      (directory / "questions_tenues.jsonl").open(encoding="utf-8"))}
    by_point: dict[tuple[str, str], list[str]] = {}
    for path in sorted(directory.glob("reponses-*.jsonl")):
        for line in path.open(encoding="utf-8"):
            answered = json.loads(line)
            flat = " ".join(sorted(moore._WORD.findall(answered["question"].casefold())))
            if flat not in held:
                continue
            arm = answered.get("depart", "").partition("@")[0] or "?"
            key = answer_key(arm, answered["point"])
            by_point.setdefault(key, []).append(answered["obtenu"])
    return by_point


def collapsed(point: dict, answers: dict[tuple[str, str], list[str]]) -> str | None:
    """Why this point is not keepable for want of variety, or None.

    Two rules, because one opening count sees only the first two words:

    - `moore.speaks_to_each`: no single opening may answer a majority of the
      questions;
    - `moore.answers_each`: no complete answer may be given to two questions.

    Neither replaces the other. On two decoding passes of one checkpoint they
    disagreed on both: greedy gives twelve distinct sentences under one opening
    (the first rejects, the second clears), and `T=1.0` answers four different
    questions with the same sentence (the first clears, the second rejects). A gate that
    passes a model repeating one sentence four times out of twelve does not
    measure what it claims.

    Separate from `marks_for` on purpose, because this does NOT cut the curve:
    collapse loosens as training goes on, the opposite of overfitting, so
    cutting here would throw away the better points that follow.

    A point with no answers to the current questions is **kept and flagged**,
    not rejected: a missing measurement is not evidence of collapse. It is the
    same treatment as `NOT_COMPARABLE`, and it is the weak spot of this gate,
    so the caller prints the flag next to the point it chose.
    """
    said = answers.get(answer_key(arm_of(point), point["point"]))
    if not said:
        return None
    if not moore.answers_each(said):
        distinct, biggest = moore.repeats(said)
        return (f"collapsed: {distinct} distinct answer(s) for {len(said)} "
                f"questions, one is given {biggest} times")
    if moore.speaks_to_each(said):
        return None
    distinct, biggest = moore.openings(said)
    return (f"collapsed: {distinct} opening(s) for {len(said)} questions, "
            f"the largest answers {biggest} of them")


def unchecked(point: dict, answers: dict[tuple[str, str], list[str]]) -> bool:
    """Whether the collapse checks could not run on this point for want of
    answers."""
    return not answers.get(answer_key(arm_of(point), point["point"]))


def keepable(base: dict, curve: list[dict],
             answers: dict[str, list[str]] | None = None
             ) -> tuple[dict | None, str | None, list[tuple[str, str]]]:
    """The point to keep from one arm's curve, where it was cut, what it skipped.

    Past the first stop signal a better number no longer measures the same
    thing, so the best point is the lowest Moore bpc *before* it, not
    the lowest overall and not the last one measured.

    The collapse checks are applied apart: a collapsed point is dropped from the eligible ones
    and the curve CONTINUES past it. Passing no `answers` skips the rule rather
    than inventing a verdict, and every skip is returned so the caller prints
    it.
    """
    known = {f["id"] for f in base["faits"] if f["note"] == "juste"}
    eligible: list[dict] = []
    skipped: list[tuple[str, str]] = []
    earlier = base
    best = (lambda: min(eligible,
                        key=lambda p: p["mesures"]["moore_humain"]["bits_par_caractere"])
            if eligible else None)
    for point in curve:
        if [m for m in marks_for(base, earlier, point, known)
                if m != NOT_COMPARABLE]:
            return best(), point["point"], skipped
        why = collapsed(point, answers) if answers is not None else None
        if why:
            skipped.append((point["point"], why))
        else:
            eligible.append(point)
        earlier = point
    return best(), None, skipped


def gap(before: dict, after: dict, capability: str) -> uncertainty.Interval | None:
    """Paired difference between two points, or None if either predates `par_texte`."""
    pairs = []
    for point in (before, after):
        measured = point["mesures"].get(capability, {})
        if "par_texte" not in measured:
            return None
        pairs.append([tuple(x) for x in measured["par_texte"]])
    try:
        return uncertainty.difference(*pairs)
    except ValueError:
        return None


def compare(pattern: str) -> int:
    files = sorted(glob.glob(pattern), key=lambda f: order(Path(f).stem))
    if not files:
        print(f"no checkpoint measured: {pattern}", file=sys.stderr)
        return 2
    points = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    names = [p["point"] if p["point"] == "base"
             else f"{arm_of(p)}/{p['point'].removeprefix('checkpoint-')}"
             for p in points]
    width = max(11, max(len(n) for n in names) + 2)

    print("BITS PER CHARACTER, held out\n" + "-" * (22 + width * len(points)))
    print(f"{'capability':22s}" + "".join(f"{n:>{width}s}" for n in names))
    for group, label in ((SHOULD_DROP, "should drop"),
                         (WATCH_APART, "watch apart"),
                         (SHOULD_HOLD, "should hold")):
        print(f"  [{label}]")
        for capability in group:
            row = f"  {capability:20s}"
            for point in points:
                row += f"{point['mesures'][capability]['bits_par_caractere']:>{width}.4f}"
            print(row)

    print(f"\n{'facts right':22s}"
          + "".join(f"{p['faits_compte']['juste']:>{width}d}" for p in points))
    print(f"{'of which trapped':22s}"
          + "".join(f"{p['faits_compte']['piege']:>{width}d}" for p in points))

    # Each point against the base, on the SAME held-out texts. Two independent
    # intervals routinely overlap while the paired difference is nowhere near
    # zero, so the table above cannot answer "is this gap real" and this can.
    reference = points[0]
    paired = [p for p in points[1:]
              if all("par_texte" in p["mesures"].get(c, {})
                     and "par_texte" in reference["mesures"].get(c, {})
                     for c in SHOULD_DROP)]
    if not paired:
        print("\n(no interval: these points predate `par_texte`; re-measure to"
              " get one)")
    for point in paired:
        print(f"\nAGAINST {reference['point']}, paired, 95 %  "
              f"(negative is better: bpc is a cost)")
        for capability in SHOULD_DROP + WATCH_APART + SHOULD_HOLD:
            against_base = gap(reference, point, capability)
            if against_base is None:
                print(f"  {capability:20s} not measured on the same texts")
                continue
            settled = "" if against_base.low <= 0 <= against_base.high else "  *"
            print(f"  {capability:20s} {against_base.value:>+8.4f} "
                  f"[{against_base.low:>+7.4f}, {against_base.high:>+7.4f}]{settled}")
        print("  * the interval excludes zero: the gap holds on this sample.")
        print("    No star does not mean no difference; it means a difference")
        print("    that 200 texts do not separate.")

    base = points[0]
    known = {f["id"] for f in base["faits"] if f["note"] == "juste"}
    print(f"\nSIGNALS  (reference {base['point']}, {len(known)} facts right)")

    # The overfitting signal reads a curve, and a curve belongs to one arm. All
    # arms write their points into the same directory, so pairing by position in
    # the sorted list splices B's last point onto C's first. The
    # predecessor of a point is the previous point OF ITS OWN ARM.
    previous_of: dict[str, dict] = {}
    first_crossed = None
    for point in points[1:]:
        arm = arm_of(point)
        earlier = previous_of.get(arm)
        previous_of[arm] = point
        if earlier is None:
            earlier = base          # the first point of an arm is judged against base
        current = point["mesures"]["moore_humain"]["bits_par_caractere"]
        previous = earlier["mesures"]["moore_humain"]["bits_par_caractere"]
        marks = []
        # A rise the sample supports, not any rise: comparing point
        # estimates makes the noise floor the stop criterion.
        rise = gap(earlier, point, "moore_humain")
        if rise is None:
            if current >= previous:
                marks.append(f"overfitting? (bpc {previous:.4f} -> {current:.4f}, "
                             "no interval: re-measure)")
        elif rise.low > 0:
            marks.append(f"overfitting (bpc {rise.value:+.4f} "
                         f"[{rise.low:+.4f}, {rise.high:+.4f}])")
        # Only `piege` rejects; `faux` is counted and printed.
        verdicts = {f["id"]: f["note"] for f in point["faits"]}
        if not same_fact_budget(base, point):
            # Counting these would measure the token budget, not the model.
            marks.append(NOT_COMPARABLE)
            verdicts = {}
        lost = sorted(i for i in known if verdicts.get(i) == "piege")
        gone = forgotten(known, verdicts)
        unclear = sorted(i for i in known if verdicts.get(i) == "ambigu")
        if lost:
            marks.append(f"facts learned otherwise: {', '.join(lost)}")
        if gone:
            print(f"  {point['point']:22s} no longer answers {len(gone)} of "
                  f"{len(known)}: {', '.join(gone)}")
        if unclear:
            print(f"  {point['point']:22s} unclear, read them: {', '.join(unclear)}")
        # bpc measures cost: it going UP is the loss. Comparing the wrong way
        # round would silently pass every degraded checkpoint.
        for capability in SHOULD_HOLD:
            was = base["mesures"][capability]["bits_par_caractere"]
            now = point["mesures"][capability]["bits_par_caractere"]
            if (now - was) / was > HOLD_TOLERANCE:
                marks.append(f"{capability} +{100 * (now - was) / was:.1f} % "
                             f"(over {100 * HOLD_TOLERANCE:.0f} %)")
        print(f"  {point['point']:22s} {'; '.join(marks) if marks else 'none'}")
        if marks and first_crossed is None:
            first_crossed = point["point"]

    # Checkpoints after the first signal are excluded even when one shows a
    # better number: past the signal the number no longer measures the same
    # thing.
    # The collapse checks apply HERE TOO, not only in `keepable`. The two paths
    # answer the same question, and when only one applied them `--compare`
    # returned a checkpoint that `--arms` rejected as collapsed.
    said = criterion_answers()
    eligible = []
    for point in points[1:]:
        if first_crossed and order(point["point"]) >= order(first_crossed):
            break
        why = collapsed(point, said)
        if why:
            print(f"  {point['point']:22s} {why}, not keepable")
            continue
        if unchecked(point, said):
            print(f"  {point['point']:22s} no answer to the current twelve "
                  "questions, collapse checks not applied")
        eligible.append(point)
    if not eligible:
        # On stderr, because it is the REASON for the non-zero exit. Printed on
        # stdout it left a caller that checks the return code with a traceback
        # about the launcher and an empty stderr.
        print("\nNO eligible checkpoint: the first one already crosses a signal.",
              file=sys.stderr)
        return 1

    best = min(eligible,
               key=lambda p: p["mesures"]["moore_humain"]["bits_par_caractere"])
    start = base["mesures"]["moore_humain"]["bits_par_caractere"]
    end = best["mesures"]["moore_humain"]["bits_par_caractere"]
    print(f"\nBEST: {best['point']}   Moore {start:.4f} -> {end:.4f} bpc  "
          f"({100 * (end - start) / start:+.1f} %)")
    if first_crossed:
        print(f"  (from {first_crossed} onward excluded: signal crossed)")
    return 0


def arms(pattern: str, against: str = "A") -> int:
    """Rank the arms against each other, on their last measured point.

    `compare` reads a *curve*: how one arm evolves, and where it should stop.
    This reads a *choice*: which arm to keep. They need different pairings, and
    conflating them is what made one run's table splice another run's points
    into its own curve.

    Every gap is paired on the same held-out texts, because two independent
    intervals routinely overlap while the paired difference is nowhere near zero.
    """
    files = sorted(glob.glob(pattern), key=lambda f: order(Path(f).stem))
    points = [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]
    if not points:
        print(f"no checkpoint measured: {pattern}", file=sys.stderr)
        return 2

    base = next((p for p in points if p["point"] == "base"), None)
    if base is None:
        print("no base point: nothing to judge an arm against", file=sys.stderr)
        return 2

    curves: dict[str, list[dict]] = {}
    for point in points:
        if point["point"] != "base":
            curves.setdefault(arm_of(point), []).append(point)

    # The point to keep, not the last one measured. An arm's Moore can improve
    # to its final epoch and still be worse there than three epochs in, which is
    # the whole reason for measuring a curve.
    last: dict[str, dict] = {}
    cut_at: dict[str, str] = {}
    rejected: dict[str, str] = {}
    said = criterion_answers()
    for arm, curve in curves.items():
        chosen, crossed, skipped = keepable(base, curve, said)
        for name, why in skipped:
            print(f"  {arm}/{name.removeprefix('checkpoint-')}: {why}")
        if chosen is not None and unchecked(chosen, said):
            print(f"  {arm}/{chosen['point'].removeprefix('checkpoint-')}: "
                  "no answer to the current twelve questions, "
                  "collapse checks not applied")
        if chosen is None:
            # Every arm may fail the exit criterion, which is itself a finding
            # and not a reason to print nothing: the numbers are still
            # the comparison. Show the last point measured, and say what it is.
            last[arm] = curve[-1]
            rejected[arm] = crossed or "?"
        else:
            last[arm] = chosen
            if crossed:
                cut_at[arm] = crossed

    if against not in last:
        print(f"arm {against} has no keepable point; arms present: "
              f"{', '.join(sorted(last)) or 'none'}", file=sys.stderr)
        return 2

    reference = last[against]
    others = [a for a in sorted(last) if a != against]
    print(f"ARMS, best keepable point of each, against {against} "
          f"({reference['point']})")
    print("  a gap is paired on the same held-out texts; * means the interval "
          "excludes zero")

    # A curve of one or two points cannot show a peak, so "no signal crossed"
    # means "not looked for" and must not read as "none happened".
    print()
    for arm in [against] + others:
        curve = curves[arm]
        span = (f"{order(curve[0]['point'])[1]:,} to "
                f"{order(curve[-1]['point'])[1]:,}")
        blind = ("  NO CURVE: one or two points cannot show a peak"
                 if len(curve) < 3 else "")
        if arm in rejected:
            state = (f"NOTHING KEEPABLE: {rejected[arm]} already crosses a "
                     f"signal, showing {last[arm]['point']}")
        else:
            state = f"keeping {last[arm]['point']}"
            if arm in cut_at:
                state += f", curve cut at {cut_at[arm]}"
        print(f"  {arm:4s} {len(curve)} point(s) measured, steps {span}, "
              f"{state}{blind}")

    width = 22
    print()
    # The columns are NOT all at the same rank: a rejected arm shows its last
    # point and a kept one shows its peak, which can be epochs apart. Naming the
    # point under each arm puts that where the numbers are, instead of three
    # lines above them where a reader compares 1.97 at epoch 0.2 against 1.28 at
    # epoch 5 and calls the first arm worse.
    shown = [against] + others
    print(f"{'capability':22s}" + "".join(f"{a:>{width}s}" for a in shown))
    print(f"{'':22s}" + "".join(f"{last[a]['point']:>{width}s}" for a in shown))
    for group in (SHOULD_DROP, WATCH_APART, SHOULD_HOLD):
        for capability in group:
            row = f"  {capability:20s}"
            row += f"{reference['mesures'][capability]['bits_par_caractere']:>{width}.4f}"
            for arm in others:
                measured = last[arm]["mesures"][capability]["bits_par_caractere"]
                found = gap(reference, last[arm], capability)
                mark = "" if found is None or found.low <= 0 <= found.high else "*"
                row += f"{measured:>{width - 1}.4f}{mark}"
            print(row)

    # Facts and echo are not bits per character and are reported apart: the echo
    # is invisible to bpc, and the fact probe exists because a perplexity did
    # not see a fact flip.
    print()
    row = f"  {'facts right':20s}"
    for arm in [against] + others:
        point = last[arm]
        cell = (str(point["faits_compte"]["juste"]) if base is None
                or same_fact_budget(base, point) else "n/c")
        row += f"{cell:>{width}s}"
    print(row)
    row = f"  {'Moore echo':20s}"
    for arm in [against] + others:
        echo = last[arm].get("echo_conversation_moore")
        row += f"{'' if echo is None else f'{echo:.0%}':>{width}s}"
    print(row)
    # Reported, never a rejection: what the start answered and this
    # point does not answer any more. The count carries the conclusion, the
    # per-item veto rejected every checkpoint on items that come back.
    if base is not None:
        known = {f["id"] for f in base["faits"] if f["note"] == "juste"}
        for arm in [against] + others:
            point = last[arm]
            if not same_fact_budget(base, point):
                continue
            gone = forgotten(known, {f["id"]: f["note"] for f in point["faits"]})
            if gone:
                print(f"  {point['point']} no longer answers {len(gone)} of "
                      f"{len(known)}: {', '.join(gone)}")
    if base is not None and any(not same_fact_budget(base, last[a])
                                for a in [against] + others):
        print("  n/c: facts measured at another token budget, not comparable "
              "; re-measure that arm")

    print()
    print(f"PAIRED against {against}, 95 %  (negative is better)")
    for arm in others:
        found = gap(reference, last[arm], "moore_humain")
        if found is None:
            print(f"  {arm:4s} moore_humain   not measured on the same texts")
            continue
        verdict = ("better" if found.high < 0 else
                   "worse" if found.low > 0 else
                   "not separated by these texts")
        print(f"  {arm:4s} moore_humain   {found.value:>+8.4f} "
              f"[{found.low:>+7.4f}, {found.high:>+7.4f}]   {verdict}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="'base', or a LoRA adapter path")
    parser.add_argument("--compare", nargs="?", const=str(POINTS / "*.json"),
                        help="the curve of each arm, and where it should stop")
    parser.add_argument("--arms", nargs="?", const=str(POINTS / "*.json"),
                        help="the arms against each other, on their last point")
    parser.add_argument("--against", default="A",
                        help="reference arm for --arms (default A)")
    parser.add_argument("--faits", action="store_true",
                        help="re-answer the facts of an existing point, without "
                             "recomputing its bits per character")
    parser.add_argument("--out", default=str(POINTS))
    parser.add_argument("--no-upload", dest="upload", action="store_false")
    args = parser.parse_args()

    if args.arms:
        return arms(args.arms, args.against)
    if args.compare:
        return compare(args.compare)
    if not args.checkpoint:
        parser.error("--checkpoint or --compare is required")
    if args.faits:
        return refacts(args.checkpoint, Path(args.out), args.upload)
    return measure(args.checkpoint, Path(args.out), args.upload)


if __name__ == "__main__":
    sys.exit(main())
