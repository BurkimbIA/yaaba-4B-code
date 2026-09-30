"""Measure one checkpoint, and pick the checkpoint to keep on each arm's curve.

    python scripts/measure/evaluate_checkpoint.py --checkpoint base      # on GPU
    python scripts/measure/evaluate_checkpoint.py --checkpoint /path/checkpoint-1094
    python scripts/measure/evaluate_checkpoint.py --compare              # no GPU
    python scripts/measure/evaluate_checkpoint.py --arms                 # no GPU

A checkpoint is measured three ways: bits per character on seven held-out
capabilities, 52 automatically scored facts, and 26 generative probes. Perplexity
alone missed a checkpoint that moved the capital of Burkina Faso from Ouagadougou
to Bobo-Dioulasso.

Three signals stop a curve: Moore bits per character rise, a fact the base knew is
answered with its known wrong answer, or English, French, code or maths lose more
than 3 %. The kept checkpoint has the lowest Moore bits per character before the
first signal.
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
# `measure` also takes the destination as a parameter, so a second stage can
# write elsewhere and leave the CPT curve alone.

SHOULD_DROP = ("moore_humain", "francais_parallele")
SHOULD_HOLD = ("anglais", "francais", "code", "maths")
# CPT exit criterion: each of these four may lose at most this fraction of its
# base bits per character.
HOLD_TOLERANCE = 0.03
WATCH_APART = ("moore_whisper",)

# Printed next to a point but never a stop signal: it only says that the facts
# were scored at another budget or on other items.
NOT_COMPARABLE = ("facts not comparable: another budget, "
                  "or another set of items")


def overlap(a: str, b: str) -> float:
    """Word overlap between a question and its answer, to measure echo."""
    first, second = set(a.casefold().split()), set(b.casefold().split())
    return len(first & second) / max(len(first), 1)


# An SFT point trained on a CPT arm, as the notebook names it:
# `C-4820-checkpoint-1191`. CPT points (`checkpoint-6005`) have no arm and step
# prefix.
POINT_DE_BRAS = re.compile(r"^(?:A|B|C|P|P2)-\d+-checkpoint-\d+$")


def load(checkpoint: str, socle: str | None = None):
    """The base with `socle` and then `checkpoint` merged in, and the point's name.

    Adapters are merged so that generation runs at the speed of the base.

    An SFT point trained on an arm needs `socle`, the arm's CPT adapter, because
    its SFT adapter was trained on top of base + CPT. Loaded on the bare base it
    still returns numbers (once 4.79 Moore bits per character, against 1.28 for
    the arm alone), so the guard below raises instead.
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
    """The reply, and whether it ended before the token budget.

    Kept at module level so every caller uses the same chat template, with
    `enable_thinking=False`.

    `heat` above 0 turns sampling on. It tells whether a collapse seen at greedy
    decoding comes from the decoder or from the weights: greedy takes the argmax
    at every step, so questions whose distributions start alike get the same
    opening. If sampling brings variety back, the collapse was in the decoder.

    0.0 keeps greedy decoding, as in every stored measurement. Sampling requires
    `seed`, so that a draw can be replayed.

    Give each question its own seed (`seed + index`). This function reseeds the
    global generator on every call, so one seed for all questions makes them
    draw the same random stream, and their answers then open alike for that
    reason alone (once `Yaa tɩ` twelve times at T=0.7).
    """
    import torch

    if heat and seed is None:
        raise ValueError("sampling without a seed cannot be replayed")
    if heat:
        torch.manual_seed(seed)

    message = [{"role": "user", "content": prompt}]
    try:
        # Qwen3's thinking mode is on by default; it reasons in Chinese and uses up
        # the token budget.
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
                                # transformers silently ignores `temperature`
                                # when `do_sample` is false; None makes that
                                # visible.
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
        # Stored per text so that intervals need no GPU later. The texts keep the
        # order of the held-out file, which pairs the comparisons between points.
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
    """Answer and score every fact item. Shared by `measure` and `refacts`."""
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
    """Re-answer the fact items of an existing point and leave its other measures.

    Bits per character do not depend on the generation budget and take minutes;
    the fact answers depend on it and take seconds. A change of `BUDGET` only
    needs this. `budget_faits` is stored next to the answers, since two tallies
    compare only at the same budget.

    `point_name` names the point file when it differs from the adapter folder:
    the folder `checkpoint-218` holds the adapter of `P2-6120-checkpoint-218`.

    Pass `socle` for an SFT point trained on an arm, as for `load`. The guard in
    `load` cannot catch it here, because it only sees the folder name.
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
    # Both denominators, because the probe grew from 20 items to 52.
    print(f"\n  {was}/{over} -> "
          f"{point['faits_compte']['juste']}/{len(point['faits'])}")
    print(f"-> {path}")

    if upload:
        s3.upload_file(path, f"{POINTS_URI}/{path.stem}.json")
        print(f"-> {POINTS_URI}/{path.stem}.json")
    return 0


def order(name: str) -> tuple[int, int]:
    """Sort key: base first, then checkpoints by step, compared as numbers.

    The step is the number after `checkpoint-`. Reading every digit in the name
    once sorted `A-20260831-1024-checkpoint-219` by its run timestamp.
    """
    if name == "base":
        return (0, 0)
    _, marker, tail = name.rpartition("checkpoint-")
    digits = "".join(c for c in (tail if marker else name) if c.isdigit())
    return (1, int(digits) if digits else 0)


def arm_of(point: dict) -> str:
    """The arm that produced a point, read from the run directory in its path.

    `/content/cpt/A-20260831-1024/checkpoint-6005` gives `A`. All arms write
    their points to one directory, and this is what separates their curves.
    """
    if point.get("point") == "base":
        return "base"
    parts = Path(point.get("chemin", "")).parts
    for part in reversed(parts):
        if "-" in part and not part.startswith("checkpoint"):
            return part.split("-", 1)[0]
    return "?"


def asked_the_same(point: dict) -> tuple[int, frozenset[str]]:
    """What two fact tallies must share to be compared: budget and item set."""
    return budget_of(point), frozenset(f["id"] for f in point["faits"])


def same_fact_budget(first: dict, second: dict) -> bool:
    """Whether two points scored their facts at the same budget and on the same items.

    Two points scored at 48 tokens compare fine. A point at 128 and one at 320
    do not, because a longer budget lets more answers finish. The item set
    matters in the same way: 13 of 20 and 38 of 52 would otherwise share a
    column. `lost` is keyed by item id and does not depend on this.
    """
    return asked_the_same(first) == asked_the_same(second)


def forgotten(known: set[str], verdicts: dict[str, str]) -> list[str]:
    """Known facts that this point no longer answers at all.

    Printed next to the verdict and never used to reject a point: at greedy
    decoding single items flip back and forth within a run, and a per-item veto
    rejected every checkpoint.
    """
    return sorted(i for i in known if verdicts.get(i) == "faux")


def marks_for(base: dict, earlier: dict, point: dict, known: set[str]) -> list[str]:
    """Every stop signal `point` crosses, judged against the base and `earlier`.

    `earlier` is the previous point of the same arm. `keepable` and `compare`
    both call this, so they apply one rule.
    """
    marks = []
    current = point["mesures"]["moore_humain"]["bits_par_caractere"]
    previous = earlier["mesures"]["moore_humain"]["bits_par_caractere"]

    # Only a rise whose interval excludes zero counts; comparing point estimates
    # would stop on noise.
    rise = gap(earlier, point, "moore_humain")
    if rise is None:
        if current >= previous:
            marks.append(f"overfitting? (bpc {previous:.4f} -> {current:.4f}, "
                         "no interval: re-measure)")
    elif rise.low > 0:
        marks.append(f"overfitting (bpc {rise.value:+.4f} "
                     f"[{rise.low:+.4f}, {rise.high:+.4f}])")

    # A known fact now answered with its trap is a regression. `faux` (no answer)
    # is only reported, by `forgotten`, because single items flip within a run at
    # greedy decoding. `ambigu` goes to a human reader.
    verdicts = {f["id"]: f["note"] for f in point["faits"]}
    if not same_fact_budget(base, point):
        marks.append(NOT_COMPARABLE)
        verdicts = {}
    lost = sorted(i for i in known if verdicts.get(i) == "piege")
    if lost:
        marks.append(f"facts learned otherwise: {', '.join(lost)}")

    # Bits per character are a cost, so a rise is a loss.
    for capability in SHOULD_HOLD:
        was = base["mesures"][capability]["bits_par_caractere"]
        now = point["mesures"][capability]["bits_par_caractere"]
        if (now - was) / was > HOLD_TOLERANCE:
            marks.append(f"{capability} +{100 * (now - was) / was:.1f} % "
                         f"(over {100 * HOLD_TOLERANCE:.0f} %)")
    return marks


def answer_key(arm: str, point_name: str) -> tuple[str, str]:
    """(arm, training step), the key that matches answer files to point files.

    Their names differ (`218-checkpoint-218` for the answers,
    `P2-6120-checkpoint-218` for the point), so matching on names found no
    answers. Both record the arm and the step.
    """
    return arm, point_name.rpartition("checkpoint-")[2] or point_name


def criterion_answers(directory: Path = EVAL) -> dict[tuple[str, str], list[str]]:
    """The answers to the twelve held-out questions, by (arm, step).

    `measure` does not generate them: the SFT notebook writes them to
    `evaluation/reponses-*.jsonl`. Reading them here lets point selection apply
    the collapse checks.

    Only answers to the current twelve questions count, matched on the question
    text, because older answer files hold earlier question sets. The questions
    come from `questions_tenues.jsonl` in the gated evaluation set.
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
    """Why this point collapsed, or None.

    Two checks, because the opening count sees only the first two words:

    - `moore.speaks_to_each`: no single opening answers a majority of the
      questions;
    - `moore.answers_each`: no complete answer is given to two questions.

    On one checkpoint, greedy decoding gave twelve distinct sentences under one
    opening (the first check fails, the second passes), and sampling at T=1.0
    gave one sentence to four questions (the first passes, the second fails).

    Unlike `marks_for`, this does not cut the curve: collapse fades as training
    goes on, so later points can still be kept.

    A point with no answers to the current questions gets None here, and the
    caller flags it (see `unchecked`).
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
    """Whether the point has no answers, so the collapse checks could not run."""
    return not answers.get(answer_key(arm_of(point), point["point"]))


def keepable(base: dict, curve: list[dict],
             answers: dict[str, list[str]] | None = None
             ) -> tuple[dict | None, str | None, list[tuple[str, str]]]:
    """The point to keep on one arm's curve, where the curve was cut, and the
    collapsed points skipped.

    The kept point has the lowest Moore bits per character before the first stop
    signal. Collapsed points are skipped and the curve goes on past them. With
    `answers=None` the collapse checks do not run.
    """
    known = {f["id"] for f in base["faits"] if f["note"] == "juste"}
    eligible: list[dict] = []
    skipped: list[tuple[str, str]] = []
    earlier = base

    def best() -> dict | None:
        return min(eligible, key=lambda p: p["mesures"]["moore_humain"]["bits_par_caractere"],
                   default=None)

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
    """Paired difference between two points on one capability, or None when either
    point has no per-text values.
    """
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


def compare(pattern: str, answers: dict[tuple[str, str], list[str]] | None = None) -> int:
    """The curve of each arm and where it stops. Returns 1 if an arm keeps nothing.

    `answers` defaults to `criterion_answers()`, which reads the gated set.
    """
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

    # Each point against the base, paired over the same held-out texts. Two
    # separate intervals can overlap while the paired difference excludes zero.
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

    # A curve belongs to one arm and all arms share a directory, so a point's
    # predecessor is the previous point of its own arm.
    curves: dict[str, list[dict]] = {}
    for point in points[1:]:
        curve = curves.setdefault(arm_of(point), [])
        earlier = curve[-1] if curve else base
        curve.append(point)
        verdicts = ({f["id"]: f["note"] for f in point["faits"]}
                    if same_fact_budget(base, point) else {})
        gone = forgotten(known, verdicts)
        unclear = sorted(i for i in known if verdicts.get(i) == "ambigu")
        if gone:
            print(f"  {point['point']:22s} no longer answers {len(gone)} of "
                  f"{len(known)}: {', '.join(gone)}")
        if unclear:
            print(f"  {point['point']:22s} unclear, read them: {', '.join(unclear)}")
        marks = marks_for(base, earlier, point, known)
        print(f"  {point['point']:22s} {'; '.join(marks) if marks else 'none'}")

    # `keepable` decides where each curve stops, as for `--arms`, so both commands
    # keep the same points.
    said = criterion_answers() if answers is None else answers
    start = base["mesures"]["moore_humain"]["bits_par_caractere"]
    failed = False
    for arm, curve in curves.items():
        best, crossed, skipped = keepable(base, curve, said)
        for name, why in skipped:
            print(f"  {arm}/{name.removeprefix('checkpoint-')}: {why}, not keepable")
        if best is None:
            # The reason for the non-zero exit goes to stderr, where a caller
            # checking the return code looks for it.
            why = (f"{crossed} already crosses a signal" if crossed
                   else "every point collapsed")
            print(f"\n{arm}: NO eligible checkpoint, {why}.", file=sys.stderr)
            failed = True
            continue
        if unchecked(best, said):
            print(f"  {arm}/{best['point'].removeprefix('checkpoint-')}: no answer "
                  "to the current twelve questions, collapse checks not applied")
        end = best["mesures"]["moore_humain"]["bits_par_caractere"]
        print(f"\nBEST {arm}: {best['point']}   Moore {start:.4f} -> {end:.4f} bpc  "
              f"({100 * (end - start) / start:+.1f} %)")
        if crossed:
            print(f"  (from {crossed} onward excluded: signal crossed)")
    return 1 if failed else 0


def arms(pattern: str, against: str = "A") -> int:
    """Rank the arms on their kept points, against the arm `against`.

    `compare` follows each arm's curve; this compares the arms. Every gap is a
    paired difference over the same held-out texts: two per-point intervals can
    overlap while the paired difference excludes zero.
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

    # Each arm is shown at its kept point. Moore can improve up to the last epoch
    # and still be worse there than three epochs earlier.
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
            # An arm may keep nothing. Its last point is still shown, and labelled
            # as such.
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

    # A curve of one or two points cannot show a peak, so say so rather than
    # report that no signal was crossed.
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
    # Arms can be shown at different steps (a rejected arm at its last point, a
    # kept one at its best), so the point name is printed under each column.
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

    # Facts and echo are reported apart: bits per character see neither.
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
    # Known facts this point no longer answers, reported only (see
    # `forgotten`).
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
