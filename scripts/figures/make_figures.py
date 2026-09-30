"""Figures of the paper, from the measurement files only. Written to figures/.

    python scripts/figures/make_figures.py

Every point comes from `evaluation/`; nothing is typed in by hand. Arm colours
are the validated four-slot categorical palette in fixed arm order, and every
series also carries its own marker, so identity never rests on colour alone.
Text is set in Latin Modern, the font of the PMLR paper, and embedded as
TrueType (`pdf.fonttype` 42): matplotlib's default Type 3 fonts render blurred
and are refused by some venues.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))
from yaaba import moore  # noqa: E402
from yaaba.uncertainty import difference  # noqa: E402

ARMS = {  # arm: (colour, marker, final CPT step = 5 epochs)
    "A": ("#2a78d6", "o", 6005),
    "B": ("#eb6834", "s", 2755),
    "C": ("#1baf7a", "^", 6025),
    "P2": ("#eda100", "D", 6120),
}
# The checkpoint each arm is compared at, as in Tables 3 and 4 (C at its best point).
END = {"A": 6005, "B": 2755, "C": 4820, "P2": 6120}
INK, MUTED, GRID = "#1a1a1a", "#5f5e5a", "#e4e3df"
EPOCHS = 5
BASE = ROOT / "evaluation" / "points" / "base.json"
FINAL = ROOT / "evaluation" / "points-sft-final"
RELEASED = 651
# The PMLR text width, so the fonts below print at their stated size (ACL scales by 1.05).
WIDTH = 6.0

# ponytail: the path is MiKTeX's; elsewhere the figures fall back to the default serif.
LM = Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm"
if not (LM / "lmroman10-regular.otf").exists():
    print(f"warning: Latin Modern not found in {LM}, the figures will not match the paper font")
for face in ("regular", "italic", "bold"):
    if (LM / f"lmroman10-{face}.otf").exists():
        font_manager.fontManager.addfont(str(LM / f"lmroman10-{face}.otf"))

# The French reading copy gets its own figures. A missing entry raises, so no label
# stays in English by accident.
FRENCH = {
    "Mooré bits per character (log scale)": "bits par caractère, mooré (éch. log)",
    "parallel-French bits per character": "bits par caractère, français parallèle",
    "epochs over the Mooré data": "époques sur le mooré",
    "end of CPT": "fin du CPT",
    "after the same SFT": "après le même SFT",
    "Mooré": "mooré",
    "parallel French": "français parallèle",
    "difference from P2 (bits per character)": "écart à P2 (bits par caractère)",
    "held-out SFT loss": "perte SFT tenue à l'écart",
    "Mooré bits per character": "bits par caractère, mooré",
    "rejected": "rejeté",
    "largest shared opening (of {total})": "réponses de même ouverture (sur {total})",
    "SFT step": "pas de SFT",
    "released": "publié",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8.5, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.8,
    "legend.frameon": False, "axes.unicode_minus": True,
})


def measure(path: Path, capability: str) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["mesures"][capability]


def per_text(path: Path, capability: str) -> list[tuple[float, int]]:
    return [tuple(pair) for pair in measure(path, capability)["par_texte"]]


def text(english: str, french: bool) -> str:
    return FRENCH[english] if french else english


class CommaFormatter(ScalarFormatter):
    """French decimal comma, with the tick placement of the default formatter."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def save(fig, name: str, french: bool) -> None:
    """Write the figure; the French one takes decimal commas on its numeric axes."""
    if french:
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                if type(axis.get_major_formatter()) is ScalarFormatter:
                    axis.set_major_formatter(CommaFormatter())
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}{'_fr' if french else ''}.pdf")
    plt.close(fig)


def panel_label(ax, letter: str) -> None:
    ax.text(-0.02, 1.04, f"({letter})", transform=ax.transAxes, ha="right",
            va="bottom", fontweight="bold", color=INK)


def cpt_curves(capability: str) -> dict[str, list[tuple[float, float]]]:
    """Per arm: (epoch, bpc), read from each point's own path, starting at the base model."""
    base = measure(BASE, capability)["bits_par_caractere"]
    curves: dict[str, list] = {arm: [(0.0, base)] for arm in ARMS}
    for path in (ROOT / "evaluation" / "points").glob("checkpoint-*.json"):
        chemin = json.loads(path.read_text(encoding="utf-8"))["chemin"]
        match = re.search(r"/cpt/([A-Z0-9]+)-\d{8}-\d{4}/checkpoint-(\d+)$", chemin)
        if not match or match.group(1) not in ARMS:
            continue
        arm, step = match.group(1), int(match.group(2))
        curves[arm].append((EPOCHS * step / ARMS[arm][2],
                            measure(path, capability)["bits_par_caractere"]))
    return {arm: sorted(points) for arm, points in curves.items()}


def figure_cpt(french: bool = False) -> None:
    """Bits per character along CPT, from the shared base model to five epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.55))
    for ax, capability, label, letter in (
            (axes[0], "moore_humain", text("Mooré bits per character (log scale)", french), "a"),
            (axes[1], "francais_parallele",
             text("parallel-French bits per character", french), "b")):
        curves = cpt_curves(capability)
        for arm, points in curves.items():
            colour, marker, _ = ARMS[arm]
            # Arm A was measured at three points only: a dashed line says so.
            ax.plot(*zip(*points, strict=True), color=colour, marker=marker,
                    linestyle="--" if arm == "A" else "-", label=arm,
                    markevery=range(1, len(points)), zorder=3)
        base = curves["A"][0][1]
        ax.plot([0], [base], marker="*", color=INK, markersize=7, zorder=4,
                linestyle="none")
        value = f"{base:.2f}".replace(".", "," if french else ".")
        ax.annotate(f"Qwen3-4B : {value}" if french else f"Qwen3-4B: {value}", (0, base),
                    xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK)
        ax.set_xlabel(text("epochs over the Mooré data", french))
        ax.set_ylabel(label)
        ax.set_xlim(-0.15, EPOCHS + 0.15)
        panel_label(ax, letter)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(FixedLocator([1.2, 1.5, 2, 3, 4.5]))
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylim(1.12, 5.0)
    axes[1].set_ylim(0.88, 1.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, handlelength=2.2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.5)
    save(fig, "cpt_curves", french)


def figure_transfer(french: bool = False) -> None:
    """Each arm minus P2, paired over the same texts, at the end of CPT and after SFT."""
    def at(arm: str, stage: str) -> Path:
        if stage == "cpt":
            return ROOT / "evaluation" / "points" / f"checkpoint-{END[arm]}.json"
        return ROOT / "evaluation" / "points-sft" / f"{arm}-{END[arm]}-checkpoint-1191.json"

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2), sharey=True)
    stages = (("cpt", text("end of CPT", french), "white", -0.15),
              ("sft", text("after the same SFT", french), None, 0.15))
    for ax, capability, label, letter in (
            (axes[0], "moore_humain", text("Mooré", french), "a"),
            (axes[1], "francais_parallele", text("parallel French", french), "b")):
        for row, arm in enumerate(("A", "C", "B")):
            colour, marker, _ = ARMS[arm]
            for stage, _, face, shift in stages:
                gap = difference(per_text(at("P2", stage), capability),
                                 per_text(at(arm, stage), capability))
                ax.errorbar(gap.value, row + shift,
                            xerr=[[gap.value - gap.low], [gap.high - gap.value]],
                            color=colour, marker=marker, markersize=4.5, capsize=2,
                            elinewidth=1, markerfacecolor=face or colour,
                            markeredgewidth=1, linestyle="none", zorder=3)
        ax.axvline(0, color=INK, linewidth=0.8)
        ax.set_xlim(left=-0.01)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(label, color=INK)
        ax.set_xlabel(text("difference from P2 (bits per character)", french))
        panel_label(ax, letter)
    axes[0].set_yticks([0, 1, 2], ["A", "C", "B"])
    axes[0].set_ylim(2.5, -0.5)
    axes[0].set_xticks([0, 0.05, 0.10, 0.15])
    # The arm is read from the row, the stage from the fill: the legend shows only the fill.
    handles = [Line2D([], [], color=MUTED, marker="o", linestyle="none", markersize=4.5,
                      markerfacecolor=face or MUTED, markeredgewidth=1)
               for _, _, face, _ in stages]
    fig.legend(handles, [name for _, name, _, _ in stages], loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=2.5)
    save(fig, "cpt_to_sft", french)


def figure_sft(french: bool = False) -> None:
    """Three ways to pick a checkpoint along the final SFT run from P2."""
    prefix, run = "P2-6120", "sft-P2-6120-20260924-0002"
    log = [json.loads(line) for line in
           (FINAL / f"{run}.progression.jsonl").open(encoding="utf-8")]
    # 1306 is the end-of-run save, four steps after 1302: plotting both says nothing more.
    steps = [r["pas"] for r in log if "eval_loss" in r and r["pas"] != 1306]
    loss = [r["eval_loss"] for r in log if "eval_loss" in r and r["pas"] != 1306]
    bpc = [measure(FINAL / f"{prefix}-checkpoint-{s}.json", "moore_humain") for s in steps]
    largest = []
    for s in steps:
        answers = [json.loads(line)["obtenu"] for line in
                   (FINAL / f"reponses-{prefix}-checkpoint-{s}.jsonl").open(encoding="utf-8")]
        largest.append((moore.openings(answers)[1], len(answers)))
    total = largest[0][1]
    colour, marker, _ = ARMS["P2"]

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    axes[0].plot(steps, loss, color=colour, marker=marker)
    axes[0].set_ylabel(text("held-out SFT loss", french))

    values = [m["bits_par_caractere"] for m in bpc]
    low = [v - m["intervalle"][0] for v, m in zip(values, bpc, strict=True)]
    high = [m["intervalle"][1] - v for v, m in zip(values, bpc, strict=True)]
    axes[1].errorbar(steps, values, yerr=[low, high], color=colour, marker=marker,
                     capsize=2, elinewidth=0.9)
    axes[1].set_ylabel(text("Mooré bits per character", french))

    limit = total / 2
    axes[2].axhspan(limit, total + 0.5, color=GRID, alpha=0.6, linewidth=0)
    axes[2].axhline(limit, color=MUTED, linewidth=0.8, linestyle="--")
    axes[2].text(steps[-1], limit + 0.35, text("rejected", french), ha="right", va="bottom",
                 color=MUTED)
    axes[2].plot(steps, [n for n, _ in largest], color=colour, zorder=3)
    for s, (n, _) in zip(steps, largest, strict=True):
        axes[2].plot(s, n, marker=marker, color=colour, zorder=4,
                     markerfacecolor="white" if n > limit else colour, markeredgewidth=1)
    axes[2].set_ylim(0, total)
    axes[2].set_ylabel(text("largest shared opening (of {total})", french).format(total=total))

    for ax, letter in zip(axes, "abc", strict=True):
        ax.axvline(RELEASED, color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_xticks(steps, minor=True)
        ax.set_xticks([steps[0], RELEASED, steps[-1]])
        ax.set_xlim(steps[0] - 110, steps[-1] + 110)
        ax.set_xlabel(text("SFT step", french))
        panel_label(ax, letter)
    for ax in axes:
        ax.annotate(text("released", french), (RELEASED, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", color=MUTED)
    fig.tight_layout(w_pad=1.6)
    save(fig, "sft_run", french)


if __name__ == "__main__":
    for french in (False, True):
        figure_cpt(french)
        figure_transfer(french)
        figure_sft(french)
    print("written:", *sorted(p.name for p in OUT.glob("*.pdf")))
