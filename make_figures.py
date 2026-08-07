"""
Regenerates Figures 1 and 2 with layout computed from the text rather than
hand-tuned constants, and asserts that no text escapes its box.

Every box height is derived from how many lines it holds. After drawing, each
text artist's rendered bounding box is compared against its container's, so
overflow is caught programmatically instead of by eye.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MID, ACCENT, GREY = "#1F3864", "#2E5496", "#B5432F", "#6B6B6B"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 300})

_checks = []          # (label, text_artist, patch) triples to verify


def register(label, artist, patch):
    _checks.append((label, artist, patch))


def verify(fig, name):
    """Fail loudly if any registered text extends beyond its container."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for label, art, patch in _checks:
        tb = art.get_window_extent(renderer=r)
        pb = patch.get_window_extent(renderer=r)
        if (tb.x0 < pb.x0 - 1 or tb.x1 > pb.x1 + 1
                or tb.y0 < pb.y0 - 1 or tb.y1 > pb.y1 + 1):
            bad.append(label)
    if bad:
        raise AssertionError(f"{name}: text overflows in -> {bad}")
    print(f"  {name}: {len(_checks)} text blocks, all inside their boxes")
    _checks.clear()


# ---------------------------------------------------------------- Figure 2
def build_pipeline():
    """Boxes sized from their content, arranged on a 3-2-3 grid."""
    boxes = {
        "deposit":  ("Public deposit",
                     ["Zenodo 10.5281/zenodo.6471045",
                      "6 participants",
                      "21\u201328 Mar 2022"]),
        "streams":  ("Raw Beiwe streams",
                     ["gps", "power_state", "survey_answers"]),
        "prep":     ("Preprocessing",
                     ["parse UTC \u2192 local time",
                      "(America/New_York)",
                      "join hourly files"]),
        "home":     ("Home estimation",
                     ["DBSCAN on true distances",
                      "night fixes 21:00\u201306:00",
                      "(after Wang et al.)"]),
        "features": ("Feature extraction",
                     ["time at home",
                      "radius of gyration",
                      "location entropy",
                      "screen-on duration"]),
        "probe":    ("Blurring probe",
                     ["overnight point-to-point",
                      "movement while the person",
                      "is almost certainly still"]),
        "valid":    ("Validation",
                     ["derived vs. self-reported",
                      "hours at home",
                      "r, bias, MAE, Bland\u2013Altman"]),
        "out":      ("Outputs",
                     ["daily_features.csv",
                      "validation_stats.txt",
                      "noise_sensitivity.csv"]),
    }

    FIG_W, FIG_H = 7.6, 5.0
    TITLE_PT, BODY_PT = 8.2, 6.9
    # Convert point sizes to axes fractions (axes fills the figure).
    line_h = (BODY_PT * 1.55) / (FIG_H * 72)
    title_h = (TITLE_PT * 1.7) / (FIG_H * 72)
    pad = 0.020

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def draw(key, cx, top, w, ec=MID, fc="#F4F8FC"):
        title, lines = boxes[key]
        h = pad * 2 + title_h + len(lines) * line_h
        x, y = cx - w / 2, top - h
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.008,rounding_size=0.016",
                               linewidth=1.1, edgecolor=ec, facecolor=fc)
        ax.add_patch(patch)
        t = ax.text(cx, top - pad, title, ha="center", va="top",
                    fontsize=TITLE_PT, fontweight="bold", color=ec)
        register(f"{key}:title", t, patch)
        yy = top - pad - title_h
        for i, ln in enumerate(lines):
            a = ax.text(cx, yy - i * line_h, ln, ha="center", va="top",
                        fontsize=BODY_PT, color="#333333")
            register(f"{key}:{i}", a, patch)
        return {"cx": cx, "top": top, "bot": y, "l": x, "r": x + w}

    W = 0.29
    row1 = 0.95
    g = {}
    g["deposit"] = draw("deposit", 0.165, row1, W, ec=INK, fc="#FFFFFF")
    g["streams"] = draw("streams", 0.500, row1, W, ec=INK, fc="#FFFFFF")
    g["prep"] = draw("prep", 0.835, row1, W, ec=INK, fc="#FFFFFF")

    row2 = 0.615
    g["home"] = draw("home", 0.835, row2, W)
    g["features"] = draw("features", 0.500, row2, W)
    g["probe"] = draw("probe", 0.165, row2, W, ec=ACCENT, fc="#FBEEEA")

    row3 = 0.275
    g["valid"] = draw("valid", 0.300, row3, 0.30)
    g["out"] = draw("out", 0.700, row3, 0.30, ec=INK, fc="#FFFFFF")

    def arrow(a, b, label=None, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.05, color=GREY, shrinkA=2,
                                     shrinkB=2,
                                     connectionstyle=f"arc3,rad={rad}"))
        if label:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.014, label,
                    ha="center", va="bottom", fontsize=6.1, color=GREY,
                    style="italic")

    my1 = (row1 + g["deposit"]["bot"]) / 2
    arrow((g["deposit"]["r"], my1), (g["streams"]["l"], my1), "download")
    arrow((g["streams"]["r"], my1), (g["prep"]["l"], my1))
    arrow((g["prep"]["cx"], g["prep"]["bot"]), (g["home"]["cx"], row2))
    my2 = (row2 + g["features"]["bot"]) / 2
    arrow((g["home"]["l"], my2), (g["features"]["r"], my2), "home point")
    arrow((g["features"]["l"], my2), (g["probe"]["r"], my2))
    arrow((g["features"]["cx"] - 0.05, g["features"]["bot"]),
          (g["valid"]["cx"] + 0.04, row3))
    arrow((g["probe"]["cx"], g["probe"]["bot"]),
          (g["valid"]["cx"] - 0.06, row3))
    my3 = (row3 + g["valid"]["bot"]) / 2
    arrow((g["valid"]["r"], my3), (g["out"]["l"], my3))

    ax.text(0.5, 0.995, "How the analysis runs, from download to results",
            ha="center", va="top", fontsize=9.2, fontweight="bold", color=INK)

    verify(fig, "Figure 2")
    fig.savefig("/home/claude/fig_pipeline.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def build_benefits():
    cols = [
        ("Benefits researchers hope for", "as listed by Oudin et al. (2023)",
         [["Symptoms tracked continuously,", "not only at visits"],
          ["Faster, more accurate", "diagnosis"],
          ["Treatment fitted to the", "individual"],
          ["Better follow-up over time"],
          ["Patients able to track their", "own condition"]],
         MID, "#F4F8FC"),
        ("Features those benefits need", "computed from phone sensor data",
         [["Hours spent at home"],
          ["How far from home a person", "ranges in a day"],
          ["Number of places visited"],
          ["Screen-on time, and when", "the phone is used"],
          ["How often people communicate"]],
         INK, "#F7F7F9"),
        ("What this study checks", "the contribution of this paper",
         [["COVERAGE", "Is enough of the day actually", "recorded, or are there gaps?"],
          ["DISTORTION", "Does privacy blurring change", "the shape of the trace?"],
          ["AGREEMENT", "Does our number match what", "people say they did?"]],
         ACCENT, "#FDF4F1"),
    ]

    FIG_W, FIG_H = 7.8, 4.4
    TITLE_PT, SUB_PT, BODY_PT = 8.4, 6.2, 6.9
    line_h = (BODY_PT * 1.5) / (FIG_H * 72)
    gap_h = line_h * 0.75
    title_h = (TITLE_PT * 1.7) / (FIG_H * 72)
    sub_h = (SUB_PT * 1.7) / (FIG_H * 72)
    pad = 0.028

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    top = 0.885
    heights = []
    for _t, _s, items, _e, _f in cols:
        heights.append(pad * 2 + title_h + sub_h
                       + sum(len(it) * line_h + gap_h for it in items))
    H = max(heights)

    W, GAP = 0.305, 0.0225
    x0 = (1 - (3 * W + 2 * GAP)) / 2

    for ci, (title, sub, items, ec, fc) in enumerate(cols):
        x = x0 + ci * (W + GAP)
        y = top - H
        patch = FancyBboxPatch((x, y), W, H,
                               boxstyle="round,pad=0.008,rounding_size=0.016",
                               linewidth=1.2, edgecolor=ec, facecolor=fc)
        ax.add_patch(patch)
        t = ax.text(x + W / 2, top - pad, title, ha="center", va="top",
                    fontsize=TITLE_PT, fontweight="bold", color=ec)
        register(f"c{ci}:title", t, patch)
        st = ax.text(x + W / 2, top - pad - title_h, sub, ha="center",
                     va="top", fontsize=SUB_PT, color="#6A6A6A", style="italic")
        register(f"c{ci}:sub", st, patch)

        yy = top - pad - title_h - sub_h - gap_h * 0.4
        for ii, lines in enumerate(items):
            d = ax.text(x + 0.016, yy, "\u2013", ha="left", va="top",
                        fontsize=BODY_PT, color=ec)
            register(f"c{ci}:{ii}:dash", d, patch)
            for li, ln in enumerate(lines):
                bold = (ci == 2 and li == 0)
                a = ax.text(x + 0.040, yy - li * line_h, ln, ha="left",
                            va="top", fontsize=BODY_PT,
                            fontweight="bold" if bold else "normal",
                            color=ec if bold else "#2B2B2B")
                register(f"c{ci}:{ii}:{li}", a, patch)
            yy -= len(lines) * line_h + gap_h

    ymid = top - H / 2
    for ci in range(2):
        xa = x0 + ci * (W + GAP) + W
        ax.add_patch(FancyArrowPatch((xa + 0.002, ymid), (xa + GAP - 0.002, ymid),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=1.25, color="#8A8A8A"))

    ax.text(0.5, 0.985,
            "A clinical benefit is only as good as the feature behind it,",
            ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0.5, 0.985 - 0.042,
            "and a feature is only as good as the data behind it",
            ha="center", va="top", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(0.5, top - H - 0.022,
            "Diagram, not data. The left column lists claims made in the cited "
            "literature, not findings of this study.",
            ha="center", va="top", fontsize=6.2, color="#777777", style="italic")

    verify(fig, "Figure 1")
    fig.savefig("/home/claude/fig_benefits.png", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


build_benefits()
build_pipeline()
print("both figures regenerated")
