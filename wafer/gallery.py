"""Grad-CAM gallery for the README: one test wafer per defect pattern, plus mixed-defect wafers.

    python -m wafer.gallery
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from wafer.data import LABELS, ROOT, load_splits  # noqa: E402
from wafer.explain import grad_cam, load_model, predict_one  # noqa: E402

CMAP = ListedColormap(["white", "#9ecae1", "#d62728"])


def main():
    model, thr = load_model()
    X, y = load_splits()["test"]
    rng = np.random.default_rng(7)
    picks = []
    for j, name in enumerate(LABELS):                       # single-defect wafers
        idx = np.nonzero((y[:, j] == 1) & (y.sum(1) == 1))[0]
        picks.append((int(rng.choice(idx)), name))
    for combo in (["Center", "Scratch"], ["Edge-Ring", "Loc"], ["Donut", "Edge-Loc", "Scratch"]):
        mask = np.all(y[:, [LABELS.index(c) for c in combo]] == 1, axis=1) & (y.sum(1) == len(combo))
        picks.append((int(rng.choice(np.nonzero(mask)[0])), " + ".join(combo)))

    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.8))
    for col, (i, title) in enumerate(picks):
        wafer = X[i]
        prob, detected = predict_one(model, thr, wafer)
        axes[0, col].imshow(wafer, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[0, col].set_title(title, fontsize=8)
        cam = np.max([grad_cam(model, wafer, LABELS.index(d)) for d in detected], axis=0) if detected \
            else np.zeros_like(wafer, dtype=float)
        axes[1, col].imshow(wafer, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        axes[1, col].imshow(np.ma.masked_where(cam < 0.5, cam), cmap="inferno", alpha=0.6)
        axes[1, col].set_title("pred: " + (", ".join(detected) or "none"), fontsize=7,
                               color="green" if set(detected) == set(title.split(" + ")) else "red")
        for a in axes[:, col]:
            a.axis("off")
    axes[0, 0].text(-8, 26, "wafer", rotation=90, va="center", fontsize=9)
    axes[1, 0].text(-8, 26, "Grad-CAM", rotation=90, va="center", fontsize=9)
    fig.suptitle("Test wafers (red = failed die) and where the CNN looked", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "gradcam_gallery.png", dpi=130)
    print("saved reports/gradcam_gallery.png")


if __name__ == "__main__":
    main()
