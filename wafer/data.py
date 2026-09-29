"""Load MixedWM38, remove duplicate wafer maps and make a stratified split."""
import hashlib
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Wafer_Map_Datasets.npz"
SPLIT_FILE = ROOT / "data" / "split_indices.npz"
SEED = 42

# Column order of the 8-dim label, from the dataset description
LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Scratch", "Random"]

# Typical manufacturing causes reported in the wafer-map literature (e.g. Wu et al., 2015)
ROOT_CAUSES = {
    "Center": "Non-uniform process across the wafer, e.g. deposition or CMP (polishing) centre effects",
    "Donut": "Ring-shaped process variation, e.g. uneven heating or chuck temperature",
    "Edge-Loc": "Localised edge problem, e.g. wafer handling or clamp contact at one spot",
    "Edge-Ring": "Edge-wide process issue, e.g. etch or thin-film non-uniformity at the rim",
    "Loc": "Local contamination or a localised equipment fault",
    "Near-Full": "Catastrophic process failure affecting almost the whole wafer",
    "Scratch": "Mechanical damage during handling or polishing",
    "Random": "Particles or random contamination spread across the wafer",
}


def load_raw(path=RAW):
    """Wafer maps (N, 52, 52) with 0 = no die, 1 = good die, 2 = defective die; labels (N, 8)."""
    d = np.load(path, allow_pickle=False)  # no pickle: loading cannot execute code
    return d["arr_0"].astype(np.uint8), d["arr_1"].astype(np.float32)


def deduplicate(X, y):
    """Drop exact duplicate maps (keep the first) so no wafer can sit in both train and test."""
    hashes = np.array([hashlib.md5(x.tobytes()).hexdigest() for x in X])
    _, first = np.unique(hashes, return_index=True)
    keep = np.sort(first)
    return X[keep], y[keep], len(X) - len(keep)


def combo_id(y):
    """One id per label combination (38 in total) — used to stratify the split."""
    return (y * (2 ** np.arange(y.shape[1]))).sum(1).astype(int)


def load_splits():
    """70/15/15 train/val/test, stratified by label combination, cached to disk."""
    X, y = load_raw()
    X, y, n_dupes = deduplicate(X, y)
    if SPLIT_FILE.exists():
        s = np.load(SPLIT_FILE)
        tr, va, te = s["train"], s["val"], s["test"]
    else:
        idx = np.arange(len(X))
        tr, rest = train_test_split(idx, test_size=0.30, stratify=combo_id(y), random_state=SEED)
        va, te = train_test_split(rest, test_size=0.50, stratify=combo_id(y[rest]), random_state=SEED)
        np.savez(SPLIT_FILE, train=tr, val=va, test=te)
    return {"train": (X[tr], y[tr]), "val": (X[va], y[va]), "test": (X[te], y[te]),
            "n_duplicates_removed": n_dupes}


def to_channels(X):
    """Two binary channels the CNN can use: 'is wafer' and 'is defective die'."""
    return np.stack([(X > 0), (X == 2)], axis=1).astype(np.float32)
