"""
Train and compare: hand-crafted features + Random Forest (baseline) vs a CNN.

    python -m wafer.train               # full run
    python -m wafer.train --epochs 2    # quick smoke run

Per-label decision thresholds are tuned on the validation set; the test set is used once.
"""
import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from wafer.data import LABELS, ROOT, SEED, load_splits, to_channels  # noqa: E402
from wafer.features import feature_matrix  # noqa: E402
from wafer.model import WaferCNN, random_dihedral  # noqa: E402

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def tune_thresholds(y, prob):
    """Per-label threshold that maximises F1 on the validation set."""
    grid = np.linspace(0.05, 0.95, 19)
    return np.array([grid[np.argmax([f1_score(y[:, j], prob[:, j] >= t, zero_division=0) for t in grid])]
                     for j in range(y.shape[1])])


def evaluate(y, prob, thr):
    pred = (prob >= thr).astype(int)
    n_defects = y.sum(1).astype(int)
    exact = (pred == y).all(1)
    return {
        "exact_match": round(float(exact.mean()), 4),             # every label correct
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y, pred, average="micro", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y, pred)), 4),
        "per_label": {LABELS[j]: {
            "precision": round(float(precision_score(y[:, j], pred[:, j], zero_division=0)), 4),
            "recall": round(float(recall_score(y[:, j], pred[:, j], zero_division=0)), 4),
            "f1": round(float(f1_score(y[:, j], pred[:, j], zero_division=0)), 4),
            "support": int(y[:, j].sum())} for j in range(len(LABELS))},
        "exact_match_by_n_defects": {int(k): round(float(exact[n_defects == k].mean()), 4)
                                     for k in np.unique(n_defects)},
        "normal_wafers_flagged": round(float(pred[n_defects == 0].any(1).mean()), 4),
    }


# ------------------------------------------------------------------ baseline
def run_baseline(splits):
    t0 = time.time()
    F = {k: feature_matrix(splits[k][0]) for k in ("train", "val", "test")}
    print(f"  hand-crafted features: {F['train'].shape[1]} per wafer ({time.time() - t0:.0f}s)")
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
    rf.fit(F["train"], splits["train"][1])
    proba = lambda X: np.stack([p[:, 1] for p in rf.predict_proba(X)], axis=1)  # noqa: E731
    thr = tune_thresholds(splits["val"][1], proba(F["val"]))
    return evaluate(splits["test"][1], proba(F["test"]), thr)


# ------------------------------------------------------------------ CNN
def predict(model, X, batch=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i + batch]))).numpy())
    return np.concatenate(out)


def run_cnn(splits, epochs):
    torch.manual_seed(SEED)
    Xtr, ytr = to_channels(splits["train"][0]), splits["train"][1]
    Xva, yva = to_channels(splits["val"][0]), splits["val"][1]
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=128, shuffle=True)
    model = WaferCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(loader))
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_f1 = [], -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in loader:
            xb = random_dihedral(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        val_f1 = f1_score(yva, predict(model, Xva) >= 0.5, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "val_macro_f1": float(val_f1)})
        print(f"  epoch {epoch:2d}  loss {total / len(Xtr):.4f}  val macro-F1 {val_f1:.4f}  ({time.time() - t0:.0f}s)")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODEL_DIR / "cnn.pt")
    model.load_state_dict(torch.load(MODEL_DIR / "cnn.pt"))
    thr = tune_thresholds(yva, predict(model, Xva))
    prob_te = predict(model, to_channels(splits["test"][0]))
    return model, thr, evaluate(splits["test"][1], prob_te, thr), history, prob_te


def plots(results, history):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(LABELS))
    for i, (name, r) in enumerate(results.items()):
        ax[0].bar(x + (i - 0.5) * 0.38, [r["per_label"][l]["f1"] for l in LABELS], 0.38, label=name)
    ax[0].set_xticks(x, LABELS, rotation=30)
    ax[0].set(ylim=(0, 1.05), title="F1 per defect pattern (test)")
    ax[0].legend(loc="lower left")
    for name, r in results.items():
        k = sorted(r["exact_match_by_n_defects"])
        ax[1].plot(k, [r["exact_match_by_n_defects"][i] for i in k], marker="o", label=name)
    ax[1].set(xlabel="Number of defect patterns on the wafer", ylabel="Exact-match accuracy",
              title="Accuracy as defects get mixed", ylim=(0, 1.05), xticks=[0, 1, 2, 3, 4])
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=120)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["epoch"] for h in history], [h["val_macro_f1"] for h in history], marker="o")
    ax.set(xlabel="Epoch", ylabel="Validation macro-F1", title="CNN training")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "cnn_training.png", dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    splits = load_splits()
    print(f"Removed {splits['n_duplicates_removed']} duplicate maps | "
          f"train {len(splits['train'][0])} / val {len(splits['val'][0])} / test {len(splits['test'][0])}")

    results = {}
    if not args.skip_baseline:
        print("Baseline: hand-crafted features + Random Forest")
        results["Features + Random Forest"] = run_baseline(splits)
        print(f"  test exact-match {results['Features + Random Forest']['exact_match']:.4f}, "
              f"macro-F1 {results['Features + Random Forest']['macro_f1']:.4f}")
    print("CNN")
    model, thr, results["CNN"], history, prob_te = run_cnn(splits, args.epochs)
    print(f"  test exact-match {results['CNN']['exact_match']:.4f}, macro-F1 {results['CNN']['macro_f1']:.4f}")

    json.dump({"labels": LABELS, "thresholds": thr.round(2).tolist()}, open(MODEL_DIR / "thresholds.json", "w"), indent=2)
    json.dump({"n_duplicates_removed": splits["n_duplicates_removed"],
               "sizes": {k: len(splits[k][0]) for k in ("train", "val", "test")},
               "epochs": args.epochs, "results": results, "history": history},
              open(REPORT_DIR / "metrics.json", "w"), indent=2)
    np.save(REPORT_DIR / "test_probabilities.npy", prob_te)
    plots(results, history)
    print(json.dumps({k: {m: v[m] for m in ("exact_match", "macro_f1", "hamming_loss",
                                           "exact_match_by_n_defects", "normal_wafers_flagged")}
                      for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
