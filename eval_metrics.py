"""
eval_metrics.py — MediGuard AI Rule Engine Evaluation
Compares fraud_type (rule engine prediction) vs fraud_label (ground truth)
"""

import json
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("output/claims_flagged.csv")
df["fraud_type"] = df["fraud_type"].fillna("CLEAN")

y_true = df["fraud_label"]
y_pred = df["fraud_type"]

CLASSES = ["CLEAN", "UPCODING", "UNBUNDLING", "ICD_CPT_MISMATCH", "MEDICALLY_UNNECESSARY"]
n_claims = len(df)

# ── 2. Overall metrics ────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
recall    = recall_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
f1        = f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)

# ── 3. Per-class metrics ──────────────────────────────────────────────────────
report = classification_report(
    y_true, y_pred,
    labels=CLASSES,
    output_dict=True,
    zero_division=0,
)
per_class = {
    cls: {
        "precision": report[cls]["precision"],
        "recall":    report[cls]["recall"],
        "f1":        report[cls]["f1-score"],
        "support":   int(report[cls]["support"]),
    }
    for cls in CLASSES
}

# ── 4. Confusion matrix heatmap ───────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    linecolor="#cccccc",
    ax=ax,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={"shrink": 0.8},
)

ax.set_title(
    "MediGuard AI — Rule Engine Confusion Matrix",
    fontsize=14,
    fontweight="bold",
    pad=16,
)
ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
ax.set_ylabel("True Label", fontsize=11, labelpad=10)
ax.tick_params(axis="x", labelrotation=30, labelsize=9)
ax.tick_params(axis="y", labelrotation=0,  labelsize=9)
plt.tight_layout()

heatmap_path = "output/confusion_matrix.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# ── 5. Console report ─────────────────────────────────────────────────────────
SEP = "=" * 55

print(SEP)
print("  MEDIAGUARD AI -- EVALUATION METRICS")
print(f"  Rule Engine vs Ground Truth ({n_claims} claims)")
print(SEP)
print(f"  Overall Accuracy:    {accuracy * 100:.1f}%")
print(f"  Weighted Precision:  {precision * 100:.1f}%")
print(f"  Weighted Recall:     {recall * 100:.1f}%")
print(f"  Weighted F1:         {f1 * 100:.1f}%")
print()
print("  Per-Class Results:")
print(f"  {'Class':<25} {'Precision':>10} {'Recall':>9} {'F1':>9} {'Support':>9}")
print("  " + "-" * 65)
for cls in CLASSES:
    pc = per_class[cls]
    print(
        f"  {cls:<25} {pc['precision']*100:>9.1f}% "
        f"{pc['recall']*100:>8.1f}% "
        f"{pc['f1']*100:>8.1f}% "
        f"{pc['support']:>9}"
    )
print()
print(f"  Confusion Matrix saved -> {heatmap_path}")
print(SEP)

# ── 6. Save JSON ──────────────────────────────────────────────────────────────
metrics_out = {
    "n_claims": n_claims,
    "overall": {
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    },
    "per_class": {
        cls: {k: round(v, 4) if isinstance(v, float) else v for k, v in vals.items()}
        for cls, vals in per_class.items()
    },
    "confusion_matrix": {
        "labels": CLASSES,
        "matrix": cm.tolist(),
    },
    "heatmap_path": heatmap_path,
}

json_path = "output/eval_metrics.json"
with open(json_path, "w") as f:
    json.dump(metrics_out, f, indent=2)

print(f"\n  Metrics JSON saved  -> {json_path}")
