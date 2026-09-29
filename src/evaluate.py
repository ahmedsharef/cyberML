"""
evaluate.py
───────────
Model evaluation utilities: compute metrics, generate plots, write the
model_comparison_report.md.

Public API
──────────
    evaluate_model(clf, X_test, y_test, ...) → dict of metrics
    save_comparison_report(results, class_names)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.config import FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    model_name: str,
    is_binary: bool = False,
) -> dict:
    """
    Compute accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.

    Parameters
    ----------
    clf         : Fitted scikit-learn compatible estimator
    X_test      : Feature matrix
    y_test      : True labels (integer encoded)
    class_names : Human-readable class names (matches encoder order)
    model_name  : Used for plot filenames
    is_binary   : If True, use binary averaging for metrics

    Returns
    -------
    dict with all metric values + paths to saved plot files
    """
    y_pred = clf.predict(X_test)

    avg = "binary" if is_binary else "weighted"

    accuracy   = accuracy_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall     = recall_score(y_test, y_pred, average=avg, zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    # Only include labels that actually appear in y_test or y_pred to avoid
    # "Number of classes does not match size of target_names" error
    present_labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    present_names  = [
        class_names[i] for i in present_labels if i < len(class_names)
    ]
    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )

    # ROC-AUC (one-vs-rest for multi-class)
    roc_auc = None
    try:
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test)
            if is_binary:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                n_classes = len(class_names)
                # Only use labels present in both y_test and the model
                present_labels = sorted(set(y_test.tolist()))
                y_bin = label_binarize(y_test, classes=present_labels)
                # Map model class indices to present_labels positions
                clf_classes = list(clf.classes_) if hasattr(clf, "classes_") else list(range(y_prob.shape[1]))
                # Build prob matrix aligned to present_labels
                prob_cols = []
                for lbl in present_labels:
                    if lbl in clf_classes:
                        prob_cols.append(y_prob[:, clf_classes.index(lbl)])
                    else:
                        prob_cols.append(np.zeros(len(y_test)))
                import numpy as _np
                y_prob_aligned = _np.column_stack(prob_cols)
                if y_bin.shape[1] > 1:
                    roc_auc = roc_auc_score(
                        y_bin, y_prob_aligned,
                        average="weighted", multi_class="ovr"
                    )
    except Exception as exc:
        logger.warning("ROC-AUC computation failed for %s: %s", model_name, exc)

    logger.info(
        "%s → Acc=%.4f, wF1=%.4f, Prec=%.4f, Rec=%.4f, AUC=%s",
        model_name, accuracy, weighted_f1, precision, recall,
        f"{roc_auc:.4f}" if roc_auc is not None else "n/a",
    )

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm_path = _plot_confusion_matrix(
        y_test, y_pred, class_names, model_name
    )

    # ── ROC curve plot ────────────────────────────────────────────────────────
    roc_path = None
    if hasattr(clf, "predict_proba"):
        roc_path = _plot_roc_curves(
            y_test, clf.predict_proba(X_test), class_names, model_name, is_binary
        )

    return {
        "accuracy":     accuracy,
        "precision":    precision,
        "recall":       recall,
        "weighted_f1":  weighted_f1,
        "macro_f1":     macro_f1,
        "roc_auc":      roc_auc,
        "report":       report,
        "cm_path":      str(cm_path),
        "roc_path":     str(roc_path) if roc_path else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"cm_{model_name}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.debug("Saved confusion matrix → %s", out_path)
    return out_path


def _plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    model_name: str,
    is_binary: bool,
) -> Optional[Path]:
    """Plot ROC curves (one per class for multi-class, single for binary)."""
    from sklearn.metrics import roc_curve, auc

    try:
        fig, ax = plt.subplots(figsize=(8, 6))

        if is_binary:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", lw=2)
        else:
            present_labels = sorted(set(y_true.tolist()))
            y_bin = label_binarize(y_true, classes=present_labels)
            clf_classes = list(range(y_prob.shape[1]))
            for col_i, lbl in enumerate(present_labels):
                cname = class_names[lbl] if lbl < len(class_names) else str(lbl)
                prob_col = y_prob[:, lbl] if lbl < y_prob.shape[1] else y_prob[:, col_i]
                bin_col  = y_bin[:, col_i]
                fpr, tpr, _ = roc_curve(bin_col, prob_col)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})", lw=1.5)

        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {model_name}")
        ax.legend(loc="lower right", fontsize=8)
        plt.tight_layout()

        out_path = FIGURES_DIR / f"roc_{model_name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return out_path
    except Exception as exc:
        logger.warning("ROC plot failed for %s: %s", model_name, exc)
        return None


def _plot_model_comparison(results: dict, metric: str = "weighted_f1") -> Path:
    """Bar chart comparing all models on a given metric (multi-class results)."""
    names  = list(results.keys())
    values = [results[n]["multi"][metric] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, values, color="steelblue", edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()} (Multi-class)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = FIGURES_DIR / f"model_comparison_{metric}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_report(results: dict, class_names: list[str]) -> None:
    """
    Write reports/model_comparison_report.md with a full metrics table
    and inline links to all generated figures.
    """
    comp_path = _plot_model_comparison(results, "weighted_f1")
    acc_path  = _plot_model_comparison(results, "accuracy")

    lines = [
        "# Model Comparison Report",
        "",
        "Auto-generated by `src/evaluate.py` during training.",
        "",
        "## Multi-class Classification Results",
        "",
        "| Model | Accuracy | Precision | Recall | Weighted F1 | Macro F1 | ROC-AUC |",
        "|-------|----------|-----------|--------|-------------|----------|---------|",
    ]

    best_name = None
    best_f1   = -1.0

    for name, res in results.items():
        m = res["multi"]
        auc_str = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['weighted_f1']:.4f} | "
            f"{m['macro_f1']:.4f} | {auc_str} |"
        )
        if m["weighted_f1"] > best_f1:
            best_f1   = m["weighted_f1"]
            best_name = name

    lines += [
        "",
        f"**Best model (weighted F1): `{best_name}` — {best_f1:.4f}**",
        "",
        "## Binary Classification Results (Normal vs Attack)",
        "",
        "| Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |",
        "|-------|----------|-----------|--------|-------------|---------|",
    ]

    for name, res in results.items():
        b = res["binary"]
        auc_str = f"{b['roc_auc']:.4f}" if b["roc_auc"] is not None else "n/a"
        lines.append(
            f"| {name} | {b['accuracy']:.4f} | {b['precision']:.4f} | "
            f"{b['recall']:.4f} | {b['weighted_f1']:.4f} | {auc_str} |"
        )

    lines += [
        "",
        "## Model Comparison Charts",
        "",
        f"![Weighted F1 Comparison](figures/model_comparison_weighted_f1.png)",
        f"![Accuracy Comparison](figures/model_comparison_accuracy.png)",
        "",
        "## Confusion Matrices (Multi-class)",
        "",
    ]
    for name in results:
        lines.append(f"### {name}")
        lines.append(f"![{name} confusion matrix](figures/cm_{name}_multi.png)")
        lines.append("")

    lines += [
        "## ROC Curves (Multi-class)",
        "",
    ]
    for name in results:
        roc_path = results[name]["multi"].get("roc_path")
        if roc_path:
            fn = Path(roc_path).name
            lines.append(f"### {name}")
            lines.append(f"![{name} ROC](figures/{fn})")
            lines.append("")

    lines += [
        "## Detailed Classification Reports",
        "",
    ]
    for name, res in results.items():
        lines += [
            f"### {name} — Multi-class",
            "```",
            res["multi"]["report"],
            "```",
            "",
            f"### {name} — Binary",
            "```",
            res["binary"]["report"],
            "```",
            "",
        ]

    report_path = REPORTS_DIR / "model_comparison_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Comparison report saved → %s", report_path)
