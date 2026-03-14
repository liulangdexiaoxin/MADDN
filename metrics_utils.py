import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, f1_score, recall_score,
    confusion_matrix, roc_curve
)
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

__all__ = [
    'compute_auc', 'compute_precision', 'compute_specificity', 'compute_f1',
    'compute_recall', 'compute_confusion_matrix', 'compute_basic_metrics',
    'compute_epoch_metrics', 'plot_confusion_matrix_figure', 'plot_roc_figure',
    'compute_per_class_metrics', 'plot_per_class_bars', 'compute_macro_weighted_summary', 'plot_class_support_bar'
]

def compute_auc(y_true, probs) -> float:
    try:
        probs_arr = np.array(probs)
        if probs_arr.ndim == 1:  # already positive class probs
            return roc_auc_score(y_true, probs_arr)
        if probs_arr.shape[1] == 2:
            return roc_auc_score(y_true, probs_arr[:, 1])
        return roc_auc_score(y_true, probs_arr, multi_class='ovr')
    except Exception:
        return float('nan')

def compute_precision(y_true, y_pred) -> float:
    try:
        return precision_score(y_true, y_pred, average='weighted', zero_division=0)
    except Exception:
        return 0.0

def compute_specificity(y_true, y_pred) -> float:
    try:
        cm = confusion_matrix(y_true, y_pred)
        specs = []
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specs.append(spec)
        return float(np.mean(specs)) if specs else 0.0
    except Exception:
        return 0.0

def compute_f1(y_true, y_pred, average: str = 'weighted') -> float:
    """计算 F1 分数

    Parameters
    ----------
    y_true : array-like
        真实标签
    y_pred : array-like
        预测标签
    average : str, default 'weighted'
        传递给 sklearn.f1_score 的 average 参数，可选: 'weighted', 'macro', 'micro'.
    """
    try:
        return f1_score(y_true, y_pred, average=average, zero_division=0)
    except Exception:
        return 0.0

def compute_recall(y_true, y_pred) -> float:
    try:
        return recall_score(y_true, y_pred, average='macro', zero_division=0)
    except Exception:
        return 0.0

def compute_confusion_matrix(y_true, y_pred):
    try:
        return confusion_matrix(y_true, y_pred)
    except Exception:
        return None

def compute_basic_metrics(y_true, y_pred, probs) -> Dict[str, float]:
    return {
        'precision': compute_precision(y_true, y_pred),
        'specificity': compute_specificity(y_true, y_pred),
        'f1': compute_f1(y_true, y_pred),
        'recall': compute_recall(y_true, y_pred),
        'auc': compute_auc(y_true, probs)
    }

def compute_epoch_metrics(y_true, y_pred, probs) -> Dict[str, float]:
    return compute_basic_metrics(y_true, y_pred, probs)

def plot_confusion_matrix_figure(cm, class_names=None, title: str = 'Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('Pred')
    ax.set_ylabel('True')
    if class_names:
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def plot_roc_figure(y_true, probs, title: str = 'ROC Curve'):
    try:
        probs_arr = np.array(probs)
        if probs_arr.ndim == 1 or probs_arr.shape[1] == 2:
            # binary
            if probs_arr.ndim > 1:
                pos = probs_arr[:,1]
            else:
                pos = probs_arr
            fpr, tpr, _ = roc_curve(y_true, pos)
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
            ax.legend(loc='lower right')
            return fig
        else:
            # multi-class: micro-average ROC
            from sklearn.preprocessing import label_binarize
            y_true_arr = np.array(y_true)
            classes = np.unique(y_true_arr)
            if len(classes) < 2:
                return None
            y_bin = label_binarize(y_true_arr, classes=classes)
            # probs_arr shape: (n_samples, n_classes)
            if probs_arr.shape[1] != len(classes):
                return None
            # micro-average
            from sklearn.metrics import roc_curve as _roc_curve, auc as _auc
            fpr, tpr, _ = _roc_curve(y_bin.ravel(), probs_arr.ravel())
            micro_auc = _auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(fpr, tpr, label=f'Micro-AUC={micro_auc:.4f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(title)
            ax.legend(loc='lower right')
            return fig
    except Exception:
        return None

def compute_per_class_metrics(y_true, y_pred, class_names=None):
    """返回每个类别的 precision / recall / f1 (support) 字典
    返回: { 'classes': [...], 'precision': [...], 'recall': [...], 'f1': [...], 'support': [...] }
    """
    from sklearn.metrics import precision_recall_fscore_support
    try:
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        if class_names is None:
            classes = list(range(len(p)))
        else:
            classes = class_names
        return {
            'classes': classes,
            'precision': p.tolist(),
            'recall': r.tolist(),
            'f1': f1.tolist(),
            'support': s.tolist()
        }
    except Exception:
        return {
            'classes': class_names or [],
            'precision': [], 'recall': [], 'f1': [], 'support': []
        }

def plot_per_class_bars(per_class_dict, title='Per-class Metrics'):
    try:
        classes = per_class_dict['classes']
        p = per_class_dict['precision']
        r = per_class_dict['recall']
        f1 = per_class_dict['f1']
        import numpy as np
        x = np.arange(len(classes))
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(4, len(classes)*0.7), 4))
        ax.bar(x - w, p, width=w, label='Precision')
        ax.bar(x,     r, width=w, label='Recall')
        ax.bar(x + w, f1, width=w, label='F1')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception:
        return None

def compute_macro_weighted_summary(y_true, y_pred):
    """返回 macro / weighted 聚合指标 (precision/recall/f1)"""
    from sklearn.metrics import precision_recall_fscore_support
    try:
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        return {
            'macro': {'precision': p_m, 'recall': r_m, 'f1': f1_m},
            'weighted': {'precision': p_w, 'recall': r_w, 'f1': f1_w}
        }
    except Exception:
        return {
            'macro': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
            'weighted': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        }

def plot_class_support_bar(per_class_dict, title='Class Support Distribution'):
    try:
        import numpy as np
        classes = per_class_dict['classes']
        support = per_class_dict['support']
        total = sum(support) if support else 0
        if total == 0:
            return None
        ratios = [s / total for s in support]
        x = np.arange(len(classes))
        fig, ax = plt.subplots(figsize=(max(4, len(classes)*0.7), 3.5))
        ax.bar(x, ratios, width=0.6, color='#5B8FF9')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1.05)
        for xi, ri in zip(x, ratios):
            ax.text(xi, ri + 0.01, f'{ri*100:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_title(title)
        fig.tight_layout()
        return fig
    except Exception:
        return None
