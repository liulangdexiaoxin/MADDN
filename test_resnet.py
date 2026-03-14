# 直接使用 Config 中的原始配置，不再通过命令行覆写模型类型/批大小/预训练路径等
import os
import sys
import matplotlib
os.environ.setdefault('MPLBACKEND', 'Agg')
matplotlib.use('Agg')

import torch
import numpy as np
from tqdm import tqdm
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from config import Config
from data_loader import get_data_loaders
from model_builder import build_backbone
from utils import load_checkpoint
try:
    from metrics_utils import (
        compute_auc, compute_f1, compute_precision,
        compute_specificity, compute_confusion_matrix,
        compute_per_class_metrics, plot_per_class_bars,
        compute_macro_weighted_summary, plot_class_support_bar,
        compute_recall, plot_roc_figure, plot_pr_curve, plot_recall_threshold_curve
    )
except ImportError:
    # 回退：旧版 metrics_utils 不含 PR/Recall 曲线函数
    from metrics_utils import (
        compute_auc, compute_f1, compute_precision,
        compute_specificity, compute_confusion_matrix,
        compute_per_class_metrics, plot_per_class_bars,
        compute_macro_weighted_summary, plot_class_support_bar,
        compute_recall, plot_roc_figure
    )
    def plot_pr_curve(*args, **kwargs):
        # 内嵌简化实现（仅二分类）避免修改其它文件
        try:
            y_true, probs = args[0], args[1]
            import numpy as _np
            from sklearn.metrics import precision_recall_curve, average_precision_score
            probs_arr = _np.array(probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                pos = probs_arr[:,1]
            elif probs_arr.ndim == 1:
                pos = probs_arr
            else:
                return None, None
            prec, rec, _ = precision_recall_curve(y_true, pos)
            ap = average_precision_score(y_true, pos)
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(4,4))
            ax.plot(rec, prec, label=f'AP={ap:.4f}')
            ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title(kwargs.get('title','Precision-Recall Curve'))
            ax.set_xlim(0,1); ax.set_ylim(0,1.05)
            ax.legend(loc='lower left')
            return fig, ap
        except Exception:
            return None, None
    def plot_recall_threshold_curve(*args, **kwargs):
        try:
            y_true, probs = args[0], args[1]
            import numpy as _np
            from sklearn.metrics import recall_score
            probs_arr = _np.array(probs)
            if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
                pos = probs_arr[:,1]
            elif probs_arr.ndim == 1:
                pos = probs_arr
            else:
                return None
            thresholds = _np.linspace(0,1,101)
            recalls = []
            for th in thresholds:
                preds = (pos >= th).astype(int)
                try:
                    rec = recall_score(y_true, preds, zero_division=0)
                except Exception:
                    rec = 0.0
                recalls.append(rec)
            import matplotlib.pyplot as _plt
            fig, ax = _plt.subplots(figsize=(4,4))
            ax.plot(thresholds, recalls, color='#E67E22')
            ax.set_xlabel('Threshold'); ax.set_ylabel('Recall (Positive)')
            ax.set_title(kwargs.get('title','Recall vs Threshold'))
            ax.set_xlim(0,1); ax.set_ylim(0,1.05)
            return fig
        except Exception:
            return None

def evaluate_resnet(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing (ResNet)')
        for batch in pbar:
            # 兼容 MRIDataset 返回 (data, label, path, domain) 或 (data, label, path)
            if isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    continue  # 跳过异常批次
                data = batch[0]
                target = batch[1]
            else:
                # 不符合预期结构，跳过
                continue
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)
    accuracy = (all_preds_np == all_targets_np).mean() * 100.0
    auc = compute_auc(all_targets_np, all_probs)
    weighted_f1 = compute_f1(all_targets_np, all_preds_np, average='weighted')
    precision_w = compute_precision(all_targets_np, all_preds_np)
    specificity_macro = compute_specificity(all_targets_np, all_preds_np)
    cm = compute_confusion_matrix(all_targets_np, all_preds_np)
    report = classification_report(all_targets_np, all_preds_np, target_names=class_names, output_dict=True)
    # 召回率 (macro) 以及 weighted 召回率
    recall_macro = compute_recall(all_targets_np, all_preds_np)
    recall_weighted = report.get('weighted avg', {}).get('recall', float('nan'))
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_weighted': weighted_f1,
        'precision_weighted': precision_w,
        'specificity_macro': specificity_macro,
        'confusion_matrix': cm,
        'report': report,
        'targets': all_targets_np,
        'probs': np.array(all_probs),
        'preds': all_preds_np,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted
    }

def save_confusion_matrix(cm, class_names, out_dir):
    try:
        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
        ax.set_title('Test Confusion Matrix')
        ax.set_xlabel('Pred'); ax.set_ylabel('True')
        fig.savefig(os.path.join(out_dir, 'test_confusion_matrix.png'))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: failed to save confusion matrix figure: {e}")

def parse_args():
    ap = argparse.ArgumentParser(description='Test 3D ResNet classification performance (use Config only)')
    ap.add_argument('--checkpoint', type=str, default=None, help='微调后的分类模型权重文件路径')
    ap.add_argument('--output_dir', type=str, default=None, help='结果输出目录 (不指定则使用 config.training.log_dir)')
    return ap.parse_args()

def main():
    args = parse_args()
    config = Config()
    device = config.device
    # 使用 config 中原始数据设置
    _, _, test_loader = get_data_loaders(config)
    # 构建模型（内部已根据 config.backbone.* 决定结构与是否预训练）
    model = build_backbone(config).to(device)
    # 可选加载分类微调 checkpoint
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[Load] 分类 checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
    else:
        print("[Error] 未提供微调 checkpoint，使用当前模型参数进行评估。")
        sys.exit(1)
    # 类名优先使用 config.class_names
    if hasattr(config, 'class_names') and config.class_names:
        class_names = list(config.class_names)
    else:
        class_names = ['AD', 'CN'] if config.backbone.num_classes == 2 else [f'C{i}' for i in range(config.backbone.num_classes)]
    results = evaluate_resnet(model, test_loader, device, class_names)
    out_dir = args.output_dir or os.path.join(config.training.log_dir, f"test_resnet_{config.backbone.model_type}")
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*50)
    print(f"ResNet ({config.backbone.model_type}) 测试集性能")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.2f}%")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Weighted F1: {results['f1_weighted']:.4f}")
    print(f"Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"Specificity (macro): {results['specificity_macro']:.4f}")
    print(f"Recall (macro): {results['recall_macro']:.4f}")
    if not np.isnan(results['recall_weighted']):
        print(f"Recall (weighted): {results['recall_weighted']:.4f}")
    print("\n分类报告:")
    from sklearn.metrics import classification_report as _cr
    print(_cr(results['targets'], results['preds'], target_names=class_names))
    print("\n混淆矩阵:")
    print(results['confusion_matrix'])
    save_confusion_matrix(results['confusion_matrix'], class_names, out_dir)
    # 保存 ROC 曲线图
    try:
        fig_roc = plot_roc_figure(results['targets'], results['probs'], title='Test ROC Curve')
        if fig_roc:
            fig_roc.savefig(os.path.join(out_dir, 'test_roc_curve.png'))
            plt.close(fig_roc)
    except Exception as e:
        print(f"Warning: failed to save ROC curve figure: {e}")
    # 保存 PR 曲线与 Recall-Threshold 曲线
    try:
        fig_pr, _ = plot_pr_curve(results['targets'], results['probs'], title='Test Precision-Recall Curve')
        if fig_pr:
            fig_pr.savefig(os.path.join(out_dir, 'test_pr_curve.png'))
            plt.close(fig_pr)
    except Exception as e:
        print(f"Warning: failed to save PR curve figure: {e}")
    try:
        fig_rth = plot_recall_threshold_curve(results['targets'], results['probs'], title='Test Recall vs Threshold')
        if fig_rth:
            fig_rth.savefig(os.path.join(out_dir, 'test_recall_threshold.png'))
            plt.close(fig_rth)
    except Exception as e:
        print(f"Warning: failed to save Recall-Threshold figure: {e}")
    try:
        per_cls = compute_per_class_metrics(results['targets'], results['preds'], class_names=class_names)
        summary = compute_macro_weighted_summary(results['targets'], results['preds'])
        fig_bar = plot_per_class_bars(per_cls, title='Test Per-class Metrics')
        if fig_bar:
            fig_bar.savefig(os.path.join(out_dir, 'test_per_class_metrics.png'))
            plt.close(fig_bar)
        fig_sup = plot_class_support_bar(per_cls, title='Test Class Support')
        if fig_sup:
            fig_sup.savefig(os.path.join(out_dir, 'test_class_support.png'))
            plt.close(fig_sup)
    except Exception:
        per_cls, summary = None, None
    out_json = {
        'model_type': config.backbone.model_type,
        'accuracy': results['accuracy'],
        'auc': results['auc'],
        'f1_weighted': results['f1_weighted'],
        'precision_weighted': results['precision_weighted'],
        'specificity_macro': results['specificity_macro'],
    'recall_macro': results['recall_macro'],
    'recall_weighted': results['recall_weighted'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['report'],
        'per_class': per_cls,
        'macro_weighted_summary': summary
    }
    with open(os.path.join(out_dir, 'summary_resnet_test.json'), 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f"\n结果已保存到: {out_dir}")

if __name__ == '__main__':
    main()