"""Pretrain ResNet training script

Notes on F1 logging:
    - Batch-level F1 (Train/F1_batch, Val/F1_batch) uses macro average for stability on small batches.
    - Epoch/Test-level F1 (Train/F1, Val/F1, Test/F1) uses weighted average to reflect class imbalance.

Backend note:
    为避免 Windows 下默认 TkAgg 在多线程/多进程 DataLoader (num_workers>0) 及 figure 回收时触发
    "RuntimeError: main thread is not in main loop" 和 "Tcl_AsyncDelete"，这里强制使用无界面后端 Agg。
"""
import os as _os
_os.environ.setdefault('MPLBACKEND', 'Agg')  # 优先环境变量
import matplotlib
matplotlib.use('Agg')  # 强制非交互后端，避免 Tk 相关崩溃
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import os
import time
import json
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, recall_score, roc_curve, precision_score, precision_recall_curve, average_precision_score
from metrics_utils import (
    compute_auc, compute_precision, compute_specificity, compute_f1,
    compute_recall, compute_confusion_matrix, plot_confusion_matrix_figure,
    plot_roc_figure, compute_per_class_metrics, plot_per_class_bars,
    compute_macro_weighted_summary, plot_class_support_bar
)
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
import seaborn as sns
import matplotlib.pyplot as plt  # 已使用 Agg 后端
import datetime  # 新增

# 常量，统一标签与键
ACC_LABEL = 'Accuracy (%)'
WEIGHTED_AVG_KEY = 'weighted avg'

from config import Config
from data_loader import get_data_loaders
from model_builder import build_backbone, count_parameters
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, FocalLoss, plot_confusion_matrix  # 添加导入

def log_binary_roc_pr(writer, y_true, y_probs, tag_prefix, step, positive_index=1):
    """记录二分类 ROC 与 PR 曲线及 AUC/AP 标量。
    参数:
        writer: SummaryWriter
        y_true: list/array 标签(0/1)
        y_probs: 2D array [N, C] softmax 概率
        tag_prefix: 写入前缀 ('Val/Best' / 'Test' 等)
        step: epoch 或自定义 step
        positive_index: 正类索引，默认 1
    """
    if writer is None:
        return
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
        arr_true = np.array(y_true)
        arr_probs = np.array(y_probs)
        if arr_probs.ndim != 2 or arr_probs.shape[1] <= positive_index:
            return
        pos_scores = arr_probs[:, positive_index]
        fpr, tpr, _ = roc_curve(arr_true, pos_scores)
        roc_auc_val = compute_auc(arr_true, arr_probs)
        fig_roc, axr = plt.subplots(figsize=(4,4))
        axr.plot(fpr, tpr, label=f'AUC={roc_auc_val:.4f}')
        axr.plot([0,1],[0,1],'--', color='gray')
        axr.set_xlabel('FPR'); axr.set_ylabel('TPR'); axr.set_title('ROC Curve'); axr.legend(loc='lower right')
        writer.add_figure(f'{tag_prefix}/ROC', fig_roc, step)
        plt.close(fig_roc)
        precision_vals, recall_vals, _ = precision_recall_curve(arr_true, pos_scores)
        ap = average_precision_score(arr_true, pos_scores)
        fig_pr, axpr = plt.subplots(figsize=(4,4))
        axpr.plot(recall_vals, precision_vals, label=f'AP={ap:.4f}')
        axpr.set_xlabel('Recall'); axpr.set_ylabel('Precision'); axpr.set_title('PR Curve'); axpr.legend(loc='lower left')
        writer.add_figure(f'{tag_prefix}/PR', fig_pr, step)
        plt.close(fig_pr)
        writer.add_scalar(f'{tag_prefix}/ROC_AUC', roc_auc_val, step)
        writer.add_scalar(f'{tag_prefix}/AP', ap, step)
    except Exception:
        pass

def log_epoch_metrics(writer, epoch: int, prefix: str, metrics: dict):
    """统一写入一个阶段的 epoch 级指标
    metrics: key->value (scalars)
    """
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f'{prefix}/{k}', v, epoch)

def create_optimizer(model, config):
    """创建优化器"""
    if config.training.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    return optimizer

def create_scheduler(optimizer, config, train_loader):
    """创建学习率调度器"""
    if config.training.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs * len(train_loader),
            eta_min=config.training.min_lr
        )
    elif config.training.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif config.training.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config.training.patience,
            factor=config.training.gamma,
            min_lr=config.training.min_lr
        )
    else:
        scheduler = None
    
    return scheduler

def create_loss_fn(config):
    """创建损失函数"""
    if config.training.loss_fn == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    elif config.training.loss_fn == "focal":
        return FocalLoss(alpha=config.training.focal_alpha, gamma=config.training.focal_gamma)
    else:
        raise ValueError(f"Unsupported loss function: {config.training.loss_fn}")

def train_epoch(model, train_loader, optimizer, criterion, scheduler, epoch, config, writer=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []
    grad_norm_accumulate = []
    batch_losses = []  # 新增

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} [Train]')
    for batch_idx, (data, target, _) in enumerate(pbar):
        data, target = data.to(config.device), target.to(config.device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        batch_losses.append(loss.item())  # 记录每个batch的损失
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        # 计算梯度范数
        if writer is not None:
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            grad_norm = total_norm_sq ** 0.5
        else:
            grad_norm = None
        optimizer.step()
        
        # 更新学习率
        if scheduler and config.training.lr_scheduler == "cosine":
            scheduler.step()
        
        # 记录指标
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        # 记录召回率
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)
        recall = compute_recall(target.cpu().numpy(), preds.cpu().numpy())
        recalls.update(recall, data.size(0))
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        # TensorBoard 训练 batch 级日志（统一 Train/Batch/ 前缀）
        if writer is not None:
            base_offset = getattr(getattr(config, 'training', None), 'prev_global_step', 0)
            start_epoch_cfg = getattr(getattr(config, 'training', None), 'start_epoch', 0)
            rel_epoch = epoch - start_epoch_cfg
            global_step = base_offset + rel_epoch * len(train_loader) + batch_idx
            f1_batch = compute_f1(target.cpu().numpy(), preds.cpu().numpy(), average='macro')
            writer.add_scalar('Train/Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/Accuracy', acc1.item(), global_step)
            writer.add_scalar('Train/Batch/Recall', recall, global_step)
            writer.add_scalar('Train/Batch/F1_macro', f1_batch, global_step)
            if grad_norm is not None:
                writer.add_scalar('Train/Batch/GradNorm', grad_norm, global_step)
        if grad_norm is not None:
            grad_norm_accumulate.append(grad_norm)
        pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{top1.avg:.2f}%', 'Recall': f'{recalls.avg:.2f}'})
    # 计算AUC 使用概率
    auc_val = compute_auc(all_targets, all_probs)
    grad_norm_epoch_mean = float(np.mean(grad_norm_accumulate)) if grad_norm_accumulate else 0.0
    if writer is not None:
        writer.add_scalar('Train/GradNorm_epoch_mean', grad_norm_epoch_mean, epoch)
    # Precision & Specificity (epoch)
    train_precision = compute_precision(all_targets, all_preds)
    train_specificity = compute_specificity(all_targets, all_preds)
    if writer is not None:
        writer.add_scalar('Train/Precision', train_precision, epoch)
        writer.add_scalar('Train/Specificity', train_specificity, epoch)
    return losses.avg, top1.avg, recalls.avg, auc_val, batch_losses, all_targets, all_preds, all_probs

def validate(model, val_loader, criterion, config, epoch=None, writer=None):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (data, target, _) in enumerate(pbar):
            data, target = data.to(config.device), target.to(config.device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 记录指标
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            recall = compute_recall(target.cpu().numpy(), preds.cpu().numpy())
            recalls.update(recall, data.size(0))
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            # 按需求：移除验证集 batch 日志
            pbar.set_postfix({'Loss': f'{losses.avg:.4f}', 'Acc': f'{top1.avg:.2f}%', 'Recall': f'{recalls.avg:.2f}'})
    auc_val = compute_auc(all_targets, all_probs)
    # Precision & Specificity (validation)
    val_precision = compute_precision(all_targets, all_preds)
    val_specificity = compute_specificity(all_targets, all_preds)
    if writer is not None and epoch is not None:
        writer.add_scalar('Val/Precision', val_precision, epoch)
        writer.add_scalar('Val/Specificity', val_specificity, epoch)
    return losses.avg, top1.avg, recalls.avg, auc_val, all_targets, all_preds, all_probs

def test_model(model, test_loader, config, class_names=['AD', 'CN'], save_dir=None):
    """测试模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target, _ in pbar:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    cm = compute_confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    auc = compute_auc(all_targets, all_probs)
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100

    # 绘制混淆矩阵
    out_dir = save_dir or config.training.log_dir
    plot_confusion_matrix(cm, class_names, 
        save_path=os.path.join(out_dir, 'confusion_matrix.png'))

    # 打印 weighted f1-score
    weighted_f1 = compute_f1(all_targets, all_preds)
    print(f"\nWeighted F1-score: {weighted_f1:.4f}")

    # 打印结果
    print("\n" + "="*50)
    print("测试集性能评估")
    print("="*50)
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")  # 新增
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("\n混淆矩阵:")
    print(cm)
    
    # 保存结果
    results = {
        'test_accuracy': test_acc,
        'auc': auc,
        'weighted_f1': weighted_f1,  # 新增
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Precision & Specificity (test)
    try:
        test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    except Exception:
        test_precision = 0.0
    try:
        specs = []
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specs.append(spec)
        test_specificity = float(np.mean(specs)) if specs else 0.0
    except Exception:
        test_specificity = 0.0
    return test_acc, auc, cm, report, all_probs, test_precision, test_specificity, all_targets

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制并保存学习曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_full_learning_curves(
    train_batch_losses, train_epoch_losses, val_epoch_losses, val_epoch_accs, save_dir
):
    """绘制并保存详细学习曲线"""
    steps_batch = range(1, len(train_batch_losses) + 1)
    steps_epoch = range(1, len(train_epoch_losses) + 1)
    plt.figure(figsize=(14, 10))

    # Train/Batch Loss
    plt.subplot(2, 2, 1)
    plt.plot(steps_batch, train_batch_losses, label='Train/Batch Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train/Batch Loss')
    plt.legend()

    # Train/Epoch Loss
    plt.subplot(2, 2, 2)
    plt.plot(steps_epoch, train_epoch_losses, label='Train/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Epoch Loss')
    plt.legend()

    # Val/Epoch Loss
    plt.subplot(2, 2, 3)
    plt.plot(steps_epoch, val_epoch_losses, label='Val/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Val/Epoch Loss')
    plt.legend()

    # Val/Epoch Accuracy
    plt.subplot(2, 2, 4)
    plt.plot(steps_epoch, val_epoch_accs, label='Val/Epoch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Val/Epoch Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'full_learning_curves.png'))
    plt.close()

def plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, save_dir):
    """
        绘制准确率、F1 score、召回率曲线
    """
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1-score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_curves.png'))
    plt.close()

def main():
    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = Config()
    config.backbone.pretrained = False  # 确保从头开始训练

    # 打印训练参数
    print(f"Batch size: {config.data.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 打印设备信息
    print(f"Using device: {config.device}")
    if config.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 创建模型
    model = build_backbone(config)
    print(f"device is: {config.device}")
    model = model.to(config.device)
    
    # 打印模型参数数量
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # 确认模型是否在GPU上
    print(f"Model is on: {next(model.parameters()).device}")
    
    # 创建优化器、损失函数和调度器
    optimizer = create_optimizer(model, config)
    criterion = create_loss_fn(config)
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_acc = 0
    prev_log_dir = None
    prev_global_step = 0
    if config.training.resume:
        ckpt_raw = torch.load(config.training.resume, map_location=config.device)
        start_epoch, best_acc = load_checkpoint(
            config.training.resume, model, optimizer, scheduler
        )
        prev_log_dir = ckpt_raw.get('log_dir', None)
        prev_global_step = ckpt_raw.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # TensorBoard 准备: 复用旧 log_dir（若存在），否则新建
    if prev_log_dir is not None and os.path.isdir(prev_log_dir):
        log_dir = prev_log_dir
        print(f"[Resume] Reusing existing log_dir: {log_dir}")
    else:
        run_name = f"pretrain_resnet_bs{config.data.batch_size}_lr{config.training.learning_rate}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join(config.training.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[New Run] Created log_dir: {log_dir}")
    if prev_global_step > 0:
        writer = SummaryWriter(log_dir=log_dir, purge_step=prev_global_step)
    else:
        writer = SummaryWriter(log_dir=log_dir)
    # 记录配置
    try:
        writer.add_text('Config', json.dumps(dict(config.__dict__.items()), indent=2))
    except Exception:
        pass
    
    # 在训练循环前定义
    train_batch_losses = []
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []
    train_recalls, val_recalls = [], []
    best_state = {'epoch': -1, 'val_acc': 0, 'val_auc': 0, 'val_f1': 0, 'val_recall': 0}

    # 训练循环
    for epoch in range(start_epoch, config.training.num_epochs):
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_auc, batch_losses, train_targets, train_preds, _ = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, config, writer
        )
        val_loss, val_acc, val_recall, val_auc, val_targets, val_preds, val_probs = validate(model, val_loader, criterion, config, epoch, writer)

        train_f1 = compute_f1(train_targets, train_preds, average='weighted')
        val_f1 = compute_f1(val_targets, val_preds, average='weighted')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_batch_losses.extend(batch_losses)

        # TensorBoard 每 epoch 写入（统一命名）
        log_epoch_metrics(writer, epoch, 'Train', {
            'Loss': train_loss,
            'Accuracy': train_acc,
            'F1_weighted': train_f1,
            'Recall': train_recall,
            'AUC': train_auc,
        })
        log_epoch_metrics(writer, epoch, 'Val', {
            'Loss': val_loss,
            'Accuracy': val_acc,
            'F1_weighted': val_f1,
            'Recall': val_recall,
            'AUC': val_auc,
        })
        writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], epoch)
        # 记录 epoch 平均梯度范数（可选：用最后一次 grad_norm 代表）
        # if 'grad_norm' in locals() and grad_norm is not None:
        #     writer.add_scalar('Train/GradNorm_epoch_last', grad_norm, epoch)
        if (epoch + 1) % 5 == 0:
            writer.flush()

        # 按配置间隔记录参数/梯度直方图
        try:
            lg = getattr(config, 'logging', None)
            if lg and lg.hist_interval > 0 and ((epoch + 1) % lg.hist_interval == 0):
                for name, param in model.named_parameters():
                    if getattr(lg, 'enable_param_hist', True):
                        writer.add_histogram(f'Params/{name}', param.detach().cpu().numpy(), epoch)
                    if getattr(lg, 'enable_grad_hist', True) and param.grad is not None:
                        writer.add_histogram(f'Grads/{name}', param.grad.detach().cpu().numpy(), epoch)
        except Exception:
            pass

        # 更新学习率（对于plateau调度器）
        if scheduler and config.training.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        
        # 保存最佳模型
        is_best = val_acc > best_state['val_acc']
        if is_best:
            best_state.update({
                'epoch': epoch,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_f1': val_f1,
                'val_recall': val_recall,
            })
            best_acc = val_acc
            # 最佳epoch记录混淆矩阵与 ROC 曲线 (+ 可选 per-class 条形图与 JSON)
            try:
                cm_best = confusion_matrix(val_targets, val_preds)
                fig_cm, ax = plt.subplots(figsize=(4,4))
                sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
                ax.set_title('Best Val Confusion Matrix')
                ax.set_xlabel('Pred')
                ax.set_ylabel('True')
                writer.add_figure('Val/ConfusionMatrix_best', fig_cm, epoch)
                plt.close(fig_cm)
                # per-class metrics 可选导出
                if getattr(getattr(config, 'logging', None), 'export_per_class', True):
                    try:
                        per_cls = compute_per_class_metrics(val_targets, val_preds)
                        # 条形图（指标）
                        fig_bar = plot_per_class_bars(per_cls, title='Best Val Per-class Metrics')
                        if fig_bar:
                            writer.add_figure('Val/PerClassMetrics_best', fig_bar, epoch)
                            plt.close(fig_bar)
                        # 支持分布
                        fig_sup = plot_class_support_bar(per_cls, title='Best Val Class Support')
                        if fig_sup:
                            writer.add_figure('Val/ClassSupport_best', fig_sup, epoch)
                            plt.close(fig_sup)
                        # 汇总 macro/weighted
                        summary = compute_macro_weighted_summary(val_targets, val_preds)
                        try:
                            import json as _json
                            per_cls_out = {**per_cls, 'epoch': epoch, 'type': 'best_val', 'summary': summary}
                            with open(os.path.join(log_dir, 'best_val_per_class_metrics.json'), 'w') as fpc:
                                _json.dump(per_cls_out, fpc, indent=2)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
            # 最佳验证集二分类 ROC/PR
            if len(set(val_targets)) == 2:
                pos_idx = getattr(getattr(config, 'training', None), 'positive_index', 1)
                log_binary_roc_pr(writer, val_targets, val_probs, 'Val/Best', epoch, pos_idx)
        else:
            best_acc = max(val_acc, best_acc)
        
        # 保存检查点
        if (epoch + 1) % config.training.save_freq == 0 or is_best:
            global_step_ckpt = (epoch + 1) * len(train_loader)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'config': config.__dict__,
                'log_dir': log_dir,
                'global_step': global_step_ckpt
            }, is_best, config.training.checkpoint_dir)
        
    # 进度打印
    print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.2f}, Train AUC: {train_auc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.2f}, Val AUC: {val_auc:.4f}, Best Acc: {best_acc:.2f}%')
    # 超参数与最终指标写入 (hparams)
    # 推迟 hparams 与 writer 关闭到测试之后
    # 绘制详细学习曲线
    plot_full_learning_curves(
        train_batch_losses, train_losses, val_losses, val_accs, log_dir
    )
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, log_dir)
    # 绘制指标曲线
    plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, log_dir)
    
    # 训练结束时间
    end_time = datetime.datetime.now()
    print(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"训练总耗时: {str(duration)}")
    
    # 训练完成后，加载最佳模型并在测试集上评估
    print("\n训练完成，开始在测试集上评估最佳模型...")
    
    # 加载最佳模型
    best_model_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        load_checkpoint(best_model_path, model)
    else:
        print("未找到最佳模型，使用最终模型进行评估")
    
    # 在测试集上评估
    test_acc, test_auc, cm, _, test_probs, test_precision, test_specificity, test_targets = test_model(model, test_loader, config, save_dir=log_dir)
    # 独立测试 step，避免与训练 epoch 混淆
    test_step = config.training.num_epochs
    # Test 结果写入 TensorBoard + per-class metrics
    test_f1 = compute_f1(test_targets, np.argmax(np.array(test_probs), axis=1), average='weighted') if len(test_probs) else float('nan')
    try:
        # Confusion Matrix
        fig_cm, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
        ax.set_title('Test Confusion Matrix')
        ax.set_xlabel('Pred'); ax.set_ylabel('True')
        writer.add_figure('Test/ConfusionMatrix', fig_cm, test_step)
        plt.close(fig_cm)
    except Exception:
        pass
    # 测试集 ROC/PR
    if len(set(test_targets)) == 2:
        pos_idx = getattr(getattr(config, 'training', None), 'positive_index', 1)
        log_binary_roc_pr(writer, test_targets, test_probs, 'Test', test_step, pos_idx)
    # Scalars
    writer.add_scalar('Test/Accuracy', test_acc, test_step)
    writer.add_scalar('Test/AUC', test_auc if not np.isnan(test_auc) else 0.0, test_step)
    writer.add_scalar('Test/F1_weighted', test_f1, test_step)
    writer.add_scalar('Test/Precision', test_precision, test_step)
    writer.add_scalar('Test/Specificity', test_specificity, test_step)
    # per-class metrics
    if getattr(getattr(config, 'logging', None), 'export_per_class', True):
        try:
            per_cls_test = compute_per_class_metrics(test_targets, np.argmax(np.array(test_probs), axis=1))
            fig_bar_test = plot_per_class_bars(per_cls_test, title='Test Per-class Metrics')
            if fig_bar_test:
                writer.add_figure('Test/PerClassMetrics', fig_bar_test, test_step)
                plt.close(fig_bar_test)
            fig_sup_test = plot_class_support_bar(per_cls_test, title='Test Class Support')
            if fig_sup_test:
                writer.add_figure('Test/ClassSupport', fig_sup_test, test_step)
                plt.close(fig_sup_test)
            summary_test = compute_macro_weighted_summary(test_targets, np.argmax(np.array(test_probs), axis=1))
            try:
                import json as _json
                per_cls_out_t = {**per_cls_test, 'type': 'test', 'summary': summary_test}
                with open(os.path.join(log_dir, 'test_per_class_metrics.json'), 'w') as fpt:
                    _json.dump(per_cls_out_t, fpt, indent=2)
            except Exception:
                pass
        except Exception:
            pass
    try:
        hparams = {
            'lr': config.training.learning_rate,
            'batch_size': config.data.batch_size,
            'optimizer': config.training.optimizer,
            'loss_fn': config.training.loss_fn,
            'epochs': config.training.num_epochs,
        }
        metrics = {
            'hparam/best_val_acc': best_state.get('val_acc', 0),
            'hparam/best_val_auc': best_state.get('val_auc', 0),
            'hparam/best_val_f1': best_state.get('val_f1', 0),
            'hparam/best_val_recall': best_state.get('val_recall', 0),
        }
        writer.add_hparams(hparams, metrics)
    except Exception:
        pass

    # 统一输出一个 summary_all.json 汇总关键指标（best validation + test）
    try:
        summary_all = {
            'best_val': {
                'epoch': best_state.get('epoch', -1),
                'accuracy': best_state.get('val_acc', 0),
                'auc': best_state.get('val_auc', 0),
                'f1_weighted': best_state.get('val_f1', 0),
                'recall_macro': best_state.get('val_recall', 0)
            },
            'test': {
                'accuracy': test_acc,
                'auc': test_auc,
                # test_f1 已在上方局部变量 test_f1 计算，若未定义则重新计算
                'f1_weighted': (locals().get('test_f1') if 'test_f1' in locals() else compute_f1(test_targets, np.argmax(np.array(test_probs), axis=1), average='weighted')),
                'precision_weighted': locals().get('test_precision', None),
                'specificity_macro': locals().get('test_specificity', None)
            },
            'config': {
                'optimizer': config.training.optimizer,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.data.batch_size,
                'epochs': config.training.num_epochs,
                'loss_fn': config.training.loss_fn,
                'backbone': config.backbone.model_type,
                'num_classes': config.backbone.num_classes
            }
        }
        with open(os.path.join(log_dir, 'summary_all.json'), 'w') as sf:
            json.dump(summary_all, sf, indent=2)
    except Exception:
        pass
    # 记录参数与梯度直方图
    try:
        final_epoch = best_state.get('epoch', 0)
        for name, param in model.named_parameters():
            writer.add_histogram(f'Params/{name}', param.detach().cpu().numpy(), final_epoch)
            if param.grad is not None:
                writer.add_histogram(f'Grads/{name}', param.grad.detach().cpu().numpy(), final_epoch)
    except Exception:
        pass
    # 关闭前显式 flush，减少未写入事件风险
    try:
        writer.flush()
    except Exception:
        pass
    writer.close()

if __name__ == '__main__':
    main()