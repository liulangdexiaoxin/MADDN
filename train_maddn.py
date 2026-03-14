"""MADDN training script

F1 logging policy:
    - Batch-level F1 (Train/F1_batch, Val/F1_batch): macro average (robust to temporary class absence in small batches).
    - Epoch/Test-level F1 (Train/F1, Val/F1, Test/F1): weighted average (reflects class imbalance in overall performance).
"""
import os
import matplotlib
# 统一强制使用 Agg，避免 Windows + 多进程下 Tk 异常
os.environ.setdefault('MPLBACKEND', 'Agg')
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import os
import time
import datetime
import json
from tqdm import tqdm
from sklearn.metrics import recall_score, confusion_matrix, classification_report, roc_curve, precision_score, precision_recall_curve, average_precision_score
from metrics_utils import (
    compute_auc, compute_precision, compute_specificity, compute_f1,
    compute_recall, compute_confusion_matrix, plot_confusion_matrix_figure,
    plot_roc_figure, compute_per_class_metrics, plot_per_class_bars,
    compute_macro_weighted_summary, plot_class_support_bar
)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

# 常量，避免重复字面量
ACC_LABEL = 'Accuracy (%)'
WEIGHTED_AVG_KEY = 'weighted avg'

from config import Config
from data_loader import get_data_loaders
from model_builder import build_maddn_net, count_parameters
from lora import get_lora_parameters, merge_lora_weights, clone_and_merge_lora
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, FocalLoss, plot_confusion_matrix
import math
from typing import Optional

# ================= 渐进解冻辅助函数 =================
def get_backbone_unfreeze_groups(model):
    """根据常见 ResNet3D 命名返回分组（从后往前解冻）。
    返回列表: [ [params_of_group0], [params_of_group1], ... ] group0 最靠后(高层)。
    """
    groups = []
    if hasattr(model, 'backbone'):
        bb = model.backbone
    else:
        bb = model
    # 根据常见层命名：layer4 -> layer3 -> layer2 -> layer1 -> stem
    order = []
    for name in ['layer4', 'layer3', 'layer2', 'layer1']:
        if hasattr(bb, name):
            order.append(getattr(bb, name))
    # stem 卷积 / bn / relu / maxpool
    stem_parts = []
    for stem_name in ['conv1', 'bn1']:
        if hasattr(bb, stem_name):
            stem_parts.append(getattr(bb, stem_name))
    if stem_parts:
        from torch.nn import ModuleList
        order.append(ModuleList(stem_parts))
    for module in order:
        params = [p for p in module.parameters()]
        if params:
            groups.append(params)
    return groups  # 按从后到前的顺序

def freeze_all_backbone(model):
    if hasattr(model, 'backbone'):
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        for n,p in model.named_parameters():
            if 'fc' not in n and 'classifier' not in n:
                p.requires_grad = False

def progressive_unfreeze_step(model, optimizer, config, epoch, logged_state):
    """在给定 epoch 检查是否需要解冻下一组。 logged_state 用于记录当前已解冻层数。"""
    tcfg = config.training
    milestones = getattr(tcfg, 'unfreeze_milestones', ())
    max_layers = getattr(tcfg, 'unfreeze_max_layers', 0)
    if epoch not in milestones:
        return
    if logged_state['groups'] is None:
        logged_state['groups'] = get_backbone_unfreeze_groups(model)
    # 当前已经解冻的组数量
    cur = logged_state['unfrozen']
    if cur >= min(max_layers, len(logged_state['groups'])):
        return
    # 解冻下一组 (按列表顺序)
    target_group_params = logged_state['groups'][cur]
    for p in target_group_params:
        p.requires_grad = True
    logged_state['unfrozen'] += 1
    group_idx = cur
    # 新增一个 param group 到优化器，使用分层学习率衰减
    base_lr = tcfg.learning_rate
    decay = getattr(tcfg, 'layerwise_lr_decay', 0.5)
    lr = base_lr * (decay ** (group_idx + 1))  # head 默认 group0，用 group_idx+1 衰减
    optimizer.add_param_group({'params': [p for p in target_group_params if p.requires_grad], 'lr': lr})
    print(f"[ProgressiveUnfreeze] Epoch {epoch}: 解冻第 {group_idx} 组 (params={len(target_group_params)}), lr={lr:.2e}")
    # 可选：首次解冻后重新初始化分类头（减少早期过拟合记忆）
    if group_idx == 0 and getattr(tcfg, 'reinit_classifier_after_unfreeze', False):
        for name, module in model.named_modules():
            if any(k in name for k in ['classifier', 'fc']):
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                    print(f"[ProgressiveUnfreeze] Re-init classifier module: {name}")

# ===== 可选域适配 / CORAL 损失实现 =====
def coral_loss(src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
    """CORAL 协方差对齐损失: https://arxiv.org/abs/1607.01719
    输入: src_feats [Ns, d], tgt_feats [Nt, d]
    返回: 标量 loss
    """
    if src_feats.numel() == 0 or tgt_feats.numel() == 0:
        return torch.zeros((), device=src_feats.device)
    d = src_feats.size(1)
    # 去均值
    src_centered = src_feats - src_feats.mean(dim=0, keepdim=True)
    tgt_centered = tgt_feats - tgt_feats.mean(dim=0, keepdim=True)
    cov_s = (src_centered.t() @ src_centered) / (src_feats.size(0) - 1 + 1e-8)
    cov_t = (tgt_centered.t() @ tgt_centered) / (tgt_feats.size(0) - 1 + 1e-8)
    loss = torch.mean((cov_s - cov_t) ** 2) / (4 * d * d)
    return loss

class EntropyAdaptiveLR:
    """根据预测分布熵动态调整学习率的包装器 (增强版)。
    关键修订: 避免在熵低时连续乘以 min_scale 导致学习率指数级坍缩。
    新增: base_lrs_override 支持——调用方可传入调度器更新后的 *未缩放* 基准学习率，用于一次性缩放而不复合。
    支持模式:
        linear / tanh / sigmoid / inverse / pid
    """
    def __init__(self, optimizer, num_classes, cfg):
        self.opt = optimizer
        self.num_classes = num_classes
        # 统计与目标参数
        self.window = cfg.entropy_window
        self.min_scale = cfg.entropy_min_lr_scale
        self.max_scale = cfg.entropy_max_lr_scale
        self.target = cfg.entropy_target
        self.smooth = cfg.entropy_smooth
        self.interval = cfg.entropy_adjust_interval
        self.clamp = cfg.entropy_clamp
        self.mode = getattr(cfg, 'entropy_mode', 'linear')
        self.factor = getattr(cfg, 'entropy_scale_factor', 1.0)
        self.warmup_steps = getattr(cfg, 'entropy_warmup_steps', 0)
        self.use_dynamic_base = getattr(cfg, 'entropy_use_scheduler_lr_as_base', True)
        # 抖动控制参数
        self.deadband = getattr(cfg, 'entropy_deadband', 0.0)
        self.scale_ema_alpha = getattr(cfg, 'entropy_scale_ema', 0.0)
        self.max_delta = getattr(cfg, 'entropy_max_delta', None)
        self.use_median = getattr(cfg, 'entropy_use_median', False)
        self.last_scale_applied = None
        # PID 参数
        self.kp = getattr(cfg, 'entropy_pid_kp', 0.8)
        self.ki = getattr(cfg, 'entropy_pid_ki', 0.05)
        self.kd = getattr(cfg, 'entropy_pid_kd', 0.2)
        self.integral = 0.0
        self.prev_error = None
        # 状态缓存
        self.buffer = []
        self.ema_entropy = None
        self.initial_lrs = [g['lr'] for g in self.opt.param_groups]
        self.steps_seen = 0
        # 诊断：连续单一类别
        self.single_class_counter = 0
        self.collapse_threshold = 30

    def _compute_scale(self, diff):
        # diff = avg_entropy - target (希望 diff ~ 0)
        if self.mode == 'linear':
            scale = 1.0 + self.factor * diff
        elif self.mode == 'tanh':
            scale = 1.0 + self.factor * math.tanh(diff)
        elif self.mode == 'sigmoid':
            scale = 1.0 + self.factor * (1/(1+math.exp(-diff)) - 0.5) * 2
        elif self.mode == 'inverse':
            scale = 1.0 / (1.0 + self.factor * diff)
        elif self.mode == 'pid':
            error = -diff  # 期望 avg_entropy 接近 target，error = target - avg => -diff
            self.integral += error
            derivative = 0.0 if self.prev_error is None else (error - self.prev_error)
            self.prev_error = error
            pid_out = self.kp * error + self.ki * self.integral + self.kd * derivative
            scale = 1.0 + self.factor * pid_out
        else:
            scale = 1.0 + self.factor * diff
        if self.clamp:
            scale = max(self.min_scale, min(self.max_scale, scale))
        return scale

    def step_entropy(self, probs_batch, base_lrs_override=None):
        with torch.no_grad():
            self.steps_seen += 1
            p = probs_batch.clamp(min=1e-8, max=1.0)
            ent = (-p * p.log()).sum(dim=1).mean().item()
            max_ent = math.log(self.num_classes + 1e-9)
            norm_ent = ent / max_ent
            if self.ema_entropy is None:
                self.ema_entropy = norm_ent
            else:
                self.ema_entropy = self.smooth * norm_ent + (1 - self.smooth) * self.ema_entropy
            self.buffer.append(self.ema_entropy)
            if len(self.buffer) > self.window:
                self.buffer.pop(0)
            if self.steps_seen < self.warmup_steps:
                return self.current_lr_state(update=False, avg_entropy=None, scale=None, reason='warmup')
            if self.steps_seen % self.interval != 0:
                return self.current_lr_state(update=False, avg_entropy=None, scale=None, reason='interval_skip')
            if self.use_median:
                import statistics
                avg_entropy = statistics.median(self.buffer)
            else:
                avg_entropy = sum(self.buffer) / len(self.buffer)
            diff = avg_entropy - self.target
            # deadband: diff 很小不更新，保持稳定
            if abs(diff) < self.deadband:
                return self.current_lr_state(update=False, avg_entropy=avg_entropy, scale=1.0, reason='deadband_skip')
            scale = self._compute_scale(diff)
            raw_scale = scale
            # scale EMA 平滑
            if self.scale_ema_alpha > 0:
                if self.last_scale_applied is None:
                    scale = self.last_scale_applied = scale
                else:
                    scale = self.scale_ema_alpha * scale + (1 - self.scale_ema_alpha) * self.last_scale_applied
            # 相邻变化裁剪
            if self.max_delta is not None and self.last_scale_applied is not None:
                delta = scale - self.last_scale_applied
                max_change = self.max_delta * max(1e-6, abs(self.last_scale_applied))
                if abs(delta) > max_change:
                    scale = self.last_scale_applied + math.copysign(max_change, delta)
            self.last_scale_applied = scale
            # 基准 lr (避免复合乘法): 优先使用调用方覆盖值 (scheduler 更新后原始 lr)
            if base_lrs_override is not None:
                base_current = base_lrs_override
            else:
                base_current = [g['lr'] for g in self.opt.param_groups] if self.use_dynamic_base else self.initial_lrs
            for i, g in enumerate(self.opt.param_groups):
                g['lr'] = max(1e-9, base_current[i] * scale)
            # 诊断：检测该批次是否全部预测为同一类别
            preds = torch.argmax(p, dim=1)
            if (preds == preds[0]).all():
                self.single_class_counter += 1
            else:
                self.single_class_counter = 0
            return self.current_lr_state(update=True, avg_entropy=avg_entropy, scale=scale, reason='applied', raw_scale=raw_scale)

    def current_lr_state(self, update=False, avg_entropy=None, scale=None, reason=None, raw_scale=None):
        return {
            'ema_entropy': self.ema_entropy,
            'buffer_len': len(self.buffer),
            'last_lrs': [g['lr'] for g in self.opt.param_groups],
            'updated': update,
            'avg_entropy': avg_entropy,
            'scale': scale,
            'raw_scale': raw_scale,
            'steps_seen': self.steps_seen,
            'reason': reason,
            'mode': self.mode
        }

def log_binary_roc_pr(writer, y_true, y_probs, tag_prefix, step, positive_index=1):
    """记录二分类 ROC 与 PR 曲线以及 AUC/AP 标量。
    参数:
        writer: SummaryWriter
        y_true: list/array 标签 (0/1)
        y_probs: list/array shape [N, C] 概率 (已 softmax)
        tag_prefix: 写入前缀，如 'Val/Best' 或 'Test'
        step: 日志步（epoch 或 自定义 step）
        positive_index: 正类概率列索引，默认 1
    """
    if writer is None:
        return
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
        arr_true = np.array(y_true)
        arr_probs = np.array(y_probs)
        
        # 数据验证
        if arr_probs.ndim != 2 or arr_probs.shape[1] <= positive_index:
            return
        if len(arr_true) == 0 or np.any(np.isnan(arr_true)) or np.any(np.isnan(arr_probs)):
            print("Warning: Invalid data for ROC/PR curves - contains NaN or empty")
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
    except Exception as e:
        print(f"Warning: Failed to log ROC/PR curves: {e}")

def log_epoch_metrics(writer, epoch: int, prefix: str, metrics: dict):
    """统一写入 epoch 级指标"""
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f'{prefix}/{k}', v, epoch)

def create_optimizer(model, config):
    """创建优化器"""
    # 分层学习率：仅当 freeze_epochs > 0 时启用，保持默认行为不变。
    # ====== LoRA 微调参数选择 ======
    if getattr(config, 'lora', None) and config.lora.enable_lora:
        # 收集 LoRA 参数
        lora_params = get_lora_parameters(model)
        lora_ids = {id(p) for p in lora_params}
        
        # 调试输出
        print(f"[Debug] LoRA 参数数量: {len(lora_params)}")
        print(f"[Debug] 模型总可训练参数: {sum(1 for p in model.parameters() if p.requires_grad)}")
        
        # 收集所有参数名称用于调试
        param_name_mapping = {id(p): n for n, p in model.named_parameters()}
        
        # 分类头参数：名称包含 fc/classifier 但不是 LoRA 分支的参数
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # 如果是 LoRA 分支参数，跳过（即使名称包含 fc）
            if id(p) in lora_ids:
                continue
            # 如果是分类头参数（非 LoRA）
            if any(h in n for h in ['classifier', 'fc']):
                head_params.append(p)
        
        head_ids = {id(p) for p in head_params}
        
        # 基础参数：可训练且不在 LoRA 或 head 中的参数
        base_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) not in lora_ids and id(p) not in head_ids:
                base_params.append(p)
        
        # 调试输出参数分组情况
        print(f"[Debug] Head 参数数量: {len(head_params)}")
        print(f"[Debug] LoRA 参数数量: {len(lora_params)}")
        print(f"[Debug] Base 参数数量: {len(base_params)}")
        
        # 构建参数组
        param_groups = []
        if head_params:
            param_groups.append({'params': head_params, 'lr': config.training.learning_rate})
            print(f"[Debug] 添加 head group: {len(head_params)} 参数")
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': config.training.learning_rate})
            print(f"[Debug] 添加 LoRA group: {len(lora_params)} 参数")
        if base_params and not config.lora.freeze_base:
            param_groups.append({'params': base_params, 'lr': config.training.learning_rate * 0.5})
            print(f"[Debug] 添加 base group: {len(base_params)} 参数")
        
        if not param_groups:
            param_groups = [{'params': model.parameters(), 'lr': config.training.learning_rate}]
            print("[Debug] 使用默认参数分组")
        
        # 验证参数唯一性
        all_param_ids = set()
        for gi, group in enumerate(param_groups):
            group_ids = {id(p) for p in group['params']}
            overlap = all_param_ids & group_ids
            if overlap:
                print(f"[Error] Group {gi} 与之前的组有重复参数:")
                for pid in overlap:
                    pname = param_name_mapping.get(pid, f"unknown_id_{pid}")
                    print(f"  - {pname}")
                raise ValueError("参数组之间存在重复，无法创建优化器")
            all_param_ids.update(group_ids)
    else:
        # 原有冻结策略
        if getattr(config.training, 'freeze_epochs', 0) > 0:
            head_params = []
            base_params = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in n for k in ['classifier', 'fc']):
                    head_params.append(p)
                else:
                    base_params.append(p)
            param_groups = [
                {'params': head_params, 'lr': config.training.learning_rate},
                {'params': base_params, 'lr': config.training.learning_rate * 0.1}
            ]
        else:
            param_groups = model.parameters()
    if config.training.optimizer == "adam":
        optimizer = optim.Adam(param_groups, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    elif config.training.optimizer == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    elif config.training.optimizer == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.training.learning_rate, momentum=config.training.momentum, weight_decay=config.training.weight_decay)
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
        criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    elif config.training.loss_fn == "focal":
        criterion = FocalLoss(
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )
    else:
        raise ValueError(f"Unsupported loss function: {config.training.loss_fn}")
    
    return criterion

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
    batch_losses = []

    # 初始化熵自适应调度器
    entropy_adaptor = None
    if getattr(config.training, 'entropy_adaptive', False):
        entropy_adaptor = EntropyAdaptiveLR(optimizer, config.backbone.num_classes, config.training)
        if writer is not None:
            writer.add_text('AdaptiveLR/Enabled', f'Entropy adaptive LR active (target={config.training.entropy_target}, mode={config.training.entropy_mode})', epoch)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} [Train]')
    # 兼容新数据加载返回四元组 (data,label,path,domain)
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            data, target, _ = batch
            domains = None
        else:
            data, target, _, domains = batch
        data, target = data.to(config.device), target.to(config.device)
        output = model(data)
        loss = criterion(output, target)
        batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
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
            grad_norm_accumulate.append(grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        probs = torch.softmax(output, dim=1)  # 需在调整前得到概率
        # ====== 可选 CORAL 域对齐（需模型支持 forward_features）======
        if getattr(config.training, 'use_coral', False) and domains is not None:
            # 仅在 batch 同时包含两个域时计算
            if isinstance(domains, (list, tuple)):
                # 无特征接口则跳过
                if hasattr(model, 'forward_features'):
                    feats = model.forward_features(data)
                    # 按域拆分
                    dom_arr = list(domains)
                    src_idx = [i for i,d in enumerate(dom_arr) if d == 'ADNI']
                    tgt_idx = [i for i,d in enumerate(dom_arr) if d != 'ADNI']
                    if src_idx and tgt_idx:
                        src_feats = feats[src_idx]
                        tgt_feats = feats[tgt_idx]
                        loss_coral = coral_loss(src_feats, tgt_feats) * getattr(config.training, 'coral_weight', 0.01)
                        loss = loss + loss_coral
                        if writer is not None:
                            writer.add_scalar('Train/Batch/CORAL_Loss', loss_coral.item(), batch_idx + epoch * len(train_loader))
        # 先进行 scheduler.step() 获得基准 lr，再按熵一次性缩放，避免多次复合
        if scheduler and config.training.lr_scheduler == "cosine":
            scheduler.step()
        if entropy_adaptor is not None:
            try:
                base_after_scheduler = [g['lr'] for g in optimizer.param_groups]
                state_lr = entropy_adaptor.step_entropy(probs.detach(), base_lrs_override=base_after_scheduler)
                if writer is not None:
                    if state_lr.get('updated'):
                        writer.add_scalar('AdaptiveLR/Scale', state_lr.get('scale', 1.0), state_lr.get('steps_seen', 0))
                        writer.add_scalar('AdaptiveLR/AvgEntropy', state_lr.get('avg_entropy', 0.0), state_lr.get('steps_seen', 0))
                        writer.add_scalar('AdaptiveLR/EMAEntropy', state_lr.get('ema_entropy', 0.0), state_lr.get('steps_seen', 0))
                        writer.add_scalar('AdaptiveLR/LR', optimizer.param_groups[0]['lr'], state_lr.get('steps_seen', 0))
                        if entropy_adaptor.single_class_counter >= entropy_adaptor.collapse_threshold:
                            writer.add_text('AdaptiveLR/CollapseWarning', f'Single-class predictions persisted {entropy_adaptor.single_class_counter} steps; consider adjusting entropy params.', state_lr.get('steps_seen', 0))
                    else:
                        writer.add_scalar('AdaptiveLR/EMAEntropy', state_lr.get('ema_entropy', 0.0), state_lr.get('steps_seen', 0))
            except Exception as e:
                if writer is not None:
                    writer.add_text('AdaptiveLR/Error', str(e), epoch)
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        preds = torch.argmax(probs, dim=1)
        batch_targets = target.cpu().numpy()
        batch_preds = preds.cpu().numpy()
        recall = compute_recall(batch_targets, batch_preds)
        recalls.update(recall, data.size(0))
        all_targets.extend(batch_targets)
        all_preds.extend(batch_preds)
        all_probs.extend(probs.detach().cpu().numpy())
        # TensorBoard记录每batch(仅训练集)，统一命名 Train/Batch/xxx
        if writer is not None:
            f1_batch = compute_f1(batch_targets, batch_preds, average='macro')
            base_offset = getattr(getattr(config, 'training', None), 'prev_global_step', 0)
            start_epoch_cfg = getattr(getattr(config, 'training', None), 'start_epoch', 0)
            rel_epoch = epoch - start_epoch_cfg
            global_step = base_offset + rel_epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Batch/Accuracy', acc1.item(), global_step)
            writer.add_scalar('Train/Batch/Recall', recall, global_step)
            writer.add_scalar('Train/Batch/F1_macro', f1_batch, global_step)
            if grad_norm is not None:
                writer.add_scalar('Train/Batch/GradNorm', grad_norm, global_step)
    # AUC 使用概率
    auc_val = compute_auc(all_targets, all_probs)
    # f1_epoch 计算在主循环中进行，这里不再需要局部变量，保持返回签名兼容
    if writer is not None and grad_norm_accumulate:
        writer.add_scalar('Train/GradNorm_epoch_mean', float(np.mean(grad_norm_accumulate)), epoch)
    # 训练阶段 Precision / Specificity（epoch级）
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
        for batch_idx, batch in enumerate(pbar):
            # DataLoader 返回：
            #   主数据集 MRIDataset -> (tensor, label, path, domain_name)
            #   历史/旧版本或某些调用可能只返回前三项 (tensor, label, path)
            # 因此这里根据长度兼容处理，忽略 path 与 domain，仅使用 data/label。
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target, _, _ = batch
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            batch_targets = target.cpu().numpy()
            batch_preds = preds.cpu().numpy()
            recall = compute_recall(batch_targets, batch_preds)
            recalls.update(recall, data.size(0))
            all_targets.extend(batch_targets)
            all_preds.extend(batch_preds)
            all_probs.extend(probs.detach().cpu().numpy())
            # F1-score (macro) 计算已在批次外使用 weighted 版本，不保存临时变量
            # 按需求删除验证集 batch 级日志
    auc_val = compute_auc(all_targets, all_probs)
    # epoch 级 F1 在主训练循环进行（weighted）；此处无需再计算
    # 移除内部重复写入，统一在外层 log_epoch_metrics 中记录
    # 验证阶段 Precision / Specificity
    val_precision = compute_precision(all_targets, all_preds)
    val_specificity = compute_specificity(all_targets, all_preds)
    if writer is not None and epoch is not None:
        writer.add_scalar('Val/Precision', val_precision, epoch)
        writer.add_scalar('Val/Specificity', val_specificity, epoch)
    return losses.avg, top1.avg, recalls.avg, auc_val, all_targets, all_preds, all_probs

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
    plt.ylabel(ACC_LABEL)
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
    plt.subplot(2, 2, 1)
    plt.plot(steps_batch, train_batch_losses, label='Train/Batch Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train/Batch Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(steps_epoch, train_epoch_losses, label='Train/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Epoch Loss')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(steps_epoch, val_epoch_losses, label='Val/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Val/Epoch Loss')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(steps_epoch, val_epoch_accs, label='Val/Epoch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel(ACC_LABEL)
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
    plt.ylabel(ACC_LABEL)
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

def test_model(model, test_loader, config, class_names=['AD', 'CN'], save_dir=None):
    """测试模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            # 与 validate 同样的批次结构兼容策略，忽略路径与域信息。
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target, _, _ = batch
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    # 统一使用 compute_auc
    auc = compute_auc(all_targets, all_probs)
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    out_dir = save_dir or config.training.log_dir
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(out_dir, 'confusion_matrix.png'))
    weighted_f1 = report[WEIGHTED_AVG_KEY]['f1-score']
    print(f"\nWeighted F1-score: {weighted_f1:.4f}")
    print("\n" + "="*50)
    print("测试集性能评估")
    print("="*50)
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("\n混淆矩阵:")
    print(cm)
    results = {
        'test_accuracy': test_acc,
        'auc': auc,
        'weighted_f1': weighted_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    # Precision / Specificity / ROC
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
    # 写入 TensorBoard（若存在全局 writer，在 main 中执行） -> 返回扩展指标
    return test_acc, auc, cm, report, test_precision, test_specificity, all_targets, all_probs

def main():
    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    # 可按需在此处覆写 config.backbone 相关参数
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 创建模型
    model = build_maddn_net(config)
    model = model.to(config.device)
    # ====== 冻结 / 渐进解冻初始化 ======
    prog_enabled = getattr(config.training, 'progressive_unfreeze', False) and not (
        getattr(getattr(config, 'lora', None), 'enable_lora', False) and getattr(config.lora, 'freeze_base', False)
    )
    progressive_state = {'groups': None, 'unfrozen': 0}
    if prog_enabled:
        # 初始全部冻结 backbone + 仅分类头（与 freeze_epochs 类似但使用 milestones 控制解冻）
        freeze_all_backbone(model)
        print(f"[ProgressiveUnfreeze] 启用，里程碑: {getattr(config.training, 'unfreeze_milestones', [])}")
    else:
        if getattr(config.training, 'freeze_epochs', 0) > 0:
            if hasattr(model, 'backbone'):  # MADDN 结构
                for n,p in model.backbone.named_parameters():
                    p.requires_grad = False
            else:  # 直接 ResNet
                for n,p in model.named_parameters():
                    if 'fc' not in n and 'classifier' not in n:
                        p.requires_grad = False
            print(f"[Freeze] 前 {config.training.freeze_epochs} 个 epoch 冻结除分类头外参数。")
    
    # 打印模型参数数量
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
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
        prev_log_dir = ckpt_raw.get('log_dir')
        prev_global_step = ckpt_raw.get('global_step', 0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%, prev_global_step={prev_global_step}")
        config.training.start_epoch = start_epoch
        config.training.prev_global_step = prev_global_step
    else:
        config.training.start_epoch = 0
        config.training.prev_global_step = 0

    # TensorBoard 日志目录：优先复用
    if prev_log_dir and os.path.isdir(prev_log_dir):
        log_dir = prev_log_dir
        print(f"[Resume] Reusing existing log_dir: {log_dir}")
    else:
        run_name = f"maddn_bs{config.data.batch_size}_lr{config.training.learning_rate}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_dir = os.path.join(config.training.log_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f"[New Run] Created log_dir: {log_dir}")

    # 可选：purge_step 防止重复 step 覆盖（若之前 event 未完整写完可打开）
    # 使用 purge_step 以防止在断点恢复时出现重复 step 导致曲线重叠（仅当存在先前 global_step 时）
    if 'prev_global_step' in locals() and prev_global_step > 0:
        writer = SummaryWriter(log_dir=log_dir, purge_step=prev_global_step)
    else:
        writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir={log_dir}")

    try:
        writer.add_text('Config', json.dumps(config.__dict__, indent=2))
    except Exception as e:
        print(f"Warning: Failed to log config to TensorBoard: {e}")
    
    # 训练循环
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_recalls, val_recalls = [], []
    train_batch_losses = []
    train_f1s, val_f1s = [], []
    best_state = {'epoch': -1, 'val_acc': -1, 'val_auc': 0, 'val_f1': 0, 'val_recall': 0}

    print(f"Batch size: {config.data.batch_size}")
    print(f"Total epochs: {config.training.num_epochs}")

    for epoch in range(start_epoch, config.training.num_epochs):
        # 渐进解冻检查
        if prog_enabled:
            progressive_unfreeze_step(model, optimizer, config, epoch, progressive_state)
        # 到达解冻点：只执行一次
        if (not prog_enabled) and getattr(config.training, 'freeze_epochs', 0) > 0 and epoch == config.training.freeze_epochs:
            if hasattr(model, 'backbone'):
                for n,p in model.backbone.named_parameters():
                    p.requires_grad = True  # 全部解冻或可按需挑层
            else:
                for n,p in model.named_parameters():
                    p.requires_grad = True
            print(f"[Unfreeze] Epoch {epoch} 解冻 backbone 参数。")
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_auc, batch_losses, train_targets, train_preds, _ = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, config, writer
        )
        # 验证
        val_loss, val_acc, val_recall, val_auc, val_targets, val_preds, val_probs = validate(
            model, val_loader, criterion, config, epoch, writer
        )

        from sklearn.metrics import f1_score
        train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_batch_losses.extend(batch_losses)

        # 每 epoch 写入统一命名
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
        except Exception as e:
            print(f"Warning: Failed to log histograms: {e}")

        # batch 级写入已在 train_epoch 内完成，无需重复
        # 更新学习率（对于plateau调度器）
        if scheduler and config.training.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        # 保存最佳模型
        is_best = val_acc > best_state['val_acc']
        if is_best:
            best_state.update({'epoch': epoch, 'val_acc': val_acc, 'val_auc': val_auc, 'val_f1': val_f1, 'val_recall': val_recall})
            best_acc = val_acc
            # 写入最佳混淆矩阵与 ROC (+ 可选 per-class 条形图/JSON)
            try:
                cm_best = confusion_matrix(val_targets, val_preds)
                fig_cm, ax = plt.subplots(figsize=(4,4))
                sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
                ax.set_title('Best Val Confusion Matrix')
                ax.set_xlabel('Pred')
                ax.set_ylabel('True')
                writer.add_figure('Val/ConfusionMatrix_best', fig_cm, epoch)
                plt.close(fig_cm)
                if getattr(getattr(config, 'logging', None), 'export_per_class', True):
                    # per-class metrics bar
                    try:
                        per_cls = compute_per_class_metrics(val_targets, val_preds)
                        fig_bar = plot_per_class_bars(per_cls, title='Best Val Per-class Metrics')
                        if fig_bar:
                            writer.add_figure('Val/PerClassMetrics_best', fig_bar, epoch)
                            plt.close(fig_bar)
                        fig_sup = plot_class_support_bar(per_cls, title='Best Val Class Support')
                        if fig_sup:
                            writer.add_figure('Val/ClassSupport_best', fig_sup, epoch)
                            plt.close(fig_sup)
                        summary = compute_macro_weighted_summary(val_targets, val_preds)
                        # JSON 保存
                        try:
                            import json as _json
                            per_cls_out = {**per_cls, 'epoch': epoch, 'type': 'best_val', 'summary': summary}
                            with open(os.path.join(log_dir, 'best_val_per_class_metrics.json'), 'w') as fpc:
                                _json.dump(per_cls_out, fpc, indent=2)
                        except Exception as e:
                            print(f"Warning: Failed to save best val per-class metrics JSON: {e}")
                    except Exception as e:
                        print(f"Warning: Failed to compute per-class metrics for best validation: {e}")
            except Exception as e:
                print(f"Warning: Failed to log best validation confusion matrix: {e}")
            # 记录最佳验证 ROC/PR
            if len(set(val_targets)) == 2:
                pos_idx = getattr(getattr(config, 'training', None), 'positive_index', 1)
                log_binary_roc_pr(writer, val_targets, val_probs, 'Val/Best', epoch, pos_idx)
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
            
            # 分离保存组件权重
            if config.training.save_separate_components:
                should_save_separate = False
                if config.training.separate_components_on_best and is_best:
                    should_save_separate = True
                elif config.training.separate_components_freq > 0 and (epoch + 1) % config.training.separate_components_freq == 0:
                    should_save_separate = True
                
                if should_save_separate:
                    try:
                        from utils import save_model_components
                        os.makedirs(config.training.separate_components_dir, exist_ok=True)
                        save_model_components(
                            model,
                            config.training.separate_components_dir,
                            include_classifier=config.training.separate_include_classifier,
                            include_lora=config.training.separate_include_lora
                        )
                        print(f"[分离保存] Epoch {epoch+1}: 已保存分离组件权重到 {config.training.separate_components_dir}")
                    except Exception as e:
                        print(f"[分离保存] 保存失败: {e}")
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.4f}')
    # （以上指标已经在前面 append，不再重复，移除重复代码）

    # 绘制详细学习曲线
    plot_full_learning_curves(
        train_batch_losses, train_losses, val_losses, val_accs, log_dir
    )
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, log_dir)
    # 绘制指标曲线
    plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, log_dir)
    
    print("\n训练完成，开始在测试集上评估最佳模型...")
    print("="*70)
    
    best_model_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        # 使用 ASCII 提示，避免 GBK 控制台 UnicodeEncodeError
        print(f"[OK] 加载最佳验证模型: {best_model_path}")
        print("  注意：此模型包含 LoRA 权重（未合并状态）")
        print("  包含的参数：Backbone + MADDN + LoRA 分支 + 分类头")
        load_checkpoint(best_model_path, model)
    else:
        print("[WARN] 未找到最佳模型，使用最终训练状态进行评估")
    
    print("="*70 + "\n")
    test_acc, test_auc, cm_test, _, test_precision, test_specificity, test_targets, test_probs = test_model(
        model, test_loader, config, save_dir=log_dir
    )
    # 统一测试阶段独立 step
    test_step = config.training.num_epochs
    try:
        # ROC 曲线（仅二分类）
        if len(set(test_targets)) == 2:
            pos_idx = getattr(getattr(config, 'training', None), 'positive_index', 1)
            log_binary_roc_pr(writer, test_targets, test_probs, 'Test', test_step, pos_idx)
        # 混淆矩阵
        fig_cm, axc = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axc)
        axc.set_title('Test Confusion Matrix')
        axc.set_xlabel('Pred'); axc.set_ylabel('True')
        writer.add_figure('Test/ConfusionMatrix', fig_cm, test_step)
        plt.close(fig_cm)
        # Scalar 指标
        from sklearn.metrics import f1_score as _f1
        test_f1 = _f1(test_targets, np.argmax(np.array(test_probs), axis=1), average='weighted', zero_division=0)
        writer.add_scalar('Test/Accuracy', test_acc, test_step)
        writer.add_scalar('Test/AUC', test_auc, test_step)
        writer.add_scalar('Test/F1_weighted', test_f1, test_step)
        writer.add_scalar('Test/Precision', test_precision, test_step)
        writer.add_scalar('Test/Specificity', test_specificity, test_step)
        # PR 曲线
        # （二分类 PR 已在 log_binary_roc_pr 中记录）
        # per-class 指标与 JSON
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
                except Exception as e:
                    print(f"Warning: Failed to save test per-class metrics JSON: {e}")
            except Exception as e:
                print(f"Warning: Failed to compute test per-class metrics: {e}")
    except Exception as e:
        print(f"Warning: Failed to log test metrics to TensorBoard: {e}")
    print("\n最终测试结果:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {test_auc:.4f}")
    end_time = datetime.datetime.now()
    print(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"训练总耗时: {str(duration)}")
    # 写入 hparams
    try:
        hparams = {
            'lr': config.training.learning_rate,
            'batch_size': config.data.batch_size,
            'optimizer': config.training.optimizer,
            'loss_fn': config.training.loss_fn,
            'epochs': config.training.num_epochs,
        }
        metrics = {
            'hparam/best_val_acc': best_state['val_acc'],
            'hparam/best_val_auc': best_state['val_auc'],
            'hparam/best_val_f1': best_state['val_f1'],
            'hparam/best_val_recall': best_state['val_recall'],
        }
        writer.add_hparams(hparams, metrics)
    except Exception as e:
        print(f"Warning: Failed to log hyperparameters: {e}")
    # 统一输出 summary_all.json（best validation + test 聚合）
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
    except Exception as e:
        print(f"Warning: Failed to save summary_all.json: {e}")
    # 确保缓冲区事件全部写入磁盘
    try:
        writer.flush()
    except Exception:
        pass
    # ===== 训练结束可选合并 LoRA 权重 =====
    # ===== LoRA 权重导出策略 =====
    if getattr(config, 'lora', None) and config.lora.enable_lora:
        print("\n" + "="*50)
        print("LoRA 权重合并与导出")
        print("="*50)
        
        # 保存到 checkpoint 目录，更容易找到
        checkpoint_dir = config.training.checkpoint_dir
        
        if config.lora.merge_weights:
            try:
                # 直接在原模型合并（只适合之后不再继续微调的情况）
                merge_lora_weights(model)
                merged_path = os.path.join(checkpoint_dir, 'model_merged_inplace.pth')
                torch.save({'state_dict': model.state_dict()}, merged_path)
                print(f'[LoRA] 已合并 Linear 层并保存到: {merged_path}')
                print('[LoRA] 注意：原模型已被修改，LoRA 参数已合并到基础权重中')
            except Exception as e:
                print(f'[LoRA] 合并失败: {e}')
        
        if config.lora.export_merged:
            try:
                # 克隆并合并（推荐方式，不影响原模型）
                merged_clone = clone_and_merge_lora(model)
                merged_clone_path = os.path.join(checkpoint_dir, 'model_merged_clone.pth')
                torch.save({'state_dict': merged_clone.state_dict()}, merged_clone_path)
                print(f'[LoRA] 已生成克隆并合并版本: {merged_clone_path}')
                print('[LoRA] 原模型保持未合并状态，可继续微调')
                
                # 同时保存到 log_dir 方便查看
                log_merged_path = os.path.join(log_dir, 'model_merged_clone.pth')
                torch.save({'state_dict': merged_clone.state_dict()}, log_merged_path)
                print(f'[LoRA] 同时保存副本到: {log_merged_path}')

                # ===== 新增：自动导出双组件干净权重 =====
                try:
                    clean_out_dir = os.path.join(checkpoint_dir, 'clean_components')
                    os.makedirs(clean_out_dir, exist_ok=True)
                    backbone_state = {}
                    fusion_state = {}
                    for k, v in merged_clone.state_dict().items():
                        if 'lora_' in k or '.base.' in k:
                            continue
                        if k.startswith('backbone.'):
                            backbone_state[k.replace('backbone.', '')] = v
                        elif k.startswith('fusion_network.') or k.startswith('shared_transformer.'):
                            fusion_state[k] = v
                        elif k.startswith('classifier.') or k.startswith('fc.'):
                            pass  # 先收集在下面
                    classifier_state = {}
                    for k, v in merged_clone.state_dict().items():
                        if ('classifier.' in k or k.startswith('fc.')) and 'lora_' not in k and '.base.' not in k:
                            classifier_state[k] = v
                    bb_path = os.path.join(clean_out_dir, 'backbone_finetuned_clean.pth')
                    mo_path = os.path.join(clean_out_dir, 'maddn_finetuned_clean.pth')
                    if backbone_state:
                        torch.save({'state_dict': backbone_state}, bb_path)
                        print(f'[Export] 已导出微调后 Backbone 清洁权重: {bb_path} (layers={len(backbone_state)})')
                    else:
                        print('[Export][Warn] 未找到 backbone.* 参数，跳过导出。')
                    if fusion_state:
                        torch.save({'state_dict': fusion_state}, mo_path)
                        print(f'[Export] 已导出微调后 MADDN 清洁权重: {mo_path} (layers={len(fusion_state)})')
                    else:
                        print('[Export][Warn] 未找到 MADDN 相关参数，跳过导出。')
                    # 分类头
                    if classifier_state:
                        cls_path = os.path.join(clean_out_dir, 'classifier_finetuned_clean.pth')
                        torch.save({'state_dict': classifier_state}, cls_path)
                        print(f'[Export] 已导出微调后分类头权重: {cls_path} (layers={len(classifier_state)})')
                    else:
                        print('[Export][Warn] 未找到分类头参数（classifier./fc.），可能未使用独立分类层。')
                except Exception as ce:
                    print(f'[Export][Error] 导出 clean 组件失败: {ce}')
                
            except Exception as e:
                print(f'[LoRA] 克隆合并失败: {e}')
        
        print("\n推荐使用合并后的模型进行部署和推理！")
        print("="*50 + "\n")
    writer.close()

if __name__ == '__main__':
    main()