import torch
import torch.nn as nn  # 添加这行导入
import os
import shutil
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """计算指定k的准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """保存检查点"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_path)

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载检查点"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 使用 weights_only=False 来加载包含自定义类的检查点
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    
    # 加载模型状态
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # 获取epoch和最佳准确率
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0)
    
    return epoch, best_acc

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def evaluate_metrics(targets, preds, probs, class_names):
    """计算评估指标"""
    # 计算各种指标
    cm = confusion_matrix(targets, preds)
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    
    # 计算AUC（对于二分类）
    if len(class_names) == 2:
        auc = roc_auc_score(targets, [p[1] for p in probs])
    else:
        auc = roc_auc_score(targets, probs, multi_class='ovr')
    
    return cm, report, auc

def plot_confusion_matrix(cm, class_names, save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # 释放内存，替换plt.show()

def save_model_components(model, output_dir, include_classifier=True, include_lora=True):
    """
    分离保存模型的各个组件权重
    
    Args:
        model: 完整的模型
        output_dir: 保存目录
        include_classifier: 是否包含分类头
        include_lora: 是否包含 LoRA 权重
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取完整的 state_dict
    full_state = model.state_dict()
    
    # 1. 保存 backbone 权重
    backbone_state = {}
    for name, param in full_state.items():
        if name.startswith('backbone.') and 'lora_' not in name:
            # 移除 'backbone.' 前缀以便单独加载
            clean_name = name.replace('backbone.', '')
            backbone_state[clean_name] = param
    
    if backbone_state:
        backbone_path = os.path.join(output_dir, 'backbone_only.pth')
        torch.save({'state_dict': backbone_state}, backbone_path)
        print(f"[分离保存] Backbone 权重已保存: {backbone_path}")
    
    # 2. 保存 MADDN fusion_network 权重
    maddn_state = {}
    for name, param in full_state.items():
        if (name.startswith('fusion_network.') or name.startswith('shared_transformer.')) and 'lora_' not in name:
            maddn_state[name] = param
    
    if maddn_state:
        maddn_path = os.path.join(output_dir, 'maddn_only.pth')
        torch.save({'state_dict': maddn_state}, maddn_path)
        print(f"[分离保存] MADDN 权重已保存: {maddn_path}")
    
    # 3. 保存分类头权重（可选）
    if include_classifier:
        classifier_state = {}
        for name, param in full_state.items():
            if 'classifier' in name or name.startswith('fc.'):
                classifier_state[name] = param
        
        if classifier_state:
            classifier_path = os.path.join(output_dir, 'classifier_only.pth')
            torch.save({'state_dict': classifier_state}, classifier_path)
            print(f"[分离保存] 分类头权重已保存: {classifier_path}")
    
    # 4. 保存 LoRA 权重（可选）
    if include_lora:
        lora_state = {}
        for name, param in full_state.items():
            if 'lora_' in name or '.base.' in name:
                lora_state[name] = param
        
        if lora_state:
            lora_path = os.path.join(output_dir, 'lora_only.pth')
            torch.save({'state_dict': lora_state}, lora_path)
            print(f"[分离保存] LoRA 权重已保存: {lora_path} ({len(lora_state)} 个参数)")