"""
使用合并后的 LoRA 模型进行测试
这个脚本专门用于测试 model_merged_clone.pth 的性能
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from config import Config
from data_loader import get_data_loaders
from model_builder import build_maddn_net
from utils import plot_confusion_matrix
from lora import clone_and_merge_lora  # 标准合并函数
import os

def test_merged_model(merged_model_path, config):
    """使用合并后的模型进行测试"""
    
    print("="*70)
    print("使用合并后的 LoRA 模型进行测试")
    print("="*70)
    print(f"模型路径: {merged_model_path}")
    print("说明: 此模型已将 LoRA 权重合并到基础权重中")
    print("="*70 + "\n")
    
    # 加载数据
    _, _, test_loader = get_data_loaders(config)
    
    # 构建模型架构（注意：不注入 LoRA，因为权重已合并）
    config.lora.enable_lora = False  # 禁用 LoRA 注入
    model = build_maddn_net(config)
    model = model.to(config.device)
    
    # 加载合并后的权重（统一标准流程）
    print("加载合并后的模型权重...")
    checkpoint = torch.load(merged_model_path, map_location=config.device, weights_only=False)
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    has_lora = any(('lora_' in k or '.base.' in k) for k in state.keys())
    if has_lora:
        print("[Info] 检测到未合并 LoRA 权重，重建带 LoRA 模型并执行标准合并。")
        # 构建带 LoRA 的模型
        lora_cfg = Config()
        lora_cfg.backbone = config.backbone
        lora_cfg.maddn = config.maddn
        lora_cfg.training = config.training
        lora_cfg.device = config.device
        lora_cfg.lora.enable_lora = True
        lora_model = build_maddn_net(lora_cfg).to(config.device)
        load_result = lora_model.load_state_dict(state, strict=False)
        print(f"[Load][LoRA] missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}")
        # 合并
        model = clone_and_merge_lora(lora_model).to(config.device)
        config.lora.enable_lora = False
        print("[Merge] LoRA 权重已合并到基座。")
    else:
        model.load_state_dict(state, strict=True)
        print("[OK] 加载纯净合并权重。")
    print("[OK] 模型准备完成\n")
    
    # 评估
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    # 调试统计
    debug_batches = 3
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing Merged Model')
        batch_index = 0
        for batch in pbar:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target, _, _ = batch
            # 数据基础统计
            if batch_index < debug_batches:
                print(f"[Debug] Batch {batch_index} label_counts={np.bincount(target.numpy(), minlength=2)}")
                print(f"[Debug] Data stats mean={data.float().mean().item():.4f} std={data.float().std().item():.4f} min={data.min().item():.4f} max={data.max().item():.4f}")
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            if batch_index < debug_batches:
                print(f"[Debug] Logits[0]={output[0].detach().cpu().numpy()} Probs[0]={probs[0].detach().cpu().numpy()}")
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            batch_index += 1
    
    # 计算指标
    cm = confusion_matrix(all_targets, all_preds)
    class_names = ['AD', 'CN']
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    
    # AUC
    from metrics_utils import compute_auc
    auc = compute_auc(all_targets, all_probs)
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    weighted_f1 = report['weighted avg']['f1-score']
    
    # 打印结果
    # 额外整体概率统计
    all_probs_arr = np.array(all_probs)
    print(f"[Dist] Mean probs per class: {all_probs_arr.mean(axis=0)}")
    print(f"[Dist] Fraction predicted class0={np.mean(np.array(all_preds)==0):.4f} class1={np.mean(np.array(all_preds)==1):.4f}")
    print("\n" + "="*70)
    print("合并模型测试结果")
    print("="*70)
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("\n混淆矩阵:")
    print(cm)
    print("="*70)
    
    # 保存结果
    save_dir = os.path.dirname(merged_model_path)
    plot_confusion_matrix(cm, class_names, 
                         save_path=os.path.join(save_dir, 'merged_model_confusion_matrix.png'))
    
    return test_acc, auc, weighted_f1

def main():
    config = Config()
    
    # 查找合并后的模型
    merged_model_path = os.path.join(config.training.checkpoint_dir, 'model_merged_clone.pth')
    
    if not os.path.exists(merged_model_path):
        print(f"[ERROR] 未找到合并后的模型: {merged_model_path}")
        print("\n请先运行训练脚本并确保 config.lora.export_merged = True")
        print("训练完成后会自动生成合并后的模型。")
        return
    
    test_merged_model(merged_model_path, config)

if __name__ == '__main__':
    main()
