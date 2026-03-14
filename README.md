# MADDN

基于 3D ResNet Backbone + 多尺度 Vision Transformer 的阿尔茨海默病（AD）MRI 影像自动分类深度学习框架。

支持完整训练、LoRA 参数高效微调、跨域 CORAL 对齐、断点恢复与多指标评估。

---

## 主要特性

| 特性 | 说明 |
|------|------|
| 多尺度 3D Transformer | 5 个尺度 ResNet 特征独立/共享 Transformer 融合 |
| LoRA 微调 | 极少参数适配新域，支持训练后权重合并导出 |
| 跨域 CORAL 对齐 | Deep CORAL 损失驱动的域自适应训练 |
| 断点恢复 | Checkpoint + TensorBoard global_step 连续，曲线无错位 |
| 多指标评估 | Acc / AUC / F1 / Recall / Precision / Specificity |
| 可配置损失函数 | CrossEntropy / Focal Loss + Label Smoothing |
| 渐进解冻策略 | Backbone 逐层解冻，稳定迁移学习 |

---

## 目录结构

```text
MADDN/
├── config.py                   # 全局配置（DataConfig / BackboneConfig / MADDNConfig / LoRAConfig / TrainingConfig）
├── data_loader.py              # 数据集构建与 DataLoader
├── augment.py                  # 3D MRI 数据增强
├── resnet.py                   # 3D ResNet-18 / ResNet-50 实现
├── maddn_net.py                # MADDN 核心网络（各尺度独立 Transformer）
├── maddn_net_simple.py         # MADDNShard（各尺度共享 Transformer，参数更少）
├── model_builder.py            # 统一模型构建入口（含 LoRA 注入）
├── lora.py                     # LoRALinear / LoRAConv3d / 注入与合并工具
├── metrics_utils.py            # 指标计算与绘图（AUC、F1、混淆矩阵等）
├── utils.py                    # AverageMeter / checkpoint / FocalLoss / accuracy
├── pretrain_resnet.py          # 3D ResNet 独立预训练脚本
├── train_maddn.py              # MADDN 主训练脚本
├── test_lora.py                # LoRA 注入单元测试
├── test_lora_merge.py          # LoRA 合并等价性测试
├── test_merged_model.py        # 合并后模型推理测试（含完整评估指标）
├── test_resnet.py              # ResNet backbone 单元测试
├── export_clean_components.py  # 导出干净权重组件（backbone / fusion / classifier）
├── environment.yml             # Conda 完整环境依赖
├── requirements.txt            # pip 核心依赖
└── checkpoints/                # 训练产物（自动创建）
    ├── checkpoint.pth.tar
    ├── model_best.pth.tar
    ├── model_merged_clone.pth
    └── clean_components/
        ├── backbone_finetuned_clean.pth
        ├── maddn_finetuned_clean.pth
        └── classifier_finetuned_clean.pth
```

---

## 环境准备

```bash
conda env create -f environment.yml
conda activate maddn
```

---

## 数据说明

在 `config.py` 的 `DataConfig.data_root` 中指定数据根目录。

目录结构：

```text
data_root/
├── AD/
│   ├── subject_001.nii.gz
│   └── ...
└── CN/
    ├── subject_001.nii.gz
    └── ...
```

- 支持 `.nii` / `.nii.gz` 格式，单通道灰度 3D MRI
- 输入尺寸默认 `128x128x128`（通过 `DataConfig.target_size` 修改）
- 自动按 7:1.5:1.5 划分 train / val / test

---

## 快速开始

### 1. 预训练 3D ResNet Backbone

```bash
python pretrain_resnet.py
```

### 2. 训练 MADDN

```bash
python train_maddn.py
```

### 3. 修改超参数

直接编辑 `config.py`，或在脚本中覆写：

```python
from config import Config
cfg = Config()
cfg.training.learning_rate = 5e-5
cfg.training.num_epochs = 50
cfg.maddn.use_shared_transformer = True   # True=MADDNShard; False=MADDN(full)
cfg.backbone.model_type = 'resnet18_3d'   # 或 resnet50_3d
cfg.backbone.pretrained_path = './checkpoints/resnet_best.pth'
```

---

## 断点恢复

```python
cfg.training.resume = './checkpoints/model_best.pth.tar'
```

恢复后自动：
- 加载模型 / 优化器 / 调度器状态
- 复用原 `log_dir`（TensorBoard 曲线无间断）
- `global_step` 从上次位置连续递增

---

## 模型变体

| 模型 | 类 | Transformer | 适用场景 |
|------|----|-------------|---------|
| MADDN | `MADDN` | 各尺度独立 | 数据充足、追求更强表达力 |
| MADDNShard | `MADDNShard` | 各尺度共享 | 数据较少、资源受限（默认） |

通过 `config.maddn.use_shared_transformer` 切换（默认 `True` = MADDNShard）。

**MADDNConfig 关键字段：**

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `use_shared_transformer` | `True` | 是否使用共享 Transformer |
| `embed_dim` | `256` | 嵌入维度 |
| `depth` | `2` | Transformer 块数量 |
| `num_heads` | `8` | 多头注意力头数 |
| `drop_rate` | `0.1` | Dropout |
| `pretrained_path` | `None` | MADDN 整体预训练权重路径 |

---

## LoRA 参数高效微调

### LoRAConfig 字段说明

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `enable_lora` | `False` | LoRA 总开关 |
| `rank` | `8` | 低秩维度 r（常用 4 / 8 / 16） |
| `alpha` | `16` | 缩放系数（有效缩放 = alpha / rank） |
| `dropout` | `0.0` | LoRA 分支 Dropout |
| `apply_to_backbone` | `False` | 是否对 ResNet3D 注入 LoRA |
| `apply_to_maddn` | `True` | 是否对 MADDN fusion_network 注入 LoRA |
| `target_modules` | `None` | 名称子串过滤（None = 全部目标层） |
| `freeze_base` | `False` | 冻结基础权重，只训练 LoRA + 分类头 |
| `merge_weights` | `False` | 训练后原地合并 Linear LoRA 到基础权重 |
| `export_merged` | `False` | 额外导出克隆合并版本（推荐推理使用） |

### 示例：仅微调 MADDN Transformer

```python
from config import Config
cfg = Config()
cfg.lora.enable_lora = True
cfg.lora.apply_to_maddn = True
cfg.lora.apply_to_backbone = False
cfg.lora.freeze_base = True
cfg.lora.export_merged = True
```

### 示例：同时微调 MADDN + Backbone

```python
cfg.lora.enable_lora = True
cfg.lora.apply_to_maddn = True
cfg.lora.apply_to_backbone = True
cfg.lora.freeze_base = False
```

### 调参建议

| 场景 | 推荐配置 |
|------|---------|
| 低资源 / 小数据 | `rank=4, alpha=8, dropout=0.1` |
| 更强表达力 | `rank=16, alpha=32` |
| 训练不稳定 | 降低 `learning_rate` 或提高 `rank` |

### 测试 LoRA

```bash
# 验证注入
python test_lora.py

# 验证合并等价性
python test_lora_merge.py

# 测试合并后推理
python test_merged_model.py ./checkpoints/model_merged_clone.pth
```

---

## 跨域迁移（CORAL 对齐）

```python
cfg.training.use_coral = True
cfg.training.coral_weight = 0.01
```

Deep CORAL 最小化源域与目标域特征协方差矩阵差异，促使模型学到域不变特征。

需在 `DataConfig.secondary_root` 指定目标域数据路径，并开启 `TrainingConfig.enable_domain = True`。

---

## 训练指标与 TensorBoard

```bash
tensorboard --logdir ./logs
```

| 记录项 | 说明 |
|--------|------|
| `Train/Loss_step`、`Train/Acc_step` | 每 batch |
| `Train/Loss`、`Train/Acc` | 每 epoch |
| `Train/GradNorm` | 梯度范数 |
| `Val/Loss`、`Val/Acc`、`Val/AUC`、`Val/F1` 等 | 验证集指标 |
| `Val/ConfusionMatrix` | 混淆矩阵（每 10 epoch） |
| `Test/*` | 最终测试指标 |

测试结果同步保存：`logs/<run>/test_results.json`

---

## Checkpoint 说明

| 文件 | 用途 |
|------|------|
| `checkpoint.pth.tar` | 最近一次（含优化器，用于恢复训练） |
| `model_best.pth.tar` | 验证集最佳（含优化器，用于恢复训练） |
| `model_merged_clone.pth` | LoRA 合并后推理模型（推荐部署） |
| `clean_components/backbone_finetuned_clean.pth` | 干净 Backbone 权重 |
| `clean_components/maddn_finetuned_clean.pth` | 干净 MADDN fusion 权重 |
| `clean_components/classifier_finetuned_clean.pth` | 干净分类头权重 |

### 推理示例

```python
import torch
from config import Config
from model_builder import build_maddn_net

config = Config()
config.lora.enable_lora = False

model = build_maddn_net(config)
ckpt = torch.load('./checkpoints/model_merged_clone.pth')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

x = torch.randn(1, 1, 128, 128, 128)
with torch.no_grad():
    out = model(x)  # shape: [1, 2]
```

### 导出干净权重

```bash
python export_clean_components.py --merged ./checkpoints/model_merged_clone.pth
python export_clean_components.py --best  ./checkpoints/model_best.pth.tar
```

---
