import os
import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataConfig:
    data_root: str = "/root/01_dataset/ADNI_dataset_compressed"
    # data_root: str = "/root/01_dataset/AD_compressed/ADNI_dataset"
    batch_size: int = 32    # 根据独立显存大小设置，太大会导致每个epoch时间暴增，使用Tesla P40 24GB显存设置24，使用A100 80GB显存设置32
    num_workers: int = 16    # 根据CPU核心数和IO能力设置（CPU 6核，12GB内存）
    target_size: int = 128
    # train_split: float = 0.7
    # val_split: float = 0.15
    # test_split: float = 0.15
    # 可选第二数据域（例如 NACC）根目录，仅在启用多域训练时使用
    secondary_root: Optional[str] = None

@dataclass
class BackboneConfig:
    model_type: str = "resnet18_3d"  # "resnet18_3d" or "resnet50_3d"
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    in_channels: int = 1
    num_classes: int = 2
    # 可选适配层，在迁移学习微调时启用，默认关闭保证原始行为不变
    use_adapter: bool = False

@dataclass
class MADDNConfig:
    """
    MADDNConfig: MADDN 模型的配置参数
    包含模型结构、注意力机制、dropout率等关键超参数
    """
    use_shared_transformer: bool = True  # True=MADDNShard（共享权重，参数少）; False=MADDN（独立权重，表达力强）
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    # ===== 预训练加载 =====
    pretrained: bool = False               # 是否加载 MADDN 层预训练权重（区别于 backbone.pretrained）
    pretrained_path: Optional[str] = None  # MADDN 预训练 checkpoint 路径（可包含整网 state_dict）
    load_strict: bool = False              # 加载时是否 strict；默认 False 以便跳过分类头维度不匹配
    ignore_classifier: bool = True         # 加载 MADDN 权重时是否忽略分类头 (fc/classifier)

@dataclass
class LoRAConfig:
    """LoRA 低秩适配配置
    enable_lora: 总开关
    rank: 低秩分解维度 r
    alpha: 缩放因子 (实际缩放 = alpha / rank)
    dropout: LoRA 分支 dropout
    apply_to_backbone: 是否在 ResNet3D 上注入（Conv3d / fc）
    apply_to_maddn: 是否在 MADDN Transformer 相关 Linear 层注入 (qkv / proj / mlp fc)
    target_modules: 过滤模块名称子串匹配（为空则按默认规则）
    freeze_base: 是否冻结原始权重只训练 LoRA 分支 + 新分类头
    merge_weights: 训练结束是否将 LoRA 权重合并回基座（实验后固化）
    """
    enable_lora: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    apply_to_backbone: bool = False
    apply_to_maddn: bool = True       # 是否在 MADDN Transformer 相关 Linear 层注入 LoRA (qkv / proj / mlp fc)
    target_modules: Optional[list] = None
    freeze_base: bool = False
    merge_weights: bool = False
    export_merged: bool = False  # 训练结束时额外导出合并后推理权重（同时保留未合并）

@dataclass
class TrainingConfig:

    """
    训练配置类，包含模型训练所需的各种参数设置。
    包括优化器、学习率调度、训练参数、损失函数以及检查点与日志相关配置。
    """
    # 优化器相关参数
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # 用于SGD
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    step_size: int = 10  # 用于step scheduler，每多少epoch调整一次学习率
    gamma: float = 0.1  # 用于step scheduler
    min_lr: float = 1e-7  # 用于cosine/plateau scheduler
    patience: int = 5  # 用于plateau scheduler
    
    # 训练参数
    num_epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # 损失函数
    loss_fn: str = "focal"  # "cross_entropy", "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # 检查点与日志
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 10  # 每多少epoch保存一次
    resume: Optional[str] = None  # 恢复训练的检查点路径
    # 分离保存选项（新增）
    save_separate_components: bool = False  # 是否保存独立组件权重
    separate_components_on_best: bool = False  # 仅在最佳模型时保存分离组件
    separate_components_freq: int = 0  # 每多少 epoch 保存一次分离组件（0=仅最佳时）
    separate_components_dir: str = "./separate_weights"  # 分离组件保存目录
    separate_include_classifier: bool = False  # 分离保存中是否包含分类头
    separate_include_lora: bool = False  # 分离保存中是否包含 LoRA 权重
    # 熵自适应学习率相关参数
    entropy_adaptive: bool = False          # 是否启用熵驱动学习率调整
    entropy_window: int = 120               # 计算滑动窗口平均熵的步数
    entropy_min_lr_scale: float = 0.3       # 当熵极低(模型非常确定)时的最小 lr 缩放比例
    entropy_max_lr_scale: float = 1.5       # 当熵较高(模型不确定)时的最大 lr 缩放比例
    entropy_target: float = 0.55             # 目标平均熵 (基于 log(num_classes) 归一化后 0~1)
    entropy_smooth: float = 0.1             # EMA 平滑系数 (0-1)
    entropy_adjust_interval: int = 20       # 每多少个 batch 进行一次学习率调整
    entropy_clamp: bool = True              # 是否对动态缩放进行区间裁剪
    # 新增高级控制参数
    entropy_mode: str = "tanh"            # 调整模式: linear|tanh|sigmoid|inverse|pid
    entropy_scale_factor: float = 0.5       # 对 diff 的整体缩放放大/缩小调整幅度
    entropy_warmup_steps: int = 0           # 前若干 step 不做自适应，仅收集统计
    entropy_use_scheduler_lr_as_base: bool = True  # 是否每次以调度器当前 lr 为基准再缩放
    # --- 抖动控制新增参数 ---
    entropy_deadband: float = 0.03          # diff 绝对值低于该阈值不调整（减少无意义抖动）
    entropy_scale_ema: float = 0.25          # 对 scale 做 EMA 平滑 (0=关闭)
    entropy_max_delta: float = 0.15         # 相邻两次 scale 变动的最大允许比例差 (裁剪)
    entropy_use_median: bool = True        # 使用窗口中位数代替平均 (减少极值影响)
    # PID 模式参数
    entropy_pid_kp: float = 0.8
    entropy_pid_ki: float = 0.05
    entropy_pid_kd: float = 0.2
    # ====== 迁移学习 / 域适配可选参数（默认全部关闭，不影响原训练） ======
    enable_domain: bool = False            # 是否在数据加载时返回域标签并尝试多域训练
    use_coral: bool = False                # 是否启用 CORAL 协方差对齐损失
    coral_weight: float = 0.01             # CORAL 损失权重
    freeze_epochs: int = 0                 # 前 freeze_epochs 仅训练分类头（0 表示不冻结）
    positive_index: int = 1                # 二分类中正类索引用于 ROC/PR 记录
    # 渐进解冻 / 分层学习率（新）—— 为保持向后兼容默认关闭
    progressive_unfreeze: bool = False     # 是否启用按里程碑(epoch)渐进解冻 (启用后优先于 freeze_epochs)
    unfreeze_milestones: tuple = (5, 15, 30)  # 到达这些 epoch 时依次解冻更多层（示例值）
    unfreeze_max_layers: int = 3           # 最多解冻的“组”数量（防止过度解冻）
    layerwise_lr_decay: float = 0.25       # 每向前一组层学习率乘以该衰减系数 (head lr * decay^k)
    reinit_classifier_after_unfreeze: bool = False  # 解冻第一组后是否重新初始化分类头（某些场景提升稳定性）

@dataclass
class LoggingConfig:
    """日志相关扩展配置"""
    export_per_class: bool = True  # 是否导出 per-class 指标与图像
    enable_param_hist: bool = True  # 是否记录参数直方图
    enable_grad_hist: bool = True   # 是否记录梯度直方图
    hist_interval: int = 10         # 每多少个 epoch 记录一次直方图（在 epoch 结束时）

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    maddn: MADDNConfig = field(default_factory=MADDNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        # 创建必要的目录
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)