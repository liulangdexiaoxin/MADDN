import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple

class LoRALinear(nn.Module):
    """对 nn.Linear 进行 LoRA 低秩适配包装。
    公式: W' = W + (alpha/r) * B @ A
    A: down (in_features -> r)
    B: up   (r -> out_features)
    """
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        assert rank > 0, "LoRA rank must be > 0"
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # LoRA 权重
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        # 初始化: A 正态，B 零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        lora_out = self.dropout(x) @ self.lora_A.t()  # [B, rank]
        lora_out = lora_out @ self.lora_B.t()         # [B, out]
        return out + lora_out * self.scaling

    def merge_weights(self):
        """将 LoRA 权重合并回基础层并标记 merged = True。"""
        if self.merged:
            return
        # base.weight: [out_features, in_features]
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        with torch.no_grad():
            self.base.weight += delta
        self.merged = True

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}"

class LoRAConv3d(nn.Module):
    """对 nn.Conv3d 进行 LoRA 包装: 使用相同 kernel size 的卷积进行低秩调整。
    改进实现：
        原始 W: [out_c, in_c, k, k, k]
        LoRA: A: kxkxk 降维 (in_c -> rank), B: 1x1x1 升维 (rank -> out_c)
        确保维度匹配和正确的空间处理
    """
    def __init__(self, base: nn.Conv3d, rank: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        in_c = base.in_channels
        out_c = base.out_channels
        kernel_size = base.kernel_size
        stride = base.stride
        padding = base.padding
        
        # LoRA A: 使用与原始卷积相同的参数，输出到 rank 维度
        self.lora_A = nn.Conv3d(
            in_c, rank, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        # LoRA B: 使用 1x1x1 升维到输出通道
        self.lora_B = nn.Conv3d(rank, out_c, kernel_size=1, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        
        # LoRA 前向传播：先通过 A 降维，再通过 B 升维
        lora_out = self.lora_A(self.dropout(x))  # [B, rank, H', W', D']
        lora_out = self.lora_B(lora_out)         # [B, out_c, H', W', D']
        
        return out + lora_out * self.scaling

    def merge_weights(self):
        if self.merged:
            return
        # 对于复杂的 Conv3d，暂时保持动态分支不合并
        # 未来可以实现权重合并逻辑
        pass

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, merged={self.merged}, base_kernel={self.base.kernel_size}"


def is_lora_target(name: str, module: nn.Module, target_substrings: Optional[List[str]]) -> bool:
    # 避免对已经 LoRA 包装内部的 base 进行重复注入（会导致递归膨胀）
    if name.endswith('.base'):
        return False
    if isinstance(module, (LoRALinear, LoRAConv3d)):
        return False
    if target_substrings:
        return any(ts in name for ts in target_substrings)
    # 默认规则: Linear 层 或 Conv3d 层（现在支持任意 kernel size）
    if isinstance(module, nn.Linear):
        return True
    if isinstance(module, nn.Conv3d):
        return True
    return False


def inject_lora(model: nn.Module, rank: int, alpha: int, dropout: float,
                target_substrings: Optional[List[str]] = None,
                include_conv3d: bool = True,
                verbose: bool = False) -> Tuple[int,int]:
    """遍历模型并替换目标层为 LoRA 包装。
    返回 (替换的 Linear 数量, 替换的 Conv3d 数量)
    """
    linear_count = 0
    conv_count = 0
    for name, module in model.named_modules():
        # 跳过已经是 LoRA 包装的模块，防止其内部 base 被再次遍历替换
        if isinstance(module, (LoRALinear, LoRAConv3d)):
            continue
        # 遍历该模块的直接子模块进行可能的替换
        for child_name, child in module.named_children():
            if child_name == 'base':
                continue  # 跳过包装层内部引用
            if isinstance(child, (LoRALinear, LoRAConv3d)):
                continue  # 已经注入
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and is_lora_target(full_name, child, target_substrings):
                lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                linear_count += 1
                if verbose:
                    print(f"[LoRA][Linear] Injected at {full_name}")
            elif include_conv3d and isinstance(child, nn.Conv3d) and is_lora_target(full_name, child, target_substrings):
                lora_layer = LoRAConv3d(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, child_name, lora_layer)
                conv_count += 1
                if verbose:
                    print(f"[LoRA][Conv3d] Injected at {full_name}")
    return linear_count, conv_count


def get_lora_parameters(model: nn.Module):
    """获取所有 LoRA 适配分支的参数 (不含已合并或基础层权重)。"""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params.append(m.lora_A)
            params.append(m.lora_B)
        elif isinstance(m, LoRAConv3d):
            params.append(m.lora_A.weight)
            params.append(m.lora_B.weight)
    return params


def merge_lora_weights(model: nn.Module):
    """合并所有支持 merge 的 LoRA 层。"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge_weights()
        # Conv3d 目前保留动态分支不合并

def clone_and_merge_lora(model: nn.Module) -> nn.Module:
    """克隆一个模型副本并在副本中合并 LoRA 权重，返回用于推理的合并后模型。
    原模型保持未合并状态，便于继续微调。
    """
    import copy
    merged = copy.deepcopy(model)
    merge_lora_weights(merged)
    return merged

__all__ = [
    'LoRALinear', 'LoRAConv3d', 'inject_lora', 'get_lora_parameters', 'merge_lora_weights', 'clone_and_merge_lora'
]
