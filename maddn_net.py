"""
MADDN: Multi-scale Alzheimer's Disease Detection Network
不共享 transformer 权重的多尺度 3D Vision Transformer 融合网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class PatchEmbed3D(nn.Module):
    """3D特征图分块嵌入"""
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, D, H, W]
        x = self.proj(x)  # [B, embed_dim, D//ps, H//ps, W//ps]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

class Attention3D(nn.Module):
    """3D多头注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock3D(nn.Module):
    """3D Transformer块"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ScaleSpecificTransformer(nn.Module):
    """尺度特定的Transformer编码器"""
    def __init__(self, in_channels, patch_size, embed_dim, depth, num_heads, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 65, embed_dim))  # 64 patches + 1 cls token

        self.blocks = nn.ModuleList([
            TransformerBlock3D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        # 提取patch嵌入
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 添加cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x)

        # 标准化
        x = self.norm(x)

        # 返回cls token作为该尺度的表示
        return x[:, 0]

class MultiScaleViT3D(nn.Module):
    """多尺度3D Vision Transformer融合网络"""
    def __init__(self, feature_channels, num_classes=2, embed_dim=256, depth=2,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.feature_channels = feature_channels

        # 为每个尺度创建特定的Transformer编码器
        self.scale_transformers = nn.ModuleList()

        # 定义每个尺度的patch大小
        # 特征图尺寸: [32,32,32], [32,32,32], [16,16,16], [8,8,8], [4,4,4]
        # 目标: 都分成4x4x4的网格，所以patch大小分别为8,8,4,2,1
        patch_sizes = [8, 8, 4, 2, 1]

        for i, (channels, patch_size) in enumerate(zip(feature_channels, patch_sizes)):
            self.scale_transformers.append(
                ScaleSpecificTransformer(
                    in_channels=channels,
                    patch_size=patch_size,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate
                )
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * len(feature_channels), embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, feature_maps):
        """
        前向传播
        feature_maps: 包含5个特征图的列表，尺寸分别为:
            [B, 64, 32, 32, 32]
            [B, 64, 32, 32, 32]
            [B, 128, 16, 16, 16]
            [B, 256, 8, 8, 8]
            [B, 512, 4, 4, 4]
        """
        assert len(feature_maps) == len(self.feature_channels), "特征图数量与预期不符"

        # 对每个尺度的特征图应用Transformer
        scale_representations = []
        for i, feat in enumerate(feature_maps):
            representation = self.scale_transformers[i](feat)
            scale_representations.append(representation)

        # 将所有尺度的表示拼接
        combined = torch.cat(scale_representations, dim=1)

        # 通过分类器
        output = self.classifier(combined)

        return output

    def forward_features(self, feature_maps):
        """返回多尺度拼接的融合特征（分类前向量）。"""
        assert len(feature_maps) == len(self.feature_channels), "特征图数量与预期不符"
        scale_representations = []
        for i, feat in enumerate(feature_maps):
            representation = self.scale_transformers[i](feat)
            scale_representations.append(representation)
        combined = torch.cat(scale_representations, dim=1)
        return combined


class MADDN(nn.Module):
    """MADDN: Multi-scale Alzheimer's Disease Detection Network
    完整的3D ResNet与多尺度ViT融合模型（各尺度独立Transformer）
    """
    def __init__(self, backbone, num_classes=2, embed_dim=256, depth=2, num_heads=8):
        super(MADDN, self).__init__()
        self.backbone = backbone
        self.feature_channels = [64, 64, 128, 256, 512]

        # 多尺度ViT融合网络
        self.fusion_network = MultiScaleViT3D(
            feature_channels=self.feature_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )

    def forward(self, x):
        feature_maps = self.backbone.get_feature_maps(x)
        out = self.fusion_network(feature_maps)
        return out

    def forward_features(self, x):
        """获取分类前融合特征，用于迁移学习/域对齐。"""
        feature_maps = self.backbone.get_feature_maps(x)
        return self.fusion_network.forward_features(feature_maps)


# 测试
if __name__ == "__main__":

    class MockBackbone(nn.Module):
        def __init__(self):
            super(MockBackbone, self).__init__()

        def get_feature_maps(self, x):
            batch_size = x.shape[0]
            return [
                torch.randn(batch_size, 64, 32, 32, 32),  # Feature map 0
                torch.randn(batch_size, 64, 32, 32, 32),  # Feature map 1
                torch.randn(batch_size, 128, 16, 16, 16), # Feature map 2
                torch.randn(batch_size, 256, 8, 8, 8),    # Feature map 3
                torch.randn(batch_size, 512, 4, 4, 4)     # Feature map 4
            ]

    # 创建模型
    backbone = MockBackbone()
    model = MADDN(
        backbone,
        num_classes=2,
        embed_dim=256,
        depth=2,
        num_heads=8
    )

    # 模拟输入
    input_tensor = torch.randn(2, 1, 128, 128, 128)  # [batch, channel, depth, height, width]

    # 前向传播
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
