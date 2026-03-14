import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    """
    基础块类，用于构建ResNet网络中的基本残差块
    包含两个3x3卷积层，每个卷积层后接批量归一化和ReLU激活函数
    """
    expansion: int = 1  # 扩展系数，用于控制输出通道数，在BasicBlock中保持为1

    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU(inplace=True) # inplace=True 这可以节省内存，但在某些情况下可能会影响梯度计算
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    """
    Bottleneck类实现了一个三维卷积网络中的瓶颈模块，这是ResNet架构中的一种基本构建块。
    它通过1x1和3x3卷积的组合，实现了降维和升维，同时保持计算效率。
    expansion属性用于控制输出通道数的扩展倍数。
    """
    expansion: int = 4  # 输出通道数的扩展倍数，默认为4

    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel) # 1x1卷积，降维
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.conv2 = conv3x3(out_channel, out_channel, stride) # 3x3卷积，特征提取
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.conv3 = conv1x1(out_channel, out_channel * self.expansion) # 1x1卷积，升维
        self.bn3 = nn.BatchNorm3d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(
            self,  # 构造函数
            block: Type[Union[BasicBlock, Bottleneck]],  # 块类型，可以是BasicBlock或Bottleneck
            layers: List[int],  # 每个残差块层的块数量列表
            num_classes: int = 1000,  # 分类数量，默认为1000
            zero_init_residual: bool = False,  # 是否零初始化残差分支，默认为False
            in_channels: int = 1,  # 输入通道数，默认为1
    ) -> None:  # 返回类型为None
        super(ResNet3D, self).__init__()  # 调用父类的构造函数
        self.in_channel = 64  # 初始输入通道数设为64

        # 初始卷积层
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 3D卷积层
        self.bn1 = nn.BatchNorm3d(64)  # 3D批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # 3D最大池化层

        # 残差块层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一个残差块层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 第二个残差块层，步长为2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 第三个残差块层，步长为2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 第四个残差块层，步长为2

        # 自适应平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))  # 3D自适应平均池化层
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层，用于分类

        # 权重初始化
        for m in self.modules():  # 遍历所有模块
            if isinstance(m, nn.Conv3d):  # 如果是3D卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 使用Kaiming正态分布初始化权重
            elif isinstance(m, nn.BatchNorm3d):  # 如果是3D批归一化层
                nn.init.constant_(m.weight, 1)  # 权重初始化为1
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0

        # 零初始化残差分支中的最后一个BN层
        if zero_init_residual:  # 如果开启零初始化残差
            for m in self.modules():  # 遍历所有模块
                if isinstance(m, Bottleneck):  # 如果是Bottleneck块
                    nn.init.constant_(m.bn3.weight, 0)  # 将最后一个BN层的权重初始化为0
                elif isinstance(m, BasicBlock):  # 如果是BasicBlock块
                    nn.init.constant_(m.bn2.weight, 0)  # 将最后一个BN层的权重初始化为0

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],  # block类型，可以是BasicBlock或Bottleneck
            channels: int,  # 输出通道数
            blocks: int,
            stride: int = 1,  # 步长，默认为1
    ) -> nn.Sequential:  # 返回一个Sequential容器
        # 初始化下采样层，当步长不为1或输入通道数不匹配时需要下采样
        downsample = None
        if stride != 1 or self.in_channel != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channel, channels * block.expansion, stride),
                nn.BatchNorm3d(channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channels, stride, downsample))
        self.in_channel = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """返回分类前的扁平特征，用于迁移学习 / CORAL / 蒸馏。
        不影响原 forward 行为；调用者需自行再经过 self.fc。
        """
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取不同尺寸的特征图"""
        feature_maps = []

        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feature_maps.append(x)  # 1/4

        # 残差块
        x = self.layer1(x)
        feature_maps.append(x)  # 1/4
        x = self.layer2(x)
        feature_maps.append(x)  # 1/8
        x = self.layer3(x)
        feature_maps.append(x)  # 1/16
        x = self.layer4(x)
        feature_maps.append(x)  # 1/32

        return feature_maps


def resnet18_3d(**kwargs: Any) -> ResNet3D:
    """构建3D ResNet-18模型"""
    return ResNet3D(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50_3d(**kwargs: Any) -> ResNet3D:
    """构建3D ResNet-50模型"""
    return ResNet3D(Bottleneck, [3, 4, 6, 3], **kwargs)


# 示例用法
if __name__ == "__main__":
    model = resnet18_3d(num_classes=2, in_channels=1)
    print(model)
    input_tensor = torch.randn(1, 1, 128, 128, 128)

    output = model(input_tensor)
    print(f"Output shape: {output.shape}")

    feature_maps = model.get_feature_maps(input_tensor)
    for i, feat in enumerate(feature_maps):
        print(f"Feature map {i} shape: {feat.shape}")