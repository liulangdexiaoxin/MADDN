import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.ndimage import zoom
from scipy import ndimage
import logging
from augment import augment  # 添加导入
import torchio as tio

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

train_transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
    tio.RandomNoise(mean=0, std=0.01),
    tio.RandomBiasField(),
])

class MRIDataset(Dataset):
    def __init__(self, data_dir, classes=None, mode='train', target_size=128, transform=None, domain_name: str = None):
        """
        初始化3D MRI数据集
        
        参数:
            data_dir: 数据目录路径
            classes: 类别列表，例如 ['AD', 'CN']
            mode: 数据集模式 ('train', 'val', 'test')
            target_size: 目标尺寸，默认为128
            transform: 数据增强变换
        """
        self.data_dir = data_dir
        self.mode = mode
        self.target_size = target_size
        self.transform = transform
        self.domain_name = domain_name or 'PRIMARY'
        self.classes = classes if classes else ['AD', 'CN']  # 0: AD, 1: CN
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 根据模式确定数据目录
        self.mode_dir = os.path.join(data_dir, mode)
        
        # 加载文件路径和标签
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.mode_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory {class_dir} does not exist!")
                continue
                
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                    self.samples.append((
                        os.path.join(class_dir, file_name),
                        self.class_to_idx[class_name]
                    ))
        
        logger.info(f'{mode} dataset size: {len(self.samples)}')
        if len(self.samples) == 0:
            logger.error(f"No samples found in {class_dir}. Check the directory structure and file extensions.")
    
    def _crop_black_borders(self, volume, threshold=0.1):
        """裁剪黑边"""
        # 创建掩码，识别非零区域
        mask = volume > threshold * volume.max()
        
        # 如果没有非零区域，返回原始体积
        if not np.any(mask):
            return volume
            
        # 获取非零区域的坐标
        # 使用 np.nonzero 获取坐标以避免 lint 警告
        coords = np.array(np.nonzero(mask))
        
        # 计算边界框
        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1)
        
        # 裁剪体积
        cropped_volume = volume[
            min_coords[0]:max_coords[0] + 1,
            min_coords[1]:max_coords[1] + 1,
            min_coords[2]:max_coords[2] + 1
        ]
        
        return cropped_volume
    
    def _resize_volume(self, volume, target_size=128):
        """调整体积大小"""
        # 计算缩放因子
        factors = [target_size / dim for dim in volume.shape]
        
        # 使用线性插值调整体积大小
        resized_volume = zoom(volume, factors, order=1)
        
        return resized_volume
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载NIFTI文件
        try:
            img = nib.load(img_path)
            data = img.get_fdata(dtype=np.float32)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {str(e)}")
            # 返回一个空数据或跳过，这里简单返回一个零张量
            data = np.zeros((self.target_size, self.target_size, self.target_size), dtype=np.float32)
            return torch.from_numpy(data).float(), label, img_path
        
        # 确保数据是3D的 (H, W, D)
        if data.ndim == 4:
            # 如果是4D，取第一个体积
            data = data[..., 0]
        
        # 裁剪黑边
        data = self._crop_black_borders(data)
        
        # 调整大小到目标尺寸
        data = self._resize_volume(data, self.target_size)
        
        # 数据增强（仅训练集）
        if self.mode == 'train':
            data = augment(data)
        
        # 转换为PyTorch张量，保持3D格式 (H, W, D)
        data = torch.from_numpy(data).float()
        
        # 添加通道维度 (C, H, W, D)
        data = data.unsqueeze(0)
        
        if self.transform and self.mode == 'train':
            data = self.transform(data[np.newaxis, ...])  # torchio expects (C, D, H, W)
            data = data.squeeze(0)
        return data, label, img_path, self.domain_name

def get_data_loaders(config):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        config: 配置对象
    
    返回:
        train_loader, val_loader, test_loader: 数据加载器
    """
    # 检查数据根目录是否存在
    if not os.path.exists(config.data.data_root):
        raise ValueError(f"Data root directory {config.data.data_root} does not exist!")
    
    logger.info(f"Looking for data in {config.data.data_root}")
    
    # 创建数据集 - 直接使用预定义的划分
    train_dataset = MRIDataset(
        config.data.data_root,
        mode='train',
        target_size=config.data.target_size,
        domain_name='ADNI'
    )
    
    val_dataset = MRIDataset(
        config.data.data_root,
        mode='val',
        target_size=config.data.target_size,
        domain_name='ADNI'
    )
    
    test_dataset = MRIDataset(
        config.data.data_root,
        mode='test',
        target_size=config.data.target_size,
        domain_name='ADNI'
    )

    # 可选加载第二域数据 (仅 train，用于迁移/域适配)。保持默认关闭不影响原行为。
    if getattr(config.training, 'enable_domain', False) and getattr(config.data, 'secondary_root', None):
        sec_root = config.data.secondary_root
        if os.path.isdir(sec_root):
            secondary_train = MRIDataset(
                sec_root,
                mode='train',
                target_size=config.data.target_size,
                domain_name='SECONDARY'
            )
            from torch.utils.data import ConcatDataset
            train_dataset = ConcatDataset([train_dataset, secondary_train])
            logger.info(f"Combined train dataset size (primary+secondary): {len(train_dataset)}")
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # 统计每个类别样本数，计算权重
    # 若使用组合数据集，samples 结构不同，尝试兼容
    # 计算加权采样（兼容 ConcatDataset）
    all_labels = []
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        for ds in train_dataset.datasets:
            if hasattr(ds, 'samples'):
                all_labels.extend([lbl for _, lbl in ds.samples])
    else:
        all_labels = [lbl for _, lbl in train_dataset.samples]
    sampler = None
    if all_labels:
        class_counts = np.bincount(np.array(all_labels).astype(int))
        weights = 1.0 / class_counts
        sample_weights = weights[np.array(all_labels).astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=sampler if sampler is not None else None,
        shuffle=(sampler is None),
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
    shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    from config import Config
    config = Config()
    
    # 获取数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 测试数据加载
    print("训练集批次数量:", len(train_loader))
    print("验证集批次数量:", len(val_loader))
    print("测试集批次数量:", len(test_loader))
    
    # 获取一个批次的数据
    for batch_idx, (data, labels, paths) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"数据形状: {data.shape}")  # 应该是 (batch_size, 1, 128, 128, 128)
        print(f"标签: {labels}")
        print(f"路径示例: {paths[0]}")
        break