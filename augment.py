import numpy as np
import random

def random_flip(volume):
    """随机翻转3D体积"""
    axes = [0, 1, 2]
    for axis in axes:
        if random.random() > 0.5:
            volume = np.flip(volume, axis=axis)
    return volume

def random_rotate(volume):
    """随机旋转90度的3D体积"""
    k = random.randint(0, 3)
    axis = random.choice([(0, 1), (1, 2), (0, 2)])
    volume = np.rot90(volume, k, axes=axis)
    return volume

def random_noise(volume, noise_level=0.01):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level, volume.shape)
    volume = volume + noise
    return volume

def random_intensity(volume, factor_range=(0.9, 1.1)):
    """随机强度缩放"""
    factor = random.uniform(*factor_range)
    volume = volume * factor
    return volume

def augment(volume):
    """组合所有增强操作"""
    volume = random_flip(volume)
    volume = random_rotate(volume)
    volume = random_noise(volume)
    volume = random_intensity(volume)
    return volume