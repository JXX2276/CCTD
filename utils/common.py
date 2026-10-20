import numpy as np
from torch import nn
import torch

def min_max_normalize(hsi):
    """
    向量化实现的高光谱图像逐波段最大最小归一化
    
    参数:
        hsi: 高光谱图像 numpy数组 [height, width, bands]
    
    返回:
        归一化后的高光谱图像
    """
    # 计算每个波段的最小最大值
    min_vals = np.min(hsi, axis=(0, 1), keepdims=True)
    max_vals = np.max(hsi, axis=(0, 1), keepdims=True)
    
    # 避免除以零
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # 当max=min时，设为1避免除以零
    
    # 归一化
    normalized_hsi = (hsi - min_vals) / range_vals
    
    return normalized_hsi



def grad_fn(image):
    grad_x = torch.zeros_like(image)
    grad_x[:, :-1, :] = image[:, 1:, :] - image[:, :-1, :]
    grad_y = torch.zeros_like(image)
    grad_y[:-1, :, :] = image[1:, :, :] - image[:-1, :, :]
    return grad_x, grad_y