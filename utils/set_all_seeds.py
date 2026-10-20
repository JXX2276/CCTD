import random
import numpy as np
import torch


def set_all_seeds(seed=42, deterministic=False):
    """
    增强版的随机种子设置函数
    
    参数:
        seed (int): 随机种子值
        deterministic (bool): 是否启用完全确定性模式（可能影响性能）
    """
    # Python内置随机模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多GPU
        if deterministic:
            torch.backends.cudnn.deterministic = deterministic
