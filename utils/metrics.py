import numpy as np
import scipy.ndimage
from skimage.metrics import structural_similarity as ssim

def calculate_psnr3d(original, reconstructed, max_pixel=1.0):
    """
    计算3D PSNR (Peak Signal-to-Noise Ratio)
    对每个波段分别计算2D PSNR，然后取平均
    
    参数:
        original: 原始高光谱图像 (H, W, B)
        reconstructed: 重建高光谱图像 (H, W, B)
        max_pixel: 最大像素值 (默认1.0)
    
    返回:
        平均psnr值
    """
    psnr_values = []
    for b in range(original.shape[2]):
        band_orig = original[:, :, b]
        band_recon = reconstructed[:, :, b]
        mse = np.mean((band_orig - band_recon) ** 2)
        if mse == 0:
            psnr_values.append(float('inf'))
        else:
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            psnr_values.append(psnr)
    return np.mean(psnr_values)

def calculate_ssim3d(original, reconstructed, max_pixel=1.0, win_size=7):
    """
    计算3D SSIM (Structural Similarity Index)
    
    参数:
        original: 原始高光谱图像 (H, W, B)
        reconstructed: 重建高光谱图像 (H, W, B)
        max_pixel: 最大像素值 (默认1.0)
        win_size: 滑动窗口大小 (默认7)
    
    返回:
        ssim值
    """
    # 对于3D SSIM，我们计算每个波段的2D SSIM然后取平均
    ssim_values = []
    for b in range(original.shape[2]):
        band_orig = original[:, :, b]
        band_recon = reconstructed[:, :, b]
        ssim_b = ssim(band_orig, band_recon, data_range=max_pixel, win_size=win_size)
        ssim_values.append(ssim_b)
    return np.mean(ssim_values)

def calculate_ergas(original, reconstructed, scale_factor=4):
    """
    计算ERGAS (Relative Dimensionless Global Error in Synthesis)
    
    参数:
        original: 原始高光谱图像 (H, W, B)
        reconstructed: 重建高光谱图像 (H, W, B)
        scale_factor: 下采样比例因子 (默认4)
    
    返回:
        ergas值
    """
    # 计算每个波段的RMSE
    rmse_per_band = np.sqrt(np.mean((original - reconstructed)**2, axis=(0,1)))
    
    # 计算每个波段的均值
    mean_per_band = np.mean(original, axis=(0,1))
    
    # 避免除以零
    valid_bands = mean_per_band != 0
    rmse_per_band = rmse_per_band[valid_bands]
    mean_per_band = mean_per_band[valid_bands]
    
    # 计算相对误差
    relative_rmse = rmse_per_band / mean_per_band
    
    # 计算ERGAS
    ergas = 100 / scale_factor * np.sqrt(np.mean(relative_rmse**2))
    return ergas

def calculate_sam(original, reconstructed, return_degrees=True):
    """
    计算 SAM（Spectral Angle Mapper），可选择返回弧度或角度
    
    参数:
        original: 原始高光谱图像 (H, W, B)
        reconstructed: 重建高光谱图像 (H, W, B)
        return_degrees: 是否返回角度（默认 True）
    
    返回:
        SAM 值（默认单位：度）
    """
    # 重塑为 (N, B) 矩阵（N = H * W）
    orig_flat = original.reshape(-1, original.shape[2])
    recon_flat = reconstructed.reshape(-1, reconstructed.shape[2])
    
    # 计算点积和范数
    dot_product = np.sum(orig_flat * recon_flat, axis=1)
    norm_orig = np.linalg.norm(orig_flat, axis=1)
    norm_recon = np.linalg.norm(recon_flat, axis=1)
    
    # 避免除以零
    valid_mask = (norm_orig > 0) & (norm_recon > 0)
    cos_theta = dot_product[valid_mask] / (norm_orig[valid_mask] * norm_recon[valid_mask])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保数值稳定性
    
    # 计算平均角度（弧度）
    sam_rad = np.mean(np.arccos(cos_theta))
    
    # 转换为角度（可选）
    if return_degrees:
        return sam_rad * 180 / np.pi
    else:
        return sam_rad

def evaluation(original, reconstructed, metric_name, verbose=True, verbose_prefix=None, **kwargs):
    """
    计算指标并自动格式化输出
    
    参数:
        verbose: 是否打印结果 (默认True)
    """
    metric_functions = {
        'psnr': calculate_psnr3d,
        'ssim': calculate_ssim3d,
        'ergas': calculate_ergas,
        'sam': calculate_sam
    }
    
    # 计算指标
    if isinstance(metric_name, str):
        results = {metric_name: metric_functions[metric_name](original, reconstructed, **kwargs)}
    elif isinstance(metric_name, (list, tuple)):
        results = {name: metric_functions[name](original, reconstructed, **kwargs) 
                  for name in metric_name}
    else:
        raise TypeError("metric_name must be string or list")
    
    # 格式化输出
    formatted = {k: round(v, 4) if isinstance(v, (int, float)) else v 
                for k, v in results.items()}
    
    if verbose:
        if verbose_prefix is None:
            verbose_prefix = "Evaluation Results:"
        print_info = verbose_prefix + ' '
        for k, v in formatted.items():
            print_info +=  f"{k.upper()}: {v:.4f}, "
        print(print_info)
    
    return results
