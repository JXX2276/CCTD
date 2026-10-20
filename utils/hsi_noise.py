import numpy as np

def _get_corrupt_bands(num_bands, corrupt_band_ratio):
    """Helper function to randomly select bands to corrupt.
    
    Args:
        num_bands (int): Total number of bands in the HSI
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
    
    Returns:
        numpy.ndarray: Array of band indices to be corrupted
    """
    num_corrupt = int(num_bands * corrupt_band_ratio)
    return np.random.choice(num_bands, num_corrupt, replace=False)

def add_gaussian_noise(hsi, std=0.1, corrupt_band_ratio=1.0):
    """Add Gaussian noise to hyperspectral image bands.
    
    Args:
        hsi (numpy.ndarray): Input HSI of shape (height, width, bands)
        std (float): Standard deviation of Gaussian noise
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
    
    Returns:
        numpy.ndarray: Noisy HSI clipped to [0, 1]
    """
    # Parameter validation
    if not 0 <= corrupt_band_ratio <= 1:
        raise ValueError("corrupt_band_ratio must be between 0 and 1")
    
    noisy_hsi = hsi.copy()
    corrupt_bands = _get_corrupt_bands(hsi.shape[2], corrupt_band_ratio)
    
    # Generate noise only for selected bands
    noise_shape = (hsi.shape[0], hsi.shape[1], len(corrupt_bands))
    noise = np.random.normal(0, std, noise_shape)
    noisy_hsi[..., corrupt_bands] += noise
    
    return np.clip(noisy_hsi, 0, 1)

def add_non_iid_gaussian_noise(image, std=(0.01, 0.1), corrupt_band_ratio=1.0):
    """Add band-dependent Gaussian noise with varying standard deviations.
    
    Args:
        image (numpy.ndarray): Input image normalized to [0,1]
        std (tuple): Range (min, max) of standard deviations for noise
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
    
    Returns:
        numpy.ndarray: Noisy image clipped to [0, 1]
    """
    # Parameter validation
    if not 0 <= corrupt_band_ratio <= 1:
        raise ValueError("corrupt_band_ratio must be between 0 and 1")
    if isinstance(std, (tuple, list)) and std[0] > std[1]:
        raise ValueError("std range must be (min, max)")
    
    noisy_image = image.copy()
    corrupt_bands = _get_corrupt_bands(image.shape[2], corrupt_band_ratio)
    
    # Generate different std values for each band
    noise_std = np.random.uniform(*std, size=len(corrupt_bands))
    
    # Vectorized noise generation for all corrupt bands
    noise = np.random.normal(0, noise_std, (image.shape[0], image.shape[1], len(corrupt_bands)))
    noisy_image[..., corrupt_bands] += noise
    
    return np.clip(noisy_image, 0, 1)

def _get_intensity(intensity):
    """Get noise intensity value, handling both single values and ranges.
    
    Args:
        intensity (float or tuple): Either a fixed value or (min, max) range
    
    Returns:
        float: The intensity value to use
    """
    if isinstance(intensity, (tuple, list)):
        return np.random.uniform(*intensity)
    return intensity

def add_impulse_noise(hsi, intensity=0.1, corrupt_band_ratio=1):
    """Add salt-and-pepper noise to selected bands.
    
    Args:
        hsi (numpy.ndarray): Input HSI of shape (height, width, bands)
        intensity (float or tuple): Corruption probability or range (min, max)
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
    
    Returns:
        numpy.ndarray: Noisy HSI with impulse noise
    """
    if not 0 <= corrupt_band_ratio <= 1:
        raise ValueError("corrupt_band_ratio must be between 0 and 1")
    if isinstance(intensity, (tuple, list)) and intensity[0] > intensity[1]:
        raise ValueError("intensity range must be (min, max)")
    
    noisy_hsi = hsi.copy()
    corrupt_bands = _get_corrupt_bands(hsi.shape[2], corrupt_band_ratio)
    
    for band in corrupt_bands:
        current_intensity = _get_intensity(intensity)
        # Create mask for corrupted pixels
        mask = np.random.random(hsi.shape[:2]) < current_intensity
        # Randomly select salt vs pepper
        salt = np.random.random(hsi.shape[:2]) > 0.5
        # Apply salt (max value) and pepper (min value)
        band_max, band_min = np.max(hsi[:, :, band]), np.min(hsi[:, :, band])
        noisy_hsi[:, :, band][mask & salt] = band_max
        noisy_hsi[:, :, band][mask & ~salt] = band_min
    
    return noisy_hsi

def add_stripe_noise(hsi, intensity=0.1, corrupt_band_ratio=1, vertical=True):
    """Add stripe noise (vertical or horizontal) to selected bands.
    
    Args:
        hsi (numpy.ndarray): Input HSI of shape (height, width, bands)
        intensity (float or tuple): Proportion of columns/rows to corrupt
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
        vertical (bool): If True add vertical stripes, else horizontal
    
    Returns:
        numpy.ndarray: Noisy HSI clipped to [0, 1]
    """
    if not 0 <= corrupt_band_ratio <= 1:
        raise ValueError("corrupt_band_ratio must be between 0 and 1")
    if isinstance(intensity, (tuple, list)) and intensity[0] > intensity[1]:
        raise ValueError("intensity range must be (min, max)")
        
    noisy_hsi = hsi.copy()
    height, width, _ = hsi.shape
    corrupt_bands = _get_corrupt_bands(hsi.shape[2], corrupt_band_ratio)
    
    for band in corrupt_bands:
        current_intensity = _get_intensity(intensity)
        band_std = np.std(hsi[:, :, band])
        
        if vertical:  # Vertical stripes
            num_stripes = max(1, int(width * current_intensity))
            cols = np.random.choice(width, num_stripes, replace=False)
            # Add noise proportional to band standard deviation
            noisy_hsi[:, cols, band] += np.random.normal(0, band_std*0.5, (height, num_stripes))
        else:  # Horizontal stripes
            num_stripes = max(1, int(height * current_intensity))
            rows = np.random.choice(height, num_stripes, replace=False)
            noisy_hsi[rows, :, band] += np.random.normal(0, band_std*0.5, (num_stripes, width))
    
    return np.clip(noisy_hsi, 0, 1)

def add_deadline_noise(hsi, intensity=0.1, corrupt_band_ratio=1, vertical=True):
    """Add deadline noise (missing rows/columns) to selected bands.
    
    Args:
        hsi (numpy.ndarray): Input HSI of shape (height, width, bands)
        intensity (float or tuple): Proportion of columns/rows to corrupt
        corrupt_band_ratio (float): Proportion of bands to corrupt (0-1)
        vertical (bool): If True add vertical lines, else horizontal
    
    Returns:
        numpy.ndarray: Noisy HSI with missing lines
    """
    if not 0 <= corrupt_band_ratio <= 1:
        raise ValueError("corrupt_band_ratio must be between 0 and 1")
    if isinstance(intensity, (tuple, list)) and intensity[0] > intensity[1]:
        raise ValueError("intensity range must be (min, max)")
        
    noisy_hsi = hsi.copy()
    height, width, _ = hsi.shape
    corrupt_bands = _get_corrupt_bands(hsi.shape[2], corrupt_band_ratio)
    
    for band in corrupt_bands:
        current_intensity = _get_intensity(intensity)
        if vertical:  # Vertical deadlines
            num_lines = max(1, int(width * current_intensity))
            cols = np.random.choice(width, num_lines, replace=False)
            noisy_hsi[:, cols, band] = 0
        else:  # Horizontal deadlines
            num_lines = max(1, int(height * current_intensity))
            rows = np.random.choice(height, num_lines, replace=False)
            noisy_hsi[rows, :, band] = 0
    
    return noisy_hsi