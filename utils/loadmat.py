import scipy.io
import h5py
import warnings
import numpy as np
from collections import defaultdict

def loadmat(file_path, hdf5=None):
    """
    Load a .mat file, selecting the loader based on `hdf5` parameter:
    - hdf5=None: Auto-detect version (HDF5 ≥v7.3 uses h5py, else scipy)
    - hdf5=True: Force use h5py
    - hdf5=False: Force use scipy.io.loadmat

    Parameters:
        file_path (str): Path to the .mat file.
        hdf5 (bool, optional): Loader selection flag.

    Returns:
        data (dict): Dictionary of loaded variables.
    """
    # Determine loader based on `hdf5` parameter
    if hdf5 is True:
        data = _load_h5py_mat(file_path)
        print("Loaded using h5py")
    elif hdf5 is False:
        data = scipy.io.loadmat(file_path)
        data = _sanitize_scipy_data(data)
        print("Loaded using scipy.io.loadmat")
    else:  # hdf5=None: auto-detect
        try:
            data = scipy.io.loadmat(file_path)
            data = _sanitize_scipy_data(data)
            print("Loaded using scipy.io.loadmat (auto-detected)")
        except:
            data = _load_h5py_mat(file_path)
            print("Loaded using h5py (auto-detected)")
    
    

    print("\nMAT File Contents:")
    print(_format_table(data))
    print("\n")

    return data

def _sanitize_scipy_data(data):
    """Remove MATLAB-specific metadata from scipy-loaded dictionaries."""
    return {k: v for k, v in data.items() if not k.startswith('__')}

def _load_h5py_mat(file_path):
    """Load HDF5-based MATLAB files (v7.3+) using h5py."""
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('#'):  # Skip metadata groups
                data[key] = _extract_h5py_data(f[key])
    return data

def _extract_h5py_data(dataset):
    """Recursively extract data from h5py objects."""
    if isinstance(dataset, h5py.Group):
        return {k: _extract_h5py_data(dataset[k]) for k in dataset.keys()}
    elif isinstance(dataset, h5py.Dataset):
        arr = dataset[()]
        # Convert MATLAB's dimension order (e.g., CHW to HWC)
        if arr.ndim > 1:
            arr = arr.transpose(tuple(reversed(range(arr.ndim))))
        return arr
    return dataset

def _format_table(data):
    """生成ASCII表格"""
    # 收集数据
    rows = []
    for key in sorted(data.keys()):
        val = data[key]
        if isinstance(val, dict):
            continue  # 跳过嵌套结构
        dtype = val.dtype.name if isinstance(val, np.ndarray) else type(val).__name__
        shape = val.shape if isinstance(val, np.ndarray) else '-N/A-'
        rows.append((key, dtype, shape))
    
    # 计算列宽
    cols = defaultdict(int)
    cols['Name'] = max(len('Name'), max(len(r[0]) for r in rows))
    cols['Type'] = max(len('Type'), max(len(r[1]) for r in rows))
    cols['Shape'] = max(len('Shape'), max(len(str(r[2])) for r in rows))
    
    # 生成表格
    border = '+' + '+'.join('-'*(cols[k]+2) for k in ['Name','Type','Shape']) + '+'
    header = f"| {'Name'.ljust(cols['Name'])} | {'Type'.ljust(cols['Type'])} | {'Shape'.ljust(cols['Shape'])} |"
    separator = f"+ {'-'*cols['Name']} + {'-'*cols['Type']} + {'-'*cols['Shape']} +"
    
    table = [border, header, separator]
    for row in rows:
        table.append(f"| {row[0].ljust(cols['Name'])} | {row[1].ljust(cols['Type'])} | {str(row[2]).ljust(cols['Shape'])} |")
    table.append(border)
    return '\n'.join(table)        



# import scipy.io
# import h5py
# import warnings
# import numpy as np
# from collections import defaultdict

# # ANSI color codes
# BLUE = '\033[94m'
# ENDC = '\033[0m'

# def loadmat(file_path, hdf5=None):
#     """
#     Load a .mat file, selecting the loader based on `hdf5` parameter:
#     - hdf5=None: Auto-detect version (HDF5 ≥v7.3 uses h5py, else scipy)
#     - hdf5=True: Force use h5py
#     - hdf5=False: Force use scipy.io.loadmat

#     Parameters:
#         file_path (str): Path to the .mat file.
#         hdf5 (bool, optional): Loader selection flag.

#     Returns:
#         data (dict): Dictionary of loaded variables.
#     """
#     # Determine loader based on `hdf5` parameter
#     if hdf5 is True:
#         data = _load_h5py_mat(file_path)
#         print(f"{BLUE}Loaded using h5py{ENDC}")
#     elif hdf5 is False:
#         data = scipy.io.loadmat(file_path)
#         data = _sanitize_scipy_data(data)
#         print(f"{BLUE}Loaded using scipy.io.loadmat{ENDC}")
#     else:  # hdf5=None: auto-detect
#         try:
#             data = scipy.io.loadmat(file_path)
#             data = _sanitize_scipy_data(data)
#             print(f"{BLUE}Loaded using scipy.io.loadmat (auto-detected){ENDC}")
#         except:
#             data = _load_h5py_mat(file_path)
#             print(f"{BLUE}Loaded using h5py (auto-detected){ENDC}")
    
#     print(f"{BLUE}\nMAT File Contents:{ENDC}")
#     print(f"{BLUE}{_format_table(data)}{ENDC}")
#     print(f"{BLUE}\n{ENDC}")

#     return data

# def _sanitize_scipy_data(data):
#     """Remove MATLAB-specific metadata from scipy-loaded dictionaries."""
#     return {k: v for k, v in data.items() if not k.startswith('__')}

# def _load_h5py_mat(file_path):
#     """Load HDF5-based MATLAB files (v7.3+) using h5py."""
#     data = {}
#     with h5py.File(file_path, 'r') as f:
#         for key in f.keys():
#             if not key.startswith('#'):  # Skip metadata groups
#                 data[key] = _extract_h5py_data(f[key])
#     return data

# def _extract_h5py_data(dataset):
#     """Recursively extract data from h5py objects."""
#     if isinstance(dataset, h5py.Group):
#         return {k: _extract_h5py_data(dataset[k]) for k in dataset.keys()}
#     elif isinstance(dataset, h5py.Dataset):
#         arr = dataset[()]
#         # Convert MATLAB's dimension order (e.g., CHW to HWC)
#         if arr.ndim > 1:
#             arr = arr.transpose(tuple(reversed(range(arr.ndim))))
#         return arr
#     return dataset

# def _format_table(data):
#     """生成ASCII表格"""
#     # 收集数据
#     rows = []
#     for key in sorted(data.keys()):
#         val = data[key]
#         if isinstance(val, dict):
#             continue  # 跳过嵌套结构
#         dtype = val.dtype.name if isinstance(val, np.ndarray) else type(val).__name__
#         shape = val.shape if isinstance(val, np.ndarray) else '-N/A-'
#         rows.append((key, dtype, shape))
    
#     # 计算列宽
#     cols = defaultdict(int)
#     cols['Name'] = max(len('Name'), max(len(r[0]) for r in rows))
#     cols['Type'] = max(len('Type'), max(len(r[1]) for r in rows))
#     cols['Shape'] = max(len('Shape'), max(len(str(r[2])) for r in rows))
    
#     # 生成表格
#     border = '+' + '+'.join('-'*(cols[k]+2) for k in ['Name','Type','Shape']) + '+'
#     header = f"| {'Name'.ljust(cols['Name'])} | {'Type'.ljust(cols['Type'])} | {'Shape'.ljust(cols['Shape'])} |"
#     separator = f"+ {'-'*cols['Name']} + {'-'*cols['Type']} + {'-'*cols['Shape']} +"
    
#     table = [border, header, separator]
#     for row in rows:
#         table.append(f"| {row[0].ljust(cols['Name'])} | {row[1].ljust(cols['Type'])} | {str(row[2]).ljust(cols['Shape'])} |")
#     table.append(border)
#     return '\n'.join(table)