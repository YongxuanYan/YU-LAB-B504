from ..var.globals import get_var, set_var
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def convert_array_to_pixmap(array):
    """将 NumPy 数组转换为 QPixmap"""
    # 确保数组为 2D 或 3D
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array must be 2D (grayscale) or 3D (RGB).")

    # 如果是 2D 数组（灰度），扩展为 3D
    if len(array.shape) == 2:
        array = np.stack((array, array, array), axis=-1)

    # 数组类型检查和转换
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)  # 假设输入为 [0, 1] 范围，转换为 [0, 255]

    # 获取数组形状
    height, width, channel = array.shape

    # 确保是 RGB 格式
    if channel == 3:
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        raise ValueError("Array must have 3 channels (RGB).")

    # 转换为 QPixmap
    pixmap = QPixmap.fromImage(image)
    return pixmap


def convert_array_to_pixmap_window_WL(array):
    """
    将 NumPy 数组转换为 QPixmap，并调整窗宽和窗位。

    参数:
    - array: NumPy 数组 (2D 或 3D)
    - window_width: 窗宽 (window width)
    - window_level: 窗位 (window level)

    返回:
    - QPixmap 对象
    """
    window_level = get_var("WindowLevel")
    window_width = get_var("WindowWidth")
    # 确保数组为 2D 或 3D
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array must be 2D (grayscale) or 3D (RGB).")

    # 如果是 2D 数组（灰度），扩展为 3D
    if len(array.shape) == 2:
        array = np.stack((array, array, array), axis=-1)

    # 窗宽窗位调整
    min_value = window_level - 0.5 * window_width
    max_value = window_level + 0.5 * window_width
    array = np.clip(array, min_value, max_value)  # 裁剪到窗宽窗位范围
    array = (array - min_value) / (max_value - min_value) * 255.0  # 归一化到 [0, 255]
    array = np.clip(array, 0, 255).astype(np.uint8)  # 转为 uint8 类型

    # 获取数组形状
    height, width, channel = array.shape

    # 转换为 QImage
    if channel == 3:
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        raise ValueError("Array must have 3 channels (RGB).")

    # 转换为 QPixmap
    pixmap = QPixmap.fromImage(image)
    return pixmap


def update_display_ct(C):
    """C: Container"""
    TotalSliceNum = get_var("SliceNum")
    CurrentSlice = min(get_var("CurrentSlice"), TotalSliceNum)
    CurrentSlice = max(CurrentSlice, 1)
    set_var("CurrentSlice", CurrentSlice)

    ctexist = get_var("ctexist")
    if ctexist:
        ct_slices = get_var("PixelsGrid")  # Assuming PixelsGrid is stored as a global variable
        left_image = convert_array_to_pixmap_window_WL(ct_slices[:, :, CurrentSlice - 1])
        C.ct_updateLeftImage(left_image)

    labelexits = get_var("labelexits")
    if labelexits:
        label_slices = get_var("labeldata")  # Assuming labeldata is stored as a global variable
        current_slice_array = label_slices[:, :, CurrentSlice - 1]
        current_slice_array = np.clip(current_slice_array * 255, 0, 255).astype(np.uint8)
        right_image = convert_array_to_pixmap(current_slice_array)
        C.ct_updateRightImage(right_image)

