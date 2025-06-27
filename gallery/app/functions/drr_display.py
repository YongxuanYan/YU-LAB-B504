from ..var.globals import get_var, set_var
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt

def convert_array_to_pixmap(array):
    """将 NumPy 数组转换为 QPixmap，支持灰度图像和 RGB 图像"""
    # 确保数组为 2D 或 3D
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array must be 2D (grayscale) or 3D (RGB).")

    # 数组类型检查和转换
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)  # 假设输入为 [0, 1] 范围，转换为 [0, 255]

    # 获取数组形状
    height, width = array.shape[:2]

    # 处理灰度图像
    if len(array.shape) == 2:  # 灰度图像
        image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
    elif len(array.shape) == 3 and array.shape[2] == 3:  # RGB 图像
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        raise ValueError("Array must be either 2D (grayscale) or 3D with 3 channels (RGB).")

    # 转换为 QPixmap
    pixmap = QPixmap.fromImage(image)
    return pixmap


def update_display_drr(C):
    """C: Container"""
    TotalSliceNum = 3
    CurrentSlice = min(get_var("CurrentDRRSlice"), TotalSliceNum)
    CurrentSlice = max(CurrentSlice, 1)
    set_var("CurrentDRRSlice", CurrentSlice)

    leftdrrexist = get_var("DRREXISTS_Left")
    if leftdrrexist:
        drr_slice = get_var("Left_DRR")  # Assuming PixelsGrid is stored as a global variable
        left_image = convert_array_to_pixmap(drr_slice[:, :, CurrentSlice - 1])
        C.drr_updateLeftImage(left_image)

    rightdrrexist = get_var("DRREXISTS_Right")
    if rightdrrexist:
        drr_slice = get_var("Right_DRR")  # Assuming PixelsGrid is stored as a global variable
        right_image = convert_array_to_pixmap(drr_slice[:, :, CurrentSlice - 1])
        C.drr_updateRightImage(right_image)


