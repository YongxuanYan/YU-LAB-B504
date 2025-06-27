from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from ..var.globals import get_var, set_var
import cv2

def convert_array_to_pixmap(array):
    """将 NumPy 数组转换为 QPixmap，支持灰度图像和 RGB 图像"""
    if len(array.shape) not in [2, 3]:
        raise ValueError("Array must be 2D (grayscale) or 3D (RGB).")

    if array.dtype != np.uint8:
        array = array.astype(np.uint8)

    height, width = [512, 512]

    if len(array.shape) == 2:
        image = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
    elif len(array.shape) == 3 and array.shape[2] == 3:
        image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    else:
        raise ValueError("Invalid array shape for image display.")

    return QPixmap.fromImage(image)


def update_inputs_display(container, idx):
    imgs = get_var('Model_inputs_images')
    if imgs is None or imgs.size == 0:
        return
    if idx >= imgs.shape[0]:
        return

    ColorInputs = get_var("ColorInputs")
    arr = imgs[idx]
    if arr.shape[0] != 512 or arr.shape[1] != 512:
        arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    # 如果是灰度图，shape 是 (H, W) 或 (H, W, 1)
    if not ColorInputs:
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[2] != 1:
            raise ValueError("Expected grayscale image, but got multiple channels.")

    pixmap = convert_array_to_pixmap(arr)
    container.updateLeftImage(pixmap)


def update_outputs_display(container, idx):
    outs = get_var('Model_outputs_images')
    if outs is None or outs.size == 0:
        return
    if idx >= outs.shape[0]:
        return

    arr = outs[idx]
    if arr.shape[0] != 512 or arr.shape[1] != 512:
        arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    ColorInputs = get_var("ColorInputs")
    if not ColorInputs:
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim == 3 and arr.shape[2] != 1:
            raise ValueError("Expected grayscale image, but got multiple channels.")

    pixmap = convert_array_to_pixmap(arr)
    container.updateRightImage(pixmap)
