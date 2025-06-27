import os
import numpy as np
import pydicom
from ..var.globals import set_var, del_var
from .listdlg import select_contour_from_dialog
import cv2


def fillin(data, body):
    """
    Perform morphological closing on each slice and fill holes using OpenCV.
    Args:
        data (numpy.ndarray): 3D binary label data.
    Returns:
        numpy.ndarray: Processed 3D label data.
    """
    processed_data = np.zeros_like(data)
    height, width = data.shape[:2]

    for z in range(data.shape[2]):
        slice_data = data[:, :, z] > 0
        if np.any(slice_data):
            # Find the coordinates of the non-zero pixels
            y_coords, x_coords = np.where(slice_data)

            # Calculate the bounding box of the non-zero region
            top, bottom = y_coords.min(), y_coords.max()
            left, right = x_coords.min(), x_coords.max()

            # Calculate vertical and horizontal distances
            vertical_distance = bottom - top
            horizontal_distance = right - left

            # Determine the radius
            min_distance = min(vertical_distance, horizontal_distance)
            radius = max(1, (min_distance // 2) + 10)

            if body:
                pad_size = radius
                padded_data = np.pad(slice_data.astype(np.uint8), pad_size, mode='constant', constant_values=0)

                # Create a structuring element with the calculated radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

                # Perform morphological closing on the padded image
                closed_data = cv2.morphologyEx(padded_data, cv2.MORPH_CLOSE, kernel)

                # Fill holes on the padded image
                filled_data = cv2.floodFill(closed_data.copy(), None, (0, 0), 255)[1]
                filled_data = cv2.bitwise_or(closed_data, cv2.bitwise_not(filled_data))

                # Crop back to original size
                cropped_data = filled_data[pad_size:pad_size + height, pad_size:pad_size + width]

                # Store the result
                processed_data[:, :, z] = cropped_data.astype(np.uint8) * 255
            else:
                # Create a structuring element with the calculated radius
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

                # Perform morphological closing
                closed_data = cv2.morphologyEx(slice_data.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

                # Fill holes
                filled_data = cv2.floodFill(closed_data.copy(), None, (0, 0), 255)[1]
                filled_data = cv2.bitwise_or(closed_data, cv2.bitwise_not(filled_data))

                processed_data[:, :, z] = filled_data.astype(np.uint8) * 255

    return processed_data


def LoadCT(files: str):
    """
    Load CT data and return the corresponding values.

    Args:
        files (str): Path to the folder containing CT files.
        ContourSelection (int, MUST): Selected contour for RTSTRUCT. Default is 0.
    """
    files = [os.path.join(files, f) for f in os.listdir(files) if f.endswith('.dcm')]
    num_files = len(files)
    if num_files == 0:
        msg = f"No CT data found! please make sure *.dcm is included."
        return msg

    PixelsGrid = np.zeros((512, 512, num_files), dtype=np.float32)
    SliceLocation = np.zeros(num_files, dtype=np.float32)
    LocationOfNotCTData = []
    NotCTNum = 0
    RTexist = False
    labeldata = None

    info_collected = False
    PixelSpacing = None
    SliceThickness = None
    PatientName = None

    for i, file in enumerate(files):
        info = pydicom.dcmread(file)

        if info.Modality == 'CT':
            if not info_collected:
                PixelSpacing = np.array(info.PixelSpacing)
                SliceThickness = float(info.SliceThickness)
                ImagePositionPatient = np.array(info.ImagePositionPatient)
                PatientName = info.PatientName.family_name
                # 接下来几步是为了自动获取最大窗宽范围和窗位范围
                # 获取原始像素数据
                pixel_array = info.pixel_array
                # 获取 Rescale Intercept 和 Rescale Slope
                intercept = info.RescaleIntercept if 'RescaleIntercept' in info else 0
                slope = info.RescaleSlope if 'RescaleSlope' in info else 1
                # 转换为 Hounsfield Unit (HU)
                hu_array = pixel_array * slope + intercept
                # 获取 HU 的最小值和最大值
                min_window_level = hu_array.min()
                max_window_level = hu_array.max()
                max_window_width = max_window_level - min_window_level
                info_collected = True

            pixel_array = info.pixel_array.astype(np.float32)
            # 应用 Rescale Slope 和 Rescale Intercept 转换为 HU
            intercept = float(getattr(info, 'RescaleIntercept', 0))  # 默认值为 0
            slope = float(getattr(info, 'RescaleSlope', 1))  # 默认值为 1
            hu_array = pixel_array * slope + intercept

            # 存入 PixelsGrid
            PixelsGrid[:, :, i] = hu_array
            try:
                SliceLocation[i] = float(getattr(info, 'SliceLocation', info.ImagePositionPatient[2]))
            except AttributeError:
                SliceLocation[i] = float(info.ImagePositionPatient[2])

        elif info.Modality == 'RTSTRUCT':
            RTexist = True
            RTSTRUCT_location = i
        else:
            SliceLocation[i] = np.nan
            NotCTNum += 1
            LocationOfNotCTData.append(i)

    PixelsGrid = PixelsGrid - PixelsGrid.min() - 1000  # 将空气的CT值设为-1000
    ct_max = PixelsGrid.max()
    PixelsGrid = PixelsGrid[:, :, :-1]
    CT_Shape = PixelsGrid.shape
    PHeight = PixelSpacing[1] * CT_Shape[0]
    PWidth = PixelSpacing[0] * CT_Shape[1]
    PDepth = SliceThickness * CT_Shape[2]
    set_var('MaxWindowWidth', max_window_width)
    set_var('MaxWindowLevel', max_window_level)
    set_var('MinWindowLevel', min_window_level)
    del_var('PixelsGrid')
    set_var('PixelsGrid', PixelsGrid)
    set_var('ctexist', 1)
    set_var('SliceNum', CT_Shape[2])
    set_var('SliceLocation', SliceLocation)
    #set_var('PixelSpacing', [round(value, 5) for value in PixelSpacing])
    set_var('PixelSpacing', PixelSpacing)
    set_var('SliceThickness', SliceThickness)
    set_var('LocationOfNotCTData', LocationOfNotCTData)
    set_var('PHeight', PHeight)
    set_var('PWidth', PWidth)
    set_var('PDepth', PDepth)
    set_var('PatientName', PatientName)
    set_var("CT_MAX_HU", int(ct_max))

    del PixelsGrid

    if RTexist:
        info = pydicom.dcmread(files[RTSTRUCT_location])
        # 获取所有轮廓名称
        contour_names = [
            seq.ROIName for seq in info.StructureSetROISequence
        ]
        TumorContourSelection = select_contour_from_dialog(
            contour_names, "", "Please select tumor contour data from the list.")
        if TumorContourSelection is not None:
            roi_contour = info.ROIContourSequence[TumorContourSelection - 1]
            num_slices = len(roi_contour.ContourSequence)
            labeldata = np.zeros(CT_Shape)

            # CT 与 轮廓数据对齐
            RF = -ImagePositionPatient[0] / PixelSpacing[0]
            CF = -ImagePositionPatient[1] / PixelSpacing[1]
            SF = -min(SliceLocation) / SliceThickness

            for j in range(num_slices):
                contour_data = roi_contour.ContourSequence[j].ContourData
                points = np.array(contour_data).reshape(-1, 3)
                rows = np.round(points[:, 0] / PixelSpacing[0] + RF).astype(int)
                cols = np.round(points[:, 1] / PixelSpacing[1] + CF).astype(int)
                slices = np.round(points[:, 2] / SliceThickness + SF).astype(int)

                for row, col, sl in zip(rows, cols, slices):
                    if 0 <= row < CT_Shape[1] and 0 <= col < CT_Shape[0] and 0 <= sl < CT_Shape[2]:
                        labeldata[col, row, sl] = 255
            labeldata = fillin(labeldata, 0)
            del_var('labeldata')
            set_var('labeldata', labeldata)
            set_var('labelexits', 1)
            msg = "CT data imported and tumor binary 3D data automatically generated!"
            return msg
        else:
            set_var('labelexits', 0)
            msg = "CT data imported but tumor contour was not selected thus no tumor binary 3D data generated."
            return msg
    else:
        set_var('labelexits', 0)
        msg = "CT data imported but no contour data found."
        return msg