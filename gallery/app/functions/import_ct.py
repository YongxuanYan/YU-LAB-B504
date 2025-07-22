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
    if len(files) == 0:
        return "No CT data found! Please make sure *.dcm files are included."

    # Separate CT and RTSTRUCT files
    ct_files = []  # Store (file_path, z_position)
    rt_file = None

    # First pass: Collect CT files and their z-positions
    for file in files:
        ds = pydicom.dcmread(file, stop_before_pixels=True)  # Read metadata only for speed
        if ds.Modality == 'CT':
            # Get z-position from ImagePositionPatient
            z_pos = float(ds.ImagePositionPatient[2])
            ct_files.append((file, z_pos))
        elif ds.Modality == 'RTSTRUCT':
            rt_file = file

    if not ct_files:
        return "No valid CT slices found."

    # Sort CT files by z-position (ascending order - from feet to head)
    ct_files.sort(key=lambda x: x[1])
    sorted_ct_files = [item[0] for item in ct_files]
    z_positions = [item[1] for item in ct_files]

    # Calculate slice thickness from first and last slices
    if len(z_positions) > 1:
        avg_slice_thickness = abs(z_positions[1] - z_positions[0])
    else:
        avg_slice_thickness = 0  # Will be set from DICOM header later

    # Initialize arrays
    num_ct = len(ct_files)
    PixelsGrid = np.zeros((512, 512, num_ct), dtype=np.float32)
    SliceLocation = np.zeros(num_ct, dtype=np.float32)
    LocationOfNotCTData = []
    NotCTNum = 0
    RTexist = rt_file is not None
    labeldata = None

    # Process CT files in sorted order
    info_collected = False
    PixelSpacing = None
    SliceThickness = None
    ImagePositionPatient = None
    PatientName = None

    for i, (file, z_pos) in enumerate(ct_files):
        info = pydicom.dcmread(file)
        SliceLocation[i] = z_pos

        if not info_collected:
            PixelSpacing = np.array(info.PixelSpacing)
            SliceThickness = float(info.SliceThickness) if avg_slice_thickness == 0 else avg_slice_thickness
            ImagePositionPatient = np.array(info.ImagePositionPatient)
            PatientName = info.PatientName.family_name

            # Get initial HU values for windowing
            pixel_array = info.pixel_array
            intercept = info.RescaleIntercept if 'RescaleIntercept' in info else 0
            slope = info.RescaleSlope if 'RescaleSlope' in info else 1
            hu_array = pixel_array * slope + intercept
            min_window_level = hu_array.min()
            max_window_level = hu_array.max()
            max_window_width = max_window_level - min_window_level
            info_collected = True

        # Convert to Hounsfield Units
        pixel_array = info.pixel_array.astype(np.float32)
        intercept = float(getattr(info, 'RescaleIntercept', 0))
        slope = float(getattr(info, 'RescaleSlope', 1))
        hu_array = pixel_array * slope + intercept
        PixelsGrid[:, :, i] = hu_array

    # Adjust CT values (air = -1000)
    PixelsGrid = PixelsGrid - PixelsGrid.min() - 1000
    ct_max = PixelsGrid.max()
    CT_Shape = PixelsGrid.shape

    # Calculate physical dimensions
    PHeight = PixelSpacing[1] * CT_Shape[0]
    PWidth = PixelSpacing[0] * CT_Shape[1]
    PDepth = SliceThickness * CT_Shape[2]

    # Set global variables
    set_var('MaxWindowWidth', max_window_width)
    set_var('MaxWindowLevel', max_window_level)
    set_var('MinWindowLevel', min_window_level)
    del_var('PixelsGrid')
    set_var('PixelsGrid', PixelsGrid)
    set_var('ctexist', 1)
    set_var('SliceNum', CT_Shape[2])
    set_var('SliceLocation', SliceLocation)
    set_var('PixelSpacing', PixelSpacing)
    set_var('SliceThickness', SliceThickness)
    set_var('LocationOfNotCTData', LocationOfNotCTData)
    set_var('PHeight', PHeight)
    set_var('PWidth', PWidth)
    set_var('PDepth', PDepth)
    set_var('PatientName', PatientName)
    set_var("CT_MAX_HU", int(ct_max))

    del PixelsGrid

    # Process RTSTRUCT if available
    if RTexist:
        info = pydicom.dcmread(rt_file)
        contour_names = [seq.ROIName for seq in info.StructureSetROISequence]

        TumorContourSelection = select_contour_from_dialog(
            contour_names, "", "Please select tumor contour data from the list.")

        if TumorContourSelection is not None:
            roi_contour = info.ROIContourSequence[TumorContourSelection - 1]
            num_slices = len(roi_contour.ContourSequence)
            labeldata = np.zeros(CT_Shape)

            # Calculate conversion factors
            min_z = min(z_positions)
            RF = -ImagePositionPatient[0] / PixelSpacing[0]
            CF = -ImagePositionPatient[1] / PixelSpacing[1]

            # Calculate slice index conversion factor
            if len(z_positions) > 1:
                z_range = max(z_positions) - min_z
                slice_conversion = (len(z_positions) - 1) / z_range
            else:
                slice_conversion = 1 / SliceThickness  # Fallback if only one slice

            for j in range(num_slices):
                contour_data = roi_contour.ContourSequence[j].ContourData
                points = np.array(contour_data).reshape(-1, 3)

                # Convert DICOM coordinates to pixel coordinates
                rows = np.round(points[:, 0] / PixelSpacing[0] + RF).astype(int)
                cols = np.round(points[:, 1] / PixelSpacing[1] + CF).astype(int)

                # Convert z-coordinate to slice index
                z_coords = points[:, 2]
                slices = np.round((z_coords - min_z) * slice_conversion).astype(int)

                # Clip to valid range
                slices = np.clip(slices, 0, CT_Shape[2] - 1)

                for row, col, sl in zip(rows, cols, slices):
                    if 0 <= row < CT_Shape[1] and 0 <= col < CT_Shape[0]:
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
