import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from ..var.globals import get_var
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
# from ..functions.coordinate_functions import calculate_rotation_matrix_deltaD
from .coordinate_functions import calculate_rotation_matrix_deltaD


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def compute_R(mu_water_E1, mu_water_E2, mu_air_E1, mu_air_E2, a=0.6, b=0.00001):
    """
    计算骨骼在高低能量条件下的衰减比值 R
    参数:
    a: float, 水的贡献因子 (默认 0.6)
    b: float, 空气的贡献因子 (默认 0.4)
    返回:
    R: float, 骨骼的衰减比值
    """
    # 计算骨骼的衰减
    mu_bone_E1 = a * mu_water_E1 + b * mu_air_E1
    mu_bone_E2 = a * mu_water_E2 + b * mu_air_E2
    # 计算 R
    R = mu_bone_E1 / mu_bone_E2
    return R


def suppress_bones(drr1, drr2, mu_water, mu_water2, mu_air, mu_air2, deltaI):
    R = compute_R(mu_water, mu_water2, mu_air, mu_air2)
    #判断哪个是更高的能量
    if deltaI > 0:
        drr = drr1 / (drr2 ** R)
    else:
        drr = drr2 / (drr1 ** R)

    return drr


def rotate_around_z(x, y, z, angle_degrees):
    """优化后的旋转函数"""
    # 将角度从度数转换为弧度
    angle_radians = np.radians(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    # 使用更高效的旋转矩阵计算
    return np.array([
        x * cos_theta - y * sin_theta,
        x * sin_theta + y * cos_theta,
        z
    ])


def get_voxels_on_X_ray_lines(ct_shape, pixel_spacing, slice_thickness, xray_source, imaging_points, iso_origin, OID,
                              SID):
    """优化后的射线体素计算"""
    # 获取最大起始点和终点
    startpoint, endpoint = compute_ct_space_range(
        ct_shape, pixel_spacing, slice_thickness,
        iso_origin[0], iso_origin[1], iso_origin[2],
        xray_source[0], xray_source[1], xray_source[2], OID
    )

    # 调整边界
    startpoint += 200
    endpoint -= 200

    # 计算射线上的点数
    num_points = int(np.ceil((startpoint - endpoint) / pixel_spacing[0]))

    # 预计算z坐标（所有射线共享）
    z = np.linspace(startpoint, endpoint, num_points, endpoint=False)[::-1]
    r_kz = SID - z

    # 向量化计算所有射线的x,y坐标
    # 计算比例因子
    lengths = np.sqrt(imaging_points[:, 0] ** 2 + imaging_points[:, 1] ** 2 + SID ** 2)
    kx = np.abs(imaging_points[:, 0]) / lengths
    ky = np.abs(imaging_points[:, 1]) / lengths

    # 计算符号
    sign_x = np.sign(imaging_points[:, 0])
    sign_y = np.sign(imaging_points[:, 1])

    # 使用广播计算所有射线的x,y坐标
    x = sign_x[:, None] * r_kz[None, :] * kx[:, None]
    y = sign_y[:, None] * r_kz[None, :] * ky[:, None]

    # 创建结果数组 (3, num_points, num_imaging_points)
    VoxelsOnLines = np.zeros((3, num_points, len(imaging_points)))
    VoxelsOnLines[0] = x.T
    VoxelsOnLines[1] = y.T
    VoxelsOnLines[2] = np.tile(z, (len(imaging_points), 1)).T

    return VoxelsOnLines


def get_line(startpoint, endpoint, end_position, pixel_resolution, SID):
    """优化后的射线生成函数"""
    # 计算比例因子
    length_of_line = np.sqrt(end_position[0] ** 2 + end_position[1] ** 2 + SID ** 2)
    kx = abs(end_position[0]) / length_of_line
    ky = abs(end_position[1]) / length_of_line

    # 计算符号
    sign_x = 1 if end_position[0] >= 0 else -1
    sign_y = 1 if end_position[1] >= 0 else -1

    # 生成z坐标
    num_points = int(np.ceil((startpoint - endpoint) / pixel_resolution))
    z = np.linspace(startpoint, endpoint, num_points, endpoint=False)[::-1]
    r_kz = SID - z

    # 计算x,y坐标
    x = sign_x * r_kz * kx
    y = sign_y * r_kz * ky

    return np.array([x, y, z])


def get_imaging_points(X, Y, pixelDistance, resolution, tileSize):
    """优化后的成像点计算"""
    Rc = resolution / 2 - 0.5
    X_L = -Rc * pixelDistance + tileSize * pixelDistance * (X - 1)
    X_R = X_L + (tileSize - 1) * pixelDistance
    Y_L = -Rc * pixelDistance + tileSize * pixelDistance * (Y - 1)
    Y_R = Y_L + (tileSize - 1) * pixelDistance

    # 使用meshgrid替代双重循环
    Imaging_X, Imaging_Y = np.meshgrid(
        np.linspace(X_L, X_R, tileSize),
        np.linspace(Y_L, Y_R, tileSize),
        indexing='ij'
    )

    # 创建结果数组
    ImagingPoints = np.zeros((tileSize ** 2, 3))
    ImagingPoints[:, 0] = Imaging_X.ravel()
    ImagingPoints[:, 1] = Imaging_Y.ravel()

    return ImagingPoints


def transform_voxels_to_global(traversedVoxels, transferMatrix, translationVector):
    """优化后的坐标变换"""
    # 使用einsum进行高效的矩阵乘法
    rotatedVoxels = np.einsum('ij,jkl->ikl', transferMatrix, traversedVoxels)

    # 应用平移
    translatedVoxels = rotatedVoxels - translationVector[:, np.newaxis, np.newaxis]

    return translatedVoxels


def calculate_I1_gpu(voxels_on_lines_ct, CTData, pixelSpacing, sliceThickness, muWater):
    """
    完全向量化的GPU加速版射线衰减计算
    voxels_on_lines_ct: 3D numpy array (3, num_points, num_pixels)
    CTData: CT体素数据 (numpy array)
    """
    # 解决OpenMP冲突
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    I0 = 1200

    # 转换数据到PyTorch张量
    voxels_tensor = torch.from_numpy(voxels_on_lines_ct).float().to(device)

    # 获取CT数据的形状
    CTShape = CTData.shape

    # 直接计算CT索引 (向量化操作)
    # 注意: CTData的形状是(z, x, y)
    # 原代码索引方式: CTData[CTShape[1] - int(point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1]
    # 转换为:
    z_idx = CTShape[1] - 1 - (voxels_tensor[2] / pixelSpacing[1]).round().long()
    x_idx = (voxels_tensor[0] / pixelSpacing[0]).round().long() - 1
    y_idx = (voxels_tensor[1] / sliceThickness).round().long() - 1

    # 创建有效掩码
    valid_mask = (
            (x_idx >= 0) & (x_idx < CTShape[0]) &
            (y_idx >= 0) & (y_idx < CTShape[2]) &
            (z_idx >= 0) & (z_idx < CTShape[1])
    )

    # 将CT数据转移到GPU
    CT_tensor = torch.from_numpy(CTData.astype(np.float32)).to(device)

    # 初始化CT值张量
    ct_vals = torch.zeros_like(x_idx, dtype=torch.float32)

    # 获取有效索引
    valid_indices = valid_mask.nonzero(as_tuple=True)

    # 仅对有效位置查询CT值
    if len(valid_indices[0]) > 0:
        ct_vals[valid_indices] = CT_tensor[
            z_idx[valid_indices],
            x_idx[valid_indices],
            y_idx[valid_indices]
        ]

    # 计算每条射线的平均CT值 (忽略无效点)
    sum_ct = ct_vals.sum(dim=0)  # 沿路径点维度求和 (num_points, num_pixels) -> (num_pixels)
    count_valid = valid_mask.sum(dim=0).float()
    avg_ct = torch.where(count_valid > 0, sum_ct / count_valid, torch.zeros_like(sum_ct))

    # 计算衰减 (向量化操作)
    attenuation = avg_ct * muWater / 1000.0 + muWater

    # 计算最终强度
    I1 = I0 * torch.exp(-attenuation)
    return I1.cpu().numpy()


def calculate_I1(traversedVoxels_ct, num_pixels, CTData, CTShape, muWater, muAir):
    """
    Calculate the X-ray intensity (I1) at the imaging plane using ray attenuation for CT and air.

    Parameters:
    traversedVoxels_ct: 3 x num_voxels x num_drr_pixels numpy array of global coordinates for X-ray traversed voxels.
    CTGlobalCoords: N x 4 numpy array where the first three columns are (x, y, z) coordinates, and the 4th column is the CT value.
    pixelSpacing: List or numpy array of pixel spacing in x and y directions [dx, dy].
    sliceThickness: Thickness of each CT slice in mm.
    SID: Source-to-imaging distance (mm).
    I0: Initial X-ray intensity at the source.
    muWater: X-ray attenuation coefficient for water.
    muAir: X-ray attenuation coefficient for air.
    ct_min: Minimum CT value in the dataset (for normalization to HU range).
    ct_max: Maximum CT value in the dataset (for normalization to HU range).

    Returns:
    I1: 1D numpy array of length num_drr_pixels representing the X-ray intensity at the imaging plane for each pixel.
    """
    I0 = 1200
    # Initialize I1
    I1 = np.ones(num_pixels) * I0  # Start with initial intensity for each pixel
    for pixel_idx in range(num_pixels):
        # Extract the ray path for the current pixel (3 x num_voxels -> num_voxels x 3)
        voxelPath = np.array(traversedVoxels_ct[pixel_idx])  # Shape: (num_voxels_per_ray, 3)
        if voxelPath.size == 0:
            continue
        m = 0
        CTValue = []  # 用于存储与 CT 数据相交的体素值
        for point in voxelPath:
            # 获取 CT 值，注意索引需要取整
            CTValue.append(CTData[CTShape[1] - int(point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1])
            m += 1
        # 如果有多个相交体素
        if m >= 1:
            # 计算 CT 衰减
            CTValue = np.array(CTValue)
            attenuation = np.average(CTValue * (muWater - muAir) / 1000 + muWater)
            I1[pixel_idx] = I0 * np.exp(-attenuation)
    return I1


def calculate_bone_only_I1(traversedVoxels_ct, num_pixels, CTData, CTShape, muWater, muAir, bone_threshold):
    I0 = 1200
    I1 = np.ones(num_pixels) * I0  # Start with initial intensity for each pixel
    for pixel_idx in range(num_pixels):
        # Extract the ray path for the current pixel (3 x num_voxels -> num_voxels x 3)
        voxelPath = np.array(traversedVoxels_ct[pixel_idx])  # Shape: (num_voxels_per_ray, 3)
        if voxelPath.size == 0:
            continue
        m = 0
        CTValue = []  # 用于存储与 CT 数据相交的体素值
        for point in voxelPath:
            # 获取 CT 值，注意索引需要取整
            if CTData[CTShape[1] - int(point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1] >= bone_threshold:
                value = CTData[int(CTShape[1] - point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1]
                CTValue.append(value)
                m += 1

        # 如果有多个相交体素
        if m >= 1:
            # 计算 CT 衰减
            CTValue = np.array(CTValue)
            attenuation = np.average(CTValue * (muWater - muAir) / 1000 + muWater)
            I1[pixel_idx] = I0 * np.exp(-attenuation)
    return I1


def calculate_bone_suppressed_I1_constant(traversedVoxels_ct, num_pixels, CTData, CTShape, muWater, muAir,
                                          bone_threshold, constant):
    """
    Calculate the X-ray intensity (I1) at the imaging plane using ray attenuation for CT and air.

    Parameters:
    traversedVoxels_ct: 3 x num_voxels x num_drr_pixels numpy array of global coordinates for X-ray traversed voxels.
    CTGlobalCoords: N x 4 numpy array where the first three columns are (x, y, z) coordinates, and the 4th column is the CT value.
    pixelSpacing: List or numpy array of pixel spacing in x and y directions [dx, dy].
    sliceThickness: Thickness of each CT slice in mm.
    SID: Source-to-imaging distance (mm).
    I0: Initial X-ray intensity at the source.
    muWater: X-ray attenuation coefficient for water.
    muAir: X-ray attenuation coefficient for air.
    ct_min: Minimum CT value in the dataset (for normalization to HU range).
    ct_max: Maximum CT value in the dataset (for normalization to HU range).

    Returns:
    I1: 1D numpy array of length num_drr_pixels representing the X-ray intensity at the imaging plane for each pixel.
    """
    I0 = 1200
    I1 = np.ones(num_pixels) * I0  # Start with initial intensity for each pixel
    for pixel_idx in range(num_pixels):
        # Extract the ray path for the current pixel (3 x num_voxels -> num_voxels x 3)
        voxelPath = np.array(traversedVoxels_ct[pixel_idx])  # Shape: (num_voxels_per_ray, 3)
        if voxelPath.size == 0:
            I1[pixel_idx] = I0 * np.exp(-muAir)
            continue
        m = 0
        CTValue = []  # 用于存储与 CT 数据相交的体素值
        for point in voxelPath:
            # 获取 CT 值，注意索引需要取整
            value = CTData[int(CTShape[1] - point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1] if CTData[CTShape[
                                                                                                               1] - int(
                point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1] <= bone_threshold else constant
            CTValue.append(value)
            m += 1
        # 如果有多个相交体素
        if m >= 1:
            # 计算 CT 衰减
            CTValue = np.array(CTValue)
            attenuation = np.average(CTValue * (muWater - muAir) / 1000 + muWater)
            I1[pixel_idx] = I0 * np.exp(-attenuation)

    return I1


def calculate_bone_enhanced_I1(traversedVoxels_ct, num_pixels, CTData, CTShape, muWater, muAir,
                                          bone_threshold, enhance_factor):
    """
    Calculate the X-ray intensity (I1) at the imaging plane using ray attenuation for CT and air.

    Parameters:
    traversedVoxels_ct: 3 x num_voxels x num_drr_pixels numpy array of global coordinates for X-ray traversed voxels.
    CTGlobalCoords: N x 4 numpy array where the first three columns are (x, y, z) coordinates, and the 4th column is the CT value.
    pixelSpacing: List or numpy array of pixel spacing in x and y directions [dx, dy].
    sliceThickness: Thickness of each CT slice in mm.
    SID: Source-to-imaging distance (mm).
    I0: Initial X-ray intensity at the source.
    muWater: X-ray attenuation coefficient for water.
    muAir: X-ray attenuation coefficient for air.
    ct_min: Minimum CT value in the dataset (for normalization to HU range).
    ct_max: Maximum CT value in the dataset (for normalization to HU range).

    Returns:
    I1: 1D numpy array of length num_drr_pixels representing the X-ray intensity at the imaging plane for each pixel.
    """
    I0 = 1200
    I1 = np.ones(num_pixels) * I0  # Start with initial intensity for each pixel
    for pixel_idx in range(num_pixels):
        # Extract the ray path for the current pixel (3 x num_voxels -> num_voxels x 3)
        voxelPath = np.array(traversedVoxels_ct[pixel_idx])  # Shape: (num_voxels_per_ray, 3)
        if voxelPath.size == 0:
            I1[pixel_idx] = I0 * np.exp(-muAir)
            continue
        m = 0
        CTValue = []  # 用于存储与 CT 数据相交的体素值
        for point in voxelPath:
            # 获取 CT 值，注意索引需要取整
            value = CTData[int(CTShape[1] - point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1] \
                if CTData[CTShape[1] - int(point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1] <= bone_threshold \
                else enhance_factor * CTData[int(CTShape[1] - point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1]
            CTValue.append(value)
            m += 1
        # 如果有多个相交体素
        if m >= 1:
            # 计算 CT 衰减
            CTValue = np.array(CTValue)
            attenuation = np.average(CTValue * (muWater - muAir) / 1000 + muWater)
            I1[pixel_idx] = I0 * np.exp(-attenuation)
    return I1


def compute_ct_space_range(CTShape, pixelSpacing, sliceThickness, x0, y0, z0, x_rotated, y_rotated, z_rotated, OID):
    origin_CT = np.array([x0, y0, z0])
    source = np.array([x_rotated, y_rotated, z_rotated])
    # CT空间范围
    x_min, x_max = 0, CTShape[0] * pixelSpacing[0]
    y_min, y_max = 0, CTShape[2] * sliceThickness
    z_min, z_max = 0, CTShape[1] * pixelSpacing[1]

    # 方向向量
    direction = origin_CT - source

    # 初始化进入点和退出点的参数 t
    t_min, t_max = -np.inf, np.inf

    # 遍历每个维度，计算进入和退出CT空间的 t 值
    for dim, (dim_min, dim_max) in enumerate([(x_min, x_max), (y_min, y_max), (z_min, z_max)]):
        if direction[dim] != 0:
            t1 = (dim_min - source[dim]) / direction[dim]
            t2 = (dim_max - source[dim]) / direction[dim]
            t_enter = min(t1, t2)
            t_exit = max(t1, t2)

            t_min = max(t_min, t_enter)  # 更新进入点
            t_max = min(t_max, t_exit)  # 更新退出点
        elif source[dim] < dim_min or source[dim] > dim_max:
            # 如果方向向量在该维度上为0且source点不在CT空间内，则线段完全在外部
            return z0 + OID, OID - y_max + z0 * sliceThickness

    # 计算进入点和退出点的坐标
    if t_min > t_max:
        return z0 + OID, OID - y_max + z0 * sliceThickness  # 没有交点，线段完全在外部

    # 进入点和退出点
    p_in = source + t_min * direction
    p_out = source + t_max * direction

    # 计算线段长度
    segment_length = np.linalg.norm(p_in - origin_CT)
    extended_length = np.linalg.norm(p_out - origin_CT)
    startpoint = OID + segment_length + 10
    endpoint = OID - extended_length - 10
    return startpoint, endpoint


# Function to calculate DRR
def getDRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z, sliceThickness,
           save_name, Geoinfo_save_path, muWater, muAir, bitDepth):
    """
    Generates a Digitally Reconstructed Radiograph (DRR) using ray tracing algorithm.

    Parameters:
        iso_x, iso_y, iso_z: Coordinates of the beam center in CT coordinate system (mm).
        IPEL: Side length of the imaging area (mm).
        couchAngle: Couch rotation angle (degrees) 从全局坐标系Z轴往下看顺时针为正.
        resolution: Resolution of the imaging area.
        sliceThickness: Thickness of CT slices (mm).
        tileSize: Size of image blocks for memory optimization.
        transferMatrix, translationVector: Coordinate transformation matrix and vector.
        OID: Object-to-imaging plane distance (mm).
        x, y, z: Position of the virtual X-ray source in global coordinates.
    Returns:
        DRR: 2D numpy array representing the DRR image.
    """
    I0 = 1200
    # CT information
    CTData = get_var("PixelsGrid")
    pixelSpacing = get_var("PixelSpacing")

    # Compute the new X-ray source coordinates after rotating the imaging system
    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    # Prepare CT global coordinates
    CTShape = np.array(CTData.shape)

    # Initialize DRR as a 2D array
    DRR = np.zeros((resolution, resolution))

    # 物理距离 between imaging plane pixels
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    # Block-wise calculation
    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            # Get imaging points for the current block
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)

            # Compute voxel paths
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)

            # Apply global-to-CT coordinate transformation
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)

            # Translation to CT coordinate system
            translation = np.array(
                [(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                 (CTShape[0] - iso_y - 1) * pixelSpacing[1]]
            ).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            # 直接使用向量化的GPU计算
            I1 = calculate_I1_gpu(
                voxels_on_lines_ct,
                CTData,
                pixelSpacing,
                sliceThickness,
                muWater
            )

            # Reshape I1 into a tile of size (tileSize, tileSize)
            I1_tile = I1.reshape((tileSize, tileSize), order='F')

            # Fill the corresponding block in the DRR
            x_start = (X - 1) * tileSize
            x_end = X * tileSize
            y_start = (Y - 1) * tileSize
            y_end = Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    # Save DRR as an image
    # 归一化
    DRR = (2 ** bitDepth) * (DRR - np.min(DRR)) / 200
    DRR_flipped = cv2.flip(DRR, 0)
    DRR_normalized = DRR_flipped.astype(np.uint8)
    imageName = f"{Geoinfo_save_path}/saved DRR/DRR_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_BitDepth_{bitDepth}_μwater{muWater}_μair{muAir}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray', vmin=0, vmax=2 ** bitDepth)
    return DRR_normalized


def get_bone_only_DRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z,
                      sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater, muAir, bitDepth):
    """
    Generates a Digitally Reconstructed Radiograph (DRR) using ray tracing algorithm.

    Parameters:
        iso_x, iso_y, iso_z: Coordinates of the beam center in CT coordinate system (mm).
        IPEL: Side length of the imaging area (mm).
        couchAngle: Couch rotation angle (degrees) 从全局坐标系Z轴往下看顺时针为正.
        resolution: Resolution of the imaging area.
        sliceThickness: Thickness of CT slices (mm).
        tileSize: Size of image blocks for memory optimization.
        transferMatrix, translationVector: Coordinate transformation matrix and vector.
        OID: Object-to-imaging plane distance (mm).
        x, y, z: Position of the virtual X-ray source in global coordinates.
    Returns:
        DRR: 2D numpy array representing the DRR image.
    """
    I0 = 1200
    # CT information
    CTData = get_var("PixelsGrid")
    pixelSpacing = get_var("PixelSpacing")

    # Compute the new X-ray source coordinates after rotating the imaging system
    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    # Prepare CT global coordinates
    CTShape = np.array(CTData.shape)

    # Initialize DRR as a 2D array
    DRR = np.zeros((resolution, resolution))

    # Physical distance between imaging plane pixels
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    x_max = CTShape[0]
    y_max = CTShape[2]
    z_max = CTShape[1]
    # Block-wise calculation
    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            # Get imaging points for the current block
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)

            # Compute voxel paths and intensities
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)
            # Apply global-to-CT coordinate transformation
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)

            # Translation to CT coordinate system
            translation = np.array(
                [(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                 (CTShape[0] - iso_y - 1) * pixelSpacing[1]]
            ).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            # Discretize the coordinates to CT grid
            voxels_on_lines_ct[0, :, :] = np.round(voxels_on_lines_ct[0, :, :] / pixelSpacing[0])
            voxels_on_lines_ct[1, :, :] = np.round(voxels_on_lines_ct[1, :, :] / sliceThickness)
            voxels_on_lines_ct[2, :, :] = np.round(voxels_on_lines_ct[2, :, :] / pixelSpacing[1])

            # Initialize the output as a 3D list
            num_pixels = voxels_on_lines_ct.shape[2]
            voxels_on_lines_ct_list = [[] for _ in range(num_pixels)]  # Create a list for each pixel

            # Process each pixel separately
            for pixel_idx in range(num_pixels):
                # Extract the voxel path for this pixel
                voxel_path = voxels_on_lines_ct[:, :, pixel_idx].T  # Shape: (num_voxels, 3)

                # Remove points outside the valid range
                valid_mask = (
                        (voxel_path[:, 0] >= 0) & (voxel_path[:, 0] <= x_max - 1) &
                        (voxel_path[:, 1] >= 0) & (voxel_path[:, 1] <= y_max - 1) &  # 注意：z 对应 CTShape 的第 2 轴
                        (voxel_path[:, 2] >= 0) & (voxel_path[:, 2] <= z_max - 1)
                )
                voxel_path = voxel_path[valid_mask]  # 保留有效点

                # Convert to a list and store it
                voxels_on_lines_ct_list[pixel_idx] = voxel_path.tolist()

            I1 = calculate_bone_only_I1(voxels_on_lines_ct_list, num_pixels, CTData, CTShape, muWater, muAir,
                                        bone_threshold)

            # Reshape I1 into a tile of size (tileSize, tileSize)
            I1_tile = I1.reshape((tileSize, tileSize), order='F')

            # Fill the corresponding block in the DRR
            x_start = (X - 1) * tileSize
            x_end = X * tileSize
            y_start = (Y - 1) * tileSize
            y_end = Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    # Save DRR as an image
    # 归一化
    DRR = (2 ** bitDepth) * (DRR - np.min(DRR)) / 200
    DRR_flipped = cv2.flip(DRR, 0)
    DRR_normalized = DRR_flipped.astype(np.uint8)
    imageName = f"{Geoinfo_save_path}/saved DRR/bone_only_DRR_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_BitDepth_{bitDepth}_μwater{muWater}_μair{muAir}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray')
    return DRR_normalized


def get_bone_suppressed_DRR_constant(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                     iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater,
                                     muAir, bitDepth, constant):
    """
    Generates a Digitally Reconstructed Radiograph (DRR) using ray tracing algorithm.

    Parameters:
        iso_x, iso_y, iso_z: Coordinates of the beam center in CT coordinate system (mm).
        IPEL: Side length of the imaging area (mm).
        couchAngle: Couch rotation angle (degrees) 从全局坐标系Z轴往下看顺时针为正.
        resolution: Resolution of the imaging area.
        sliceThickness: Thickness of CT slices (mm).
        tileSize: Size of image blocks for memory optimization.
        transferMatrix, translationVector: Coordinate transformation matrix and vector.
        OID: Object-to-imaging plane distance (mm).
        x, y, z: Position of the virtual X-ray source in global coordinates.
    Returns:
        DRR: 2D numpy array representing the DRR image.
    """
    I0 = 1200
    # CT information
    CTData = get_var("PixelsGrid")
    pixelSpacing = get_var("PixelSpacing")

    # Compute the new X-ray source coordinates after rotating the imaging system
    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    # Prepare CT global coordinates
    CTShape = np.array(CTData.shape)

    # Initialize DRR as a 2D array
    DRR = np.zeros((resolution, resolution))

    # Physical distance between imaging plane pixels
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    x_max = CTShape[0]
    y_max = CTShape[2]
    z_max = CTShape[1]

    # Block-wise calculation
    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            # Get imaging points for the current block
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)

            # Compute voxel paths and intensities
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)
            # Apply global-to-CT coordinate transformation
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)

            # Translation to CT coordinate system
            translation = np.array(
                [(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                 (CTShape[0] - iso_y - 1) * pixelSpacing[1]]
            ).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            # Discretize the coordinates to CT grid
            voxels_on_lines_ct[0, :, :] = np.round(voxels_on_lines_ct[0, :, :] / pixelSpacing[0])
            voxels_on_lines_ct[1, :, :] = np.round(voxels_on_lines_ct[1, :, :] / sliceThickness)
            voxels_on_lines_ct[2, :, :] = np.round(voxels_on_lines_ct[2, :, :] / pixelSpacing[1])

            # Initialize the output as a 3D list
            num_pixels = voxels_on_lines_ct.shape[2]
            voxels_on_lines_ct_list = [[] for _ in range(num_pixels)]  # Create a list for each pixel

            # Process each pixel separately
            for pixel_idx in range(num_pixels):
                # Extract the voxel path for this pixel
                voxel_path = voxels_on_lines_ct[:, :, pixel_idx].T  # Shape: (num_voxels, 3)

                # Remove points outside the valid range
                valid_mask = (
                        (voxel_path[:, 0] >= 0) & (voxel_path[:, 0] <= x_max - 1) &
                        (voxel_path[:, 1] >= 0) & (voxel_path[:, 1] <= y_max - 1) &  # 注意：z 对应 CTShape 的第 2 轴
                        (voxel_path[:, 2] >= 0) & (voxel_path[:, 2] <= z_max - 1)
                )
                voxel_path = voxel_path[valid_mask]  # 保留有效点

                # Convert to a list and store it
                voxels_on_lines_ct_list[pixel_idx] = voxel_path.tolist()

            I1 = calculate_bone_suppressed_I1_constant(voxels_on_lines_ct_list, num_pixels, CTData, CTShape, muWater,
                                                       muAir, bone_threshold, constant)

            # Reshape I1 into a tile of size (tileSize, tileSize)
            I1_tile = I1.reshape((tileSize, tileSize), order='F')

            # Fill the corresponding block in the DRR
            x_start = (X - 1) * tileSize
            x_end = X * tileSize
            y_start = (Y - 1) * tileSize
            y_end = Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    # Save DRR as an image

    # 归一化
    DRR = (2 ** bitDepth) * (DRR - np.min(DRR)) / 200
    DRR_flipped = cv2.flip(DRR, 0)
    DRR_normalized = DRR_flipped.astype(np.uint8)
    imageName = f"{Geoinfo_save_path}/saved DRR/bone_suppressed_DRR_constant{constant}_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_BitDepth_{bitDepth}_μwater{muWater}_μair{muAir}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray', vmin=0, vmax=2 ** bitDepth)
    return DRR_normalized


def get_bone_enhanced_DRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y,
                                     iso_z, sliceThickness, save_name, Geoinfo_save_path, bone_threshold, muWater,
                                     muAir, bitDepth, enhance_factor):
    """
    Generates a Digitally Reconstructed Radiograph (DRR) using ray tracing algorithm.

    Parameters:
        iso_x, iso_y, iso_z: Coordinates of the beam center in CT coordinate system (mm).
        IPEL: Side length of the imaging area (mm).
        couchAngle: Couch rotation angle (degrees) 从全局坐标系Z轴往下看顺时针为正.
        resolution: Resolution of the imaging area.
        sliceThickness: Thickness of CT slices (mm).
        tileSize: Size of image blocks for memory optimization.
        transferMatrix, translationVector: Coordinate transformation matrix and vector.
        OID: Object-to-imaging plane distance (mm).
        x, y, z: Position of the virtual X-ray source in global coordinates.
    Returns:
        DRR: 2D numpy array representing the DRR image.
    """
    I0 = 1200
    # CT information
    CTData = get_var("PixelsGrid")
    pixelSpacing = get_var("PixelSpacing")

    # Compute the new X-ray source coordinates after rotating the imaging system
    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    # Prepare CT global coordinates
    CTShape = np.array(CTData.shape)

    # Initialize DRR as a 2D array
    DRR = np.zeros((resolution, resolution))

    # Physical distance between imaging plane pixels
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    x_max = CTShape[0]
    y_max = CTShape[2]
    z_max = CTShape[1]

    # Block-wise calculation
    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            # Get imaging points for the current block
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)

            # Compute voxel paths and intensities
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)
            # Apply global-to-CT coordinate transformation
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)

            # Translation to CT coordinate system
            translation = np.array(
                [(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                 (CTShape[0] - iso_y - 1) * pixelSpacing[1]]
            ).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            # Discretize the coordinates to CT grid
            voxels_on_lines_ct[0, :, :] = np.round(voxels_on_lines_ct[0, :, :] / pixelSpacing[0])
            voxels_on_lines_ct[1, :, :] = np.round(voxels_on_lines_ct[1, :, :] / sliceThickness)
            voxels_on_lines_ct[2, :, :] = np.round(voxels_on_lines_ct[2, :, :] / pixelSpacing[1])

            # Initialize the output as a 3D list
            num_pixels = voxels_on_lines_ct.shape[2]
            voxels_on_lines_ct_list = [[] for _ in range(num_pixels)]  # Create a list for each pixel

            # Process each pixel separately
            for pixel_idx in range(num_pixels):
                # Extract the voxel path for this pixel
                voxel_path = voxels_on_lines_ct[:, :, pixel_idx].T  # Shape: (num_voxels, 3)

                # Remove points outside the valid range
                valid_mask = (
                        (voxel_path[:, 0] >= 0) & (voxel_path[:, 0] <= x_max - 1) &
                        (voxel_path[:, 1] >= 0) & (voxel_path[:, 1] <= y_max - 1) &  # 注意：z 对应 CTShape 的第 2 轴
                        (voxel_path[:, 2] >= 0) & (voxel_path[:, 2] <= z_max - 1)
                )
                voxel_path = voxel_path[valid_mask]  # 保留有效点

                # Convert to a list and store it
                voxels_on_lines_ct_list[pixel_idx] = voxel_path.tolist()

            I1 = calculate_bone_enhanced_I1(voxels_on_lines_ct_list, num_pixels, CTData, CTShape, muWater, muAir,
                                            bone_threshold, enhance_factor)

            # Reshape I1 into a tile of size (tileSize, tileSize)
            I1_tile = I1.reshape((tileSize, tileSize), order='F')

            # Fill the corresponding block in the DRR
            x_start = (X - 1) * tileSize
            x_end = X * tileSize
            y_start = (Y - 1) * tileSize
            y_end = Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    # Save DRR as an image
    # 归一化
    DRR = (2 ** bitDepth) * (DRR - np.min(DRR)) / 200
    DRR_flipped = cv2.flip(DRR, 0)
    DRR_normalized = DRR_flipped.astype(np.uint8)
    imageName = f"{Geoinfo_save_path}/saved DRR/bone_enhanced_DRR_enhanceFactor{enhance_factor}_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_BitDepth_{bitDepth}_μwater{muWater}_μair{muAir}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray', vmin=0, vmax=2 ** bitDepth)
    return DRR_normalized


def find_closest_energy(mu, material_data):
    """Find the energy (in keV) with the closest attenuation coefficient to the given mu."""
    energies = list(material_data.keys())
    mu_values = list(material_data.values())
    idx = np.argmin(np.abs(np.array(mu_values) - mu))
    return energies[idx]


def get_mu(material, energy, interpolated_data):
    """Interpolate the attenuation coefficient for a given material and energy."""
    energies = np.array(list(interpolated_data[material].keys()))
    mu_values = np.array(list(interpolated_data[material].values()))
    return np.interp(energy, energies, mu_values)


def compute_DRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z, sliceThickness, muWater, muAir):
    """Compute the raw DRR intensity array without normalization or saving."""
    I0 = 1200
    CTData = get_var("PixelsGrid")
    pixelSpacing = get_var("PixelSpacing")

    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    CTShape = np.array(CTData.shape)
    DRR = np.zeros((resolution, resolution))
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    x_max = CTShape[0]
    y_max = CTShape[2]
    z_max = CTShape[1]

    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)
            translation = np.array([(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                                    (CTShape[0] - iso_y - 1) * pixelSpacing[1]]).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            voxels_on_lines_ct[0, :, :] = np.round(voxels_on_lines_ct[0, :, :] / pixelSpacing[0])
            voxels_on_lines_ct[1, :, :] = np.round(voxels_on_lines_ct[1, :, :] / sliceThickness)
            voxels_on_lines_ct[2, :, :] = np.round(voxels_on_lines_ct[2, :, :] / pixelSpacing[1])

            num_pixels = voxels_on_lines_ct.shape[2]
            voxels_on_lines_ct_list = [[] for _ in range(num_pixels)]

            for pixel_idx in range(num_pixels):
                voxel_path = voxels_on_lines_ct[:, :, pixel_idx].T
                valid_mask = ((voxel_path[:, 0] >= 0) & (voxel_path[:, 0] <= x_max - 1) &
                              (voxel_path[:, 1] >= 0) & (voxel_path[:, 1] <= y_max - 1) &
                              (voxel_path[:, 2] >= 0) & (voxel_path[:, 2] <= z_max - 1))
                voxel_path = voxel_path[valid_mask]
                voxels_on_lines_ct_list[pixel_idx] = voxel_path.tolist()

            I1 = calculate_I1(voxels_on_lines_ct_list, num_pixels, CTData, CTShape, muWater, muAir)
            I1_tile = I1.reshape((tileSize, tileSize), order='F')
            x_start, x_end = (X - 1) * tileSize, X * tileSize
            y_start, y_end = (Y - 1) * tileSize, Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    return DRR


def get_bone_suppressed_DRR_dual_energy(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x,
                                        iso_y, iso_z, sliceThickness, save_name, Geoinfo_save_path, muWater, muAir,
                                        bitDepth, deltaI, dictionary):
    """
        Generates a bone-suppressed DRR using the dual-energy method.

        Parameters:
            x_tube, y_tube, z_tube: X-ray source position in global coordinates (mm).
            IPEL: Side length of the imaging area (mm).
            OID: Object-to-imaging plane distance (mm).
            resolution: Resolution of the imaging area (e.g., 1000 for 1000x1000).
            tileSize: Size of image blocks for memory optimization.
            couchAngle: Couch rotation angle (degrees).
            iso_x, iso_y, iso_z: Beam center coordinates in CT coordinate system (mm).
            sliceThickness: Thickness of CT slices (mm).
            save_name: Base name for the saved image file.
            Geoinfo_save_path: Directory path to save the DRR image.
            I0: Initial X-ray intensity at the source.
            muWater: Attenuation coefficient for water at lower energy (cm²/g).
            muAir: Attenuation coefficient for air at lower energy (cm²/g).
            bitDepth: Bit depth for the output image (e.g., 8 for 0-255 range).
            deltaI: Energy difference between the two X-ray energies (keV).
            dictionary: Interpolated data with attenuation coefficients for water and air.

        Returns:
            DRR_flipped: 2D numpy array of the bone-suppressed DRR image.
        """
    I0 = 1200
    # Step 2: Calculate E2
    E2 = I0 + deltaI

    # Step 3: Get attenuation coefficients for E2
    muWater2 = get_mu("water", E2, dictionary)
    muAir2 = get_mu("air", E2, dictionary)

    # Step 4: Generate DRR at E1 (lower energy)
    DRR1 = compute_DRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z,
                       sliceThickness, muWater, muAir)

    # Step 5: Generate DRR at E2 (higher energy)
    DRR2 = compute_DRR(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z,
                       sliceThickness, muWater2, muAir2)

    # Step 6: Suppress bones using dual-energy method
    DRR_suppressed = suppress_bones(DRR1, DRR2, muWater, muWater2, muAir, muAir2, deltaI)

    # Step 7: Normalize and save the image
    WA = I0 * np.exp(- muWater - muAir)
    DRR_normalized = (2 ** bitDepth) * (DRR_suppressed - WA) / (I0 - WA)
    DRR_normalized = np.clip(DRR_normalized, 0, 2 ** bitDepth - 1)  # Ensure values stay within bit depth range
    DRR_normalized = DRR_normalized.astype(np.uint8)
    DRR_flipped = cv2.flip(DRR_normalized, 0)

    imageName = f"{Geoinfo_save_path}/saved DRR/bone_suppressed_DRR_dual_energy_deltaI{deltaI}_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_BitDepth_{bitDepth}_μwater{muWater}_μair{muAir}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray', vmin=0, vmax=2 ** bitDepth)

    return DRR_flipped


def calculate_Label(voxels_on_lines_ct_list, num_pixels, Data3D, CTShape, threshold):
    I = np.zeros(num_pixels, dtype=np.uint8)  # Initialize output array with zeros

    for pixel_idx in range(num_pixels):
        voxel_path = np.array(voxels_on_lines_ct_list[pixel_idx])  # Shape: (num_voxels, 3)
        if voxel_path.size == 0:  # Skip empty paths
            continue

        count = 0  # Count of labeled voxels
        for point in voxel_path:
            x, y, z = CTShape[1] - int(point[2]) - 1, int(point[0]) - 1, int(point[1]) - 1
            if Data3D[x, y, z] > 0:  # Assuming Data3D contains positive labels for tumors
                count += 1
                if count >= threshold:  # Early exit if threshold is exceeded
                    I[pixel_idx] = 255
                    break
    return I


def getLabel(x_tube, y_tube, z_tube, IPEL, OID, resolution, tileSize, couchAngle, iso_x, iso_y, iso_z, sliceThickness,
           save_name, Geoinfo_save_path, threshold):

    # CT information
    Data3D = get_var("labeldata")
    pixelSpacing = get_var("PixelSpacing")

    # Compute the new X-ray source coordinates after rotating the imaging system
    [x_rotated, y_rotated, z_rotated] = rotate_around_z(x_tube, y_tube, z_tube, couchAngle)
    transferMatrix, translationVector = calculate_rotation_matrix_deltaD(x_rotated, y_rotated, z_rotated, OID)
    SID = OID + (x_rotated ** 2 + y_rotated ** 2 + z_rotated ** 2) ** 0.5

    # Prepare CT global coordinates
    CTShape = np.array(Data3D.shape)

    # Initialize DRR as a 2D array
    DRR = np.zeros((resolution, resolution))

    # Physical distance between imaging plane pixels
    pixelDistance = IPEL / resolution
    sectionNum = int(resolution / tileSize)

    x_max = CTShape[0]
    y_max = CTShape[2]
    z_max = CTShape[1]
    # Block-wise calculation
    for X in range(1, sectionNum + 1):
        for Y in range(1, sectionNum + 1):
            # Get imaging points for the current block
            imagingPoints = get_imaging_points(X, Y, pixelDistance, resolution, tileSize)

            # Compute voxel paths and intensities
            voxels_on_lines = get_voxels_on_X_ray_lines(CTShape, pixelSpacing, sliceThickness,
                                                        [x_rotated, y_rotated, z_rotated], imagingPoints,
                                                        [iso_x, iso_y, iso_z], OID, SID)

            # Apply global-to-CT coordinate transformation
            voxels_on_lines_global = transform_voxels_to_global(voxels_on_lines, transferMatrix, translationVector)

            # Translation to CT coordinate system
            translation = np.array(
                [(iso_x - 1) * pixelSpacing[1], (iso_z - 1) * sliceThickness,
                 (CTShape[0] - iso_y - 1) * pixelSpacing[1]]
            ).reshape(3, 1, 1)
            voxels_on_lines_ct = voxels_on_lines_global + translation

            # Discretize the coordinates to CT grid
            voxels_on_lines_ct[0, :, :] = np.round(voxels_on_lines_ct[0, :, :] / pixelSpacing[0])
            voxels_on_lines_ct[1, :, :] = np.round(voxels_on_lines_ct[1, :, :] / sliceThickness)
            voxels_on_lines_ct[2, :, :] = np.round(voxels_on_lines_ct[2, :, :] / pixelSpacing[1])

            # Initialize the output as a 3D list
            num_pixels = voxels_on_lines_ct.shape[2]
            voxels_on_lines_ct_list = [[] for _ in range(num_pixels)]  # Create a list for each pixel

            # Process each pixel separately
            for pixel_idx in range(num_pixels):
                # Extract the voxel path for this pixel
                voxel_path = voxels_on_lines_ct[:, :, pixel_idx].T  # Shape: (num_voxels, 3)

                # Remove points outside the valid range
                valid_mask = (
                        (voxel_path[:, 0] >= 0) & (voxel_path[:, 0] <= x_max - 1) &
                        (voxel_path[:, 1] >= 0) & (voxel_path[:, 1] <= y_max - 1) &  # 注意：z 对应 CTShape 的第 2 轴
                        (voxel_path[:, 2] >= 0) & (voxel_path[:, 2] <= z_max - 1)
                )
                voxel_path = voxel_path[valid_mask]  # 保留有效点

                # Convert to a list and store it
                voxels_on_lines_ct_list[pixel_idx] = voxel_path.tolist()

            I = calculate_Label(voxels_on_lines_ct_list, num_pixels, Data3D, CTShape, threshold)

            # Reshape I1 into a tile of size (tileSize, tileSize)
            I1_tile = I.reshape((tileSize, tileSize), order='F')

            # Fill the corresponding block in the DRR
            x_start = (X - 1) * tileSize
            x_end = X * tileSize
            y_start = (Y - 1) * tileSize
            y_end = Y * tileSize
            DRR[y_start:y_end, x_start:x_end] = I1_tile

    # Save DRR as an image
    # 归一化
    DRR_normalized = DRR.astype(np.uint8)
    DRR_flipped = cv2.flip(DRR_normalized, 0)
    imageName = f"{Geoinfo_save_path}/saved DRR/Label_Threshold_{threshold}_{resolution}x{resolution}_CouchAngle{couchAngle}_iso_{iso_x}_{iso_y}_{iso_z}_{save_name}.png"
    plt.imsave(imageName, DRR_flipped, cmap='gray')
    return DRR_flipped