import numpy as np
import math


def switch_coordinate(x0, y0, z0, dicip):
    # Step 1: Calculate Ol point coordinates in global coordinate system
    vector_to_origin = np.array([-x0, -y0, -z0])  # Vector from p0 to O (origin)
    vector_to_origin_normalized = vector_to_origin / np.linalg.norm(vector_to_origin)  # Unit vector
    Ol = dicip * vector_to_origin_normalized  # Ol point coordinates

    # Step 2: Calculate Angle_X
    # Z-axis of local coordinate system is from Ol to p0
    z_local = np.array([x0, y0, z0]) - Ol
    z_local_normalized = z_local / np.linalg.norm(z_local)

    # Project Z_local onto global Z-axis
    z_global = np.array([0, 0, 1])
    cos_angle_x = np.dot(z_local_normalized, z_global)  # Cosine of Angle_X
    angle_x = np.arccos(cos_angle_x)  # Radians
    angle_x_degrees = np.degrees(angle_x)  # Convert to degrees

    # Step 3: Calculate Angle_Z
    # X-axis of local coordinate system is parallel to global XY-plane and perpendicular to Z_local
    x_local = np.cross(np.array([0, 0, 1]), z_local_normalized)  # Cross product to find X-axis direction
    x_local_normalized = x_local / np.linalg.norm(x_local)

    # Project X_local onto global X-axis to find Angle_Z
    x_global = np.array([1, 0, 0])
    cos_angle_z = np.dot(x_local_normalized, x_global)  # Cosine of Angle_Z
    angle_z = np.arccos(cos_angle_z)  # Radians

    # Determine the direction of rotation for Angle_Z
    if x_local_normalized[1] < 0:  # Check Y-component of x_local_normalized
        angle_z = -angle_z

    angle_z_degrees = np.degrees(angle_z)  # Convert to degrees

    # Step 4: Calculate translations (dx, dy, dz)
    dx, dy, dz = -Ol[0], -Ol[1], -Ol[2]

    return angle_x_degrees, angle_z_degrees, (dx, dy, dz)


def local_to_global(xl, yl, zl, p0, dicip):
    """
    将局部坐标系中的点 (xl, yl, zl) 转换为全局坐标系中的点 (xg, yg, zg)

    参数：
        xl, yl, zl: 局部坐标系中点的坐标
        p0: 局部坐标系的起始点 (x0, y0, z0)，即 p0 点
        dicip: 从 p0 点出发到局部坐标系原点 Ol 的延长长度

    返回：
        xg, yg, zg: 转换到全局坐标系中的点的坐标
    """
    # 全局坐标系原点
    O = np.array([0, 0, 0])
    p0 = np.array(p0)

    # 计算 Ol 点坐标
    direction = (O - p0) / np.linalg.norm(O - p0)  # 单位方向向量
    Ol = direction * dicip

    # 局部坐标系的 Z 轴方向（从 Ol 指向 p0）
    z_axis_local = (p0 - Ol) / np.linalg.norm(p0 - Ol)

    # 局部坐标系的 X 轴方向（平行于全局 XY 平面）
    x_axis_local = np.cross([0, 0, 1], z_axis_local)
    x_axis_local /= np.linalg.norm(x_axis_local)

    # 局部坐标系的 Y 轴方向（通过叉乘确定）
    y_axis_local = np.cross(z_axis_local, x_axis_local)

    # 构建局部坐标系到全局坐标系的旋转矩阵
    rotation_matrix = np.column_stack((x_axis_local, y_axis_local, z_axis_local))

    # 将局部坐标转换为全局坐标
    local_point = np.array([xl, yl, zl])
    global_point = rotation_matrix @ local_point + Ol

    return tuple(global_point)


def calculate_rotation_matrix_deltaD(x, y, z, dicip, smooth=1e-6):
    """
        计算将成像坐标系中的X球管所在的点 (x, y, z) 转换为全局坐标系所需的变换矩阵与平移向量

        参数：
            x, y, z: 全局坐标系中X球管的坐标
            dicip: ISO中心点与成像平面中心点的三维空间距离
        返回：
            rotation_matrix: 变换矩阵
            Ol：平移向量
        """
    # 全局坐标系原点
    O = np.array([0, 0, 0])
    if x == 0:
        x = smooth
    if y == 0:
        y = smooth
    if z == 0:
        z = smooth
    p0 = np.array([x, y, z])
    # 计算 Ol 点坐标
    direction = (O - p0) / np.linalg.norm(O - p0)  # 单位方向向量
    Ol = direction * dicip

    # 局部坐标系的 Z 轴方向（从 Ol 指向 p0）
    z_axis_local = (p0 - Ol) / np.linalg.norm(p0 - Ol)

    # 局部坐标系的 X 轴方向（平行于全局 XY 平面）
    x_axis_local = np.cross([0, 0, 1], z_axis_local)
    x_axis_local /= np.linalg.norm(x_axis_local)

    # 局部坐标系的 Y 轴方向（通过叉乘确定）
    y_axis_local = np.cross(z_axis_local, x_axis_local)

    # 构建局部坐标系到全局坐标系的旋转矩阵
    rotation_matrix = np.column_stack((x_axis_local, y_axis_local, z_axis_local))

    return rotation_matrix, -Ol
