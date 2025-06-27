import numpy as np
import math


def get_X_ray_cross_ct_range(CTShape, pixelSpacing, sliceThickness, x0, y0, z0, x_rotated, y_rotated, z_rotated, OID):
    # 计算 CT 空间的全局坐标范围
    x_min, x_max = 0, CTShape[0] * pixelSpacing[0]
    y_min, y_max = 0, CTShape[2] * sliceThickness
    z_min, z_max = 0, CTShape[1] * pixelSpacing[1]

    # 全局坐标系原点在 CT 坐标系中的位置
    origin_CT = np.array([x0 * pixelSpacing[0], z0 * sliceThickness, y0 * pixelSpacing[1]])

    # 射线源点与全局原点形成的方向向量
    source = np.array([x_rotated + origin_CT[0], y_rotated + origin_CT[0], z_rotated + origin_CT[0]])
    direction = origin_CT - source

    # 确保方向向量为单位向量
    direction = direction / np.linalg.norm(direction)

    if 0 <= source[0] <= x_max and 0 <= source[1] <= y_max and 0 <= source[2] <= z_max:
        # 此时射线源在CT内部
        return [None, None]

    if source[0] < x_min:
        t = -source[0] / direction[0]
        y = source[1] + t * direction[1]
        z = source[2] + t * direction[2]
        if z < 0:
            t1 = - source[2] / direction[2]
            x_in = source[0] + t1 * direction[0]
            y_in = source[1] + t1 * direction[1]
            p_in = [x_in, y_in, 0]
            t2 = (x_max - source[0]) / direction[0]
            z1 = source[2] + t2 * direction[2]
            if 0 <= z1 <= z_max:
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 > z_max:
                t3 = (z_max - source[2]) / direction[2]
                y_out = min(source[1] + t3 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            else:
                return [None, None]
        else:
            t1 = (x_max - source[0]) / direction[0]
            z1 = source[2] + t1 * direction[2]
            if 0 <= z1 <= z_max:
                y_out = min(source[1] + t1 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [0, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 > z_max:
                t2 = (z_max - source[2]) / direction[2]
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [0, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 < 0:
                t2 = - source[2] / direction[2]
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [0, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            else:
                return [None, None]
    elif source[0] > x_max:
        t = (x_max - source[0]) / direction[0]
        y = source[1] + t * direction[1]
        z = source[2] + t * direction[2]
        if z < 0:
            t1 = - source[2] / direction[2]
            x_in = source[0] + t1 * direction[0]
            y_in = source[1] + t1 * direction[1]
            p_in = [x_in, y_in, 0]
            t2 = - source[0] / direction[0]
            z1 = source[2] + t2 * direction[2]
            if 0 <= z1 <= z_max:
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 > z_max:
                t3 = (z_max - source[2]) / direction[2]
                y_out = min(source[1] + t3 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            else:
                return [None, None]
        else:
            t1 = - source[0] / direction[0]
            z1 = source[2] + t1 * direction[2]
            if 0 <= z1 <= z_max:
                y_out = min(source[1] + t1 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [x_max, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 > z_max:
                t2 = (z_max - source[2]) / direction[2]
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [x_max, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            elif z1 < 0:
                t2 = - source[2] / direction[2]
                y_out = min(source[1] + t2 * direction[1], y_max)
                y_out = max(y_out, y_min)
                t_out = (y_out - source[1]) / direction[1]
                z_out = source[2] + t_out * direction[2]
                x_out = source[0] + t_out * direction[0]
                p_out = [x_out, y_out, z_out]
                p_in = [x_max, y, z]
                startpoint = np.linalg.norm(p_in - origin_CT) + OID
                endpoint = OID - np.linalg.norm(origin_CT - p_out)
                return [startpoint, endpoint]
            else:
                return [None, None]



"""
# 示例用法
CTShape = [512, 512, 228]  # CT 的形状
pixelSpacing = [0.9765, 0.9765]  # 每像素间距 (mm)
sliceThickness = 1.0  # 每层厚度 (mm)
x0, y0, z0 = 256, 256, 114  # 全局坐标系原点在 CT 坐标系中的位置
x_rotated, y_rotated, z_rotated = 1332, 844, 1504  # X 射线源点
OID = 2444

[start, end] = get_X_ray_cross_ct_range(CTShape, pixelSpacing, sliceThickness, x0, y0, z0, x_rotated, y_rotated, z_rotated, OID)
print(start - end, start, end)
"""