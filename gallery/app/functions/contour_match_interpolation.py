import numpy as np
import cv2


def contour_matching_interpolation(label_data, current_thickness, target_thickness=1):
    """
    基于轮廓匹配的层间插值算法，用于对二值化的3D肿瘤标签数据进行任意层厚插值。
    :param label_data: 3D肿瘤标签数据 (shape: [height, width, depth])
    :param current_thickness: 当前层厚 (mm)
    :param target_thickness: 目标层厚 (mm)，默认为 1 mm
    :return: 插值后的3D肿瘤标签数据
    """
    height, width, depth = label_data.shape
    num_new_layers = int(current_thickness / target_thickness) - 1
    new_depth = depth * (num_new_layers + 1)
    interpolated_labels = np.zeros((height, width, new_depth), dtype=np.uint8)

    kernel = np.ones((150, 150), np.uint8)

    # 保留原始层数据
    interpolated_labels[:, :, ::(num_new_layers + 1)] = label_data.astype(np.uint8)

    # 执行层间插值
    for i in range(depth - 1):
        # 获取当前层和下一层
        layer1 = label_data[:, :, i].astype(np.uint8)
        layer2 = label_data[:, :, i + 1].astype(np.uint8)

        # 生成中间层
        temp_layers = [layer1]
        for t in range(1, num_new_layers + 1):
            # 临时层的权重
            alpha = t / (num_new_layers + 1)
            temp_layer = generate_interpolated_layer(layer1, layer2, alpha, kernel)
            temp_layers.append(temp_layer)

        temp_layers.append(layer2)

        # 插入到结果中
        for t, temp_layer in enumerate(temp_layers):
            interpolated_labels[:, :, i * (num_new_layers + 1) + t] = temp_layer

    return interpolated_labels


def split_contour_into_quadrants(contour, center):
    """
    按象限分割轮廓点
    :param contour: 单个轮廓点的坐标列表 (Nx2)
    :param center: 中心点坐标 (标量值)
    :return: 划分到四个象限的点列表
    """
    center_x, center_y = center.flatten()  # 确保中心点为标量
    quadrants = [[] for _ in range(4)]
    for point in contour:
        x, y = point[0]
        if x >= center_x and y >= center_y:
            quadrants[0].append((x, y))
        elif x < center_x and y >= center_y:
            quadrants[1].append((x, y))
        elif x < center_x and y < center_y:
            quadrants[2].append((x, y))
        elif x >= center_x and y < center_y:
            quadrants[3].append((x, y))
    return quadrants


def match_and_interpolate_contours(contour1, contour2, alpha):
    """
    根据权重 alpha 匹配并插值轮廓点
    :param contour1: 前一层轮廓
    :param contour2: 后一层轮廓
    :param alpha: 权重，范围 [0, 1]
    :return: 插值后的轮廓点
    """
    n1, n2 = len(contour1), len(contour2)
    if n1 > n2:
        contour2 = np.append(contour2, [contour2[-1]] * (n1 - n2), axis=0)
    elif n2 > n1:
        contour1 = np.append(contour1, [contour1[-1]] * (n2 - n1), axis=0)

    interpolated = (1 - alpha) * np.array(contour1) + alpha * np.array(contour2)
    return interpolated


def generate_interpolated_layer(layer1, layer2, alpha, kernel):
    """
    根据权重 alpha 生成两层之间的插值层
    :param layer1: 前一层
    :param layer2: 后一层
    :param alpha: 权重，范围 [0, 1]
    :param kernel: 闭运算核
    :return: 插值层
    """
    # 提取轮廓
    contour1 = cv2.findContours(layer1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contour2 = cv2.findContours(layer2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if len(contour1) == 0 or len(contour2) == 0:
        return ((1 - alpha) * layer1 + alpha * layer2).astype(np.uint8)

    center1 = np.mean(contour1[0], axis=0).astype(int)
    center2 = np.mean(contour2[0], axis=0).astype(int)

    # 按象限分割轮廓
    quadrants1 = split_contour_into_quadrants(contour1[0], center1)
    quadrants2 = split_contour_into_quadrants(contour2[0], center2)

    # 匹配和插值轮廓
    interpolated_contour = []
    for q1, q2 in zip(quadrants1, quadrants2):
        matched_contour = match_and_interpolate_contours(q1, q2, alpha)
        interpolated_contour.extend(matched_contour)

    # 绘制插值结果
    temp_layer = np.zeros_like(layer1, dtype=np.uint8)
    interpolated_contour = np.array(interpolated_contour).astype(int)
    cv2.drawContours(temp_layer, [interpolated_contour], -1, color=1, thickness=-1)

    # 闭运算
    temp_layer = cv2.morphologyEx(temp_layer, cv2.MORPH_CLOSE, kernel)
    return temp_layer
