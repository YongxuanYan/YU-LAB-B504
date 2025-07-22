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
        if layer1.max() == 0 and layer2.max() == 0:
            continue
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


def resample_contour(contour, num_points):
    """
    使用弧长参数化对轮廓进行重采样
    :param contour: 轮廓点列表，每个元素是(x,y)坐标
    :param num_points: 目标点数
    :return: 重采样后的轮廓点列表
    """
    if len(contour) == 0:
        return []

    # 只有一个点时直接重复
    if len(contour) == 1:
        return [contour[0]] * num_points

    # 将轮廓转换为闭合曲线（添加起点到末尾）
    closed_contour = contour + [contour[0]]
    points = np.array(closed_contour)

    # 计算各线段长度和总弧长
    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    total_length = np.sum(dists)

    # 计算累计弧长
    cum_dists = np.zeros(len(points))
    cum_dists[1:] = np.cumsum(dists)

    # 在弧长上等间距采样
    sample_points = []
    step = total_length / num_points
    current_dist = 0

    for _ in range(num_points):
        # 找到当前距离所在的线段
        idx = np.searchsorted(cum_dists, current_dist, side='right') - 1

        # 计算线段上的插值比例
        seg_start = cum_dists[idx]
        seg_end = cum_dists[idx + 1]
        seg_length = seg_end - seg_start
        t = (current_dist - seg_start) / seg_length if seg_length > 0 else 0

        # 线性插值
        p1 = points[idx]
        p2 = points[idx + 1]
        new_point = (1 - t) * p1 + t * p2
        sample_points.append(tuple(new_point))

        # 移动到下一个采样点
        current_dist += step
        if current_dist > total_length:
            current_dist -= total_length  # 处理舍入误差

    return sample_points


def match_and_interpolate_contours(contour1, contour2, alpha):
    """
    改进的轮廓匹配和插值函数
    :param contour1: 前一层轮廓点列表
    :param contour2: 后一层轮廓点列表
    :param alpha: 插值权重 [0, 1]
    :return: 插值后的轮廓点
    """
    n1, n2 = len(contour1), len(contour2)

    # 处理空轮廓情况
    if n1 == 0 and n2 == 0:
        return np.array([])

    # 确定目标点数（取两者最大点数）
    target_points = max(n1, n2) if n1 > 0 and n2 > 0 else max(n1, n2)

    # 处理轮廓1为空的情况
    if n1 == 0:
        # 计算轮廓2的中心点
        contour2_arr = np.array(contour2)
        center = np.mean(contour2_arr, axis=0)
        contour1_resampled = [tuple(center)] * target_points
    else:
        # 对轮廓1进行重采样
        contour1_resampled = resample_contour(contour1, target_points)

    # 处理轮廓2为空的情况
    if n2 == 0:
        # 计算轮廓1的中心点
        contour1_arr = np.array(contour1)
        center = np.mean(contour1_arr, axis=0)
        contour2_resampled = [tuple(center)] * target_points
    else:
        # 对轮廓2进行重采样
        contour2_resampled = resample_contour(contour2, target_points)

    # 线性插值
    interpolated = (1 - alpha) * np.array(contour1_resampled) + alpha * np.array(contour2_resampled)
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
