import torch


def log_rescale(weight, target_min=0.1, target_max=100):
    """
    将权重从原始范围压缩到指定的指数范围内

    参数:
    weight: 输入的1*N tensor
    target_min: 目标范围下限，默认为0.1 (10^-1)
    target_max: 目标范围上限，默认为100 (10^2)

    返回:
    修正后的tensor
    """
    # 处理正负值
    sign = torch.sign(weight)
    abs_weight = torch.abs(weight)

    # 添加小常数避免log(0)
    eps = 1e-10
    abs_weight = torch.clamp(abs_weight, min=eps)

    # 计算原始范围的对数边界
    log_min = torch.log(torch.tensor(eps, dtype=weight.dtype, device=weight.device))
    log_max = torch.log(torch.max(abs_weight))

    # 对绝对值进行对数变换
    log_weight = torch.log(abs_weight)

    # 线性缩放到目标范围
    scaled_log = (log_weight - log_min) / (log_max - log_min)
    target_log_min = torch.log(torch.tensor(target_min, dtype=weight.dtype, device=weight.device))
    target_log_max = torch.log(torch.tensor(target_max, dtype=weight.dtype, device=weight.device))
    target_log = target_log_min + scaled_log * (target_log_max - target_log_min)

    # 指数变换回线性空间
    scaled_weight = torch.exp(target_log)

    # 恢复符号
    return sign * scaled_weight


def piecewise_power_transform(weight, target_min=0.1, target_max=100):
    """
    对不同区间的数据使用不同的幂次变换，实现非均匀压缩

    参数:
    weight: 输入的1*N tensor
    target_min: 目标范围下限，默认为0.1 (10^-1)
    target_max: 目标范围上限，默认为100 (10^2)

    返回:
    修正后的tensor
    """
    # 计算原始数据的统计量
    abs_weight = torch.abs(weight)
    max_val = torch.max(abs_weight)
    min_val = torch.min(abs_weight)

    # 定义分段点
    mid_point = torch.sqrt(max_val * target_min)  # 几何中点

    # 创建掩码
    small_mask = abs_weight < mid_point
    large_mask = abs_weight >= mid_point

    # 对小值区域使用幂次变换（放大）
    # 这里使用指数为0.5的幂次，将小值区域拉伸
    small_scaled = torch.pow(abs_weight[small_mask] / mid_point, 0.5) * mid_point

    # 对大值区域使用幂次变换（压缩）
    # 这里使用指数为2的幂次，将大值区域压缩
    large_scaled = mid_point * torch.pow(abs_weight[large_mask] / mid_point, 0.25)

    # 组合结果
    scaled_abs = torch.zeros_like(abs_weight)
    scaled_abs[small_mask] = small_scaled
    scaled_abs[large_mask] = large_scaled

    # 缩放到目标范围
    final_scaled = target_min + (scaled_abs - min_val) * (target_max - target_min) / (torch.max(scaled_abs) - min_val)

    # 恢复符号
    return torch.sign(weight) * final_scaled


def custom_sigmoid_transform(weight, target_min=0.1, target_max=100, alpha=0.1):
    """
    使用自定义S型函数将数据压缩到指定范围

    参数:
    weight: 输入的1*N tensor
    target_min: 目标范围下限，默认为0.1 (10^-1)
    target_max: 目标范围上限，默认为100 (10^2)
    alpha: 控制压缩强度的参数，越小压缩越强

    返回:
    修正后的tensor
    """
    # 计算原始数据的统计量
    mean_val = torch.mean(weight)
    std_val = torch.std(weight)

    # 标准化数据
    normalized = (weight - mean_val) / (std_val + 1e-10)

    # 自定义S型函数（比标准sigmoid更陡峭）
    compressed = target_min + (target_max - target_min) * (1 / (1 + torch.exp(-alpha * normalized)))

    return compressed


def double_threshold_compression(weight, low_threshold=1.0, high_threshold=100.0, target_min=0.1, target_max=100):
    """
    使用双阈值策略对数据进行非线性压缩

    参数:
    weight: 输入的1*N tensor
    low_threshold: 低阈值，低于此值的数据会被放大
    high_threshold: 高阈值，高于此值的数据会被压缩
    target_min: 目标范围下限，默认为0.1 (10^-1)
    target_max: 目标范围上限，默认为100 (10^2)

    返回:
    修正后的tensor
    """
    # 处理正负值
    sign = torch.sign(weight)
    abs_weight = torch.abs(weight)

    # 创建掩码
    small_mask = abs_weight < low_threshold
    medium_mask = (abs_weight >= low_threshold) & (abs_weight < high_threshold)
    large_mask = abs_weight >= high_threshold

    # 对小值区域进行放大（使用对数变换）
    small_scaled = target_min + (abs_weight[small_mask] / low_threshold) * (low_threshold - target_min)

    # 对中间区域保持线性
    medium_scaled = low_threshold + (abs_weight[medium_mask] - low_threshold) * (target_max - low_threshold) / (
        high_threshold - low_threshold
    )

    # 对大值区域进行压缩（使用平方根变换）
    large_scaled = target_max - (target_max - high_threshold) / torch.sqrt(abs_weight[large_mask] / high_threshold)

    # 组合结果
    scaled_abs = torch.zeros_like(abs_weight)
    scaled_abs[small_mask] = small_scaled
    scaled_abs[medium_mask] = medium_scaled
    scaled_abs[large_mask] = large_scaled

    # 恢复符号
    return sign * scaled_abs


def exponential_scaling(weight, target_min=0.1, target_max=100):
    """
    使用指数函数将数据压缩到指定范围

    参数:
    weight: 输入的1*N tensor
    target_min: 目标范围下限，默认为0.1 (10^-1)
    target_max: 目标范围上限，默认为100 (10^2)

    返回:
    修正后的tensor
    """
    # 计算原始数据的统计量
    min_val = torch.min(weight)
    max_val = torch.max(weight)

    # 标准化到[0,1]范围
    normalized = (weight - min_val) / (max_val - min_val + 1e-10)

    # 使用指数函数进行变换
    # 这里使用底数为e的指数函数，可根据需要调整
    exponent = torch.log(torch.tensor(target_max / target_min, dtype=weight.dtype, device=weight.device))
    scaled = target_min * torch.exp(normalized * exponent)

    return scaled


def sigmoid_plus_one(weight):
    """
    对权重进行sigmoid变换后加1

    参数:
    weight: 输入的1*N tensor

    返回:
    修正后的tensor，范围在[1, 2]之间
    """
    # 应用sigmoid函数
    sigmoid_weight = torch.sigmoid(weight)

    # 加1，使结果范围在[1, 2]之间
    result = sigmoid_weight + 1

    return result
