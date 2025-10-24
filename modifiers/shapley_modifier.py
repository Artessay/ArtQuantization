from typing import Optional, Tuple

import torch
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import (
    accumulate_hessian,
    make_empty_hessian,
)
from pydantic import Field

from modifiers.utils import (
    custom_sigmoid_transform,
    double_threshold_compression,
    exponential_scaling,
    log_rescale,
    piecewise_power_transform,
    sigmoid_plus_one,
    quantile_mapping,
)


# Manually define the get_execution_device function
def get_execution_device(model: torch.nn.Module) -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU) for the model.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_shapley_correction(
    theta: torch.Tensor,
    H: torch.Tensor,
    alpha=0.1,
    correction_type=None,
) -> torch.Tensor:
    """
    Apply Shapley correction to the Hessian matrix.

    :param theta: The weights (parameters) of the module
    :param H: The Hessian matrix
    :param alpha: The correction factor (default is 0.1)
    :param correction_type: Non-linearity modifier type, None for no non-linear correction
    :return: Corrected Hessian matrix
    """

    eps = 1e-6

    # Transpose theta for easier calculations
    theta = theta.transpose(0, 1)  # bf16
    H_diag = torch.diag(H)  # Extract the diagonal of the Hessian # fl32
    H_diagV = H_diag.unsqueeze(1)  # fl32

    # Compute the raw Hessian correction
    raw = -0.5 * theta * H_diagV  # fl32
    raw_sum = raw.sum(dim=1)  # fl32

    # Ensure both tensors have the same dtype for matrix multiplication
    Hv = H.to(theta.dtype) @ theta  # Cast H to match the dtype of theta # bf16
    shapley = -0.5 * theta * Hv  # Shapley correction # bf16
    shapley_sum = shapley.sum(dim=1)  # bf16

    # Compute the original Shapley weight (这就是W_original)
    W_original = torch.abs(shapley_sum) / (torch.abs(raw_sum) + eps)  # fl32

        
    weight = W_original.clone()
    # print(weight.min(), weight.max(), weight.mean())

    # 应用非线性变换得到修正后的权重（如果指定了修正器）
    if correction_type is None:
        pass
    elif correction_type == "clamp":
        weight = torch.clamp(weight, min=1.0, max=100.0)    # decrease performance; 分析 scale，对比INT8的均值，来确定scale
    elif correction_type == "log_rescale":
        weight = torch.clamp(weight, min=1.0, max=100.0)
        weight = log_rescale(weight)
    elif correction_type == "custom_sigmoid_transform":
        weight = custom_sigmoid_transform(weight)
    elif correction_type == "double_threshold_compression":
        weight = double_threshold_compression(weight)
    elif correction_type == "exponential_scaling":
        weight = exponential_scaling(weight)
    elif correction_type == "piecewise_power_transform":
        weight = piecewise_power_transform(weight)
    elif correction_type == "sigmoid_plus_one":
        # clip %5
        # 计算第5%和第95%分位数
        q_05 = torch.quantile(weight, 0.05)  # 第5%分位数（下限）
        q_95 = torch.quantile(weight, 0.95)  # 第95%分位数（上限）

        # 将权重裁剪到[q_05, q_95]范围内
        weight = torch.clamp(weight, min=q_05, max=q_95)
        weight = sigmoid_plus_one(weight)
    elif correction_type == "quantile_mapping":
        weight = quantile_mapping(weight, target_min=1.0, target_max=10.0)
    else:
        raise ValueError(f"Invalid non-linearity modifier: {correction_type}")

    # Apply the Shapley correction to the diagonal of Hessian
    # v1: alpha is 0 equal to OBS, only use diag elements
    corrected_diag = alpha * weight * H_diag + (1 - alpha) * H_diag
    return torch.diag_embed(corrected_diag)

    # v2: alpha is 0 equal to GPTQ, use all elements in Hessian
    corrected_H = alpha * weight * torch.diag_embed(H_diag) + (1 - alpha) * H
    return corrected_H
    
    # v3
    H_diag_corrected = alpha * weight * H_diag + (1 - alpha) * H_diag  # fl32
    corrected_H = H.clone()  # fl32
    corrected_H[torch.arange(H.shape[0]), torch.arange(H.shape[1])] = H_diag_corrected  # fl32
    return corrected_H

# Modified GPTQModifier to include Shapley correction
class GPTQModifierWithShapleyCorrection(GPTQModifier):
    alpha: float = Field(default=None, description="Shapley correction factor")
    correction_type: Optional[str] = Field(default=None, description="Non-linearity correction type")

    # 添加类变量来跟踪进度
    _module_counter = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = kwargs.get("alpha", None)
        if self.alpha is None:
            raise ValueError("alpha must be provided")
        # 重置计数器
        GPTQModifierWithShapleyCorrection._module_counter = 0
        correction_type = None # "quantile_mapping"

    def calibrate_module(
        self,
        module: torch.nn.Module,
        args: Tuple[torch.Tensor, ...],
        _output: torch.Tensor,
    ):
        """
        Quantize a module's weight according to the GPTQ algorithm with Shapley correction.

        :param module: The module being quantized
        :param args: Input arguments for the module forward pass
        """
        # 检查是否是第一次处理这个模块
        is_first_time = module not in self._num_samples

        # 增加计数器
        if is_first_time:
            GPTQModifierWithShapleyCorrection._module_counter += 1

            # 计算当前进度
            print(
                f"🔥 Processing module #{GPTQModifierWithShapleyCorrection._module_counter} (Processed: {len(self._num_samples)}): {module}"
            )

        inp = args[0]

        # Initialize Hessian if not present
        if is_first_time:
            init_device = "cpu" if self.offload_hessians else get_execution_device(module)
            self._hessians[module] = make_empty_hessian(module, device=init_device)
            self._num_samples[module] = 0

        # Accumulate Hessian with input with optional offloading
        with self._maybe_onload_hessian(module):
            self._hessians[module], self._num_samples[module] = accumulate_hessian(
                inp,
                module,
                self._hessians[module],
                self._num_samples[module],
            )

        # Apply Shapley correction to the Hessian matrix
        corrected_hessian = apply_shapley_correction(
            module.weight,
            self._hessians[module],
            alpha=self.alpha,
            correction_type=self.correction_type,
        )

        # Update the Hessian with the corrected version
        self._hessians[module] = corrected_hessian
