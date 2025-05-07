import torch
import contextlib
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization.gptq.gptq_quantize import accumulate_hessian, make_empty_hessian
from llmcompressor.modifiers.quantization.quantization.base import QuantizationModifier

# Manually define the get_execution_device function
def get_execution_device(model: torch.nn.Module) -> torch.device:
    """
    Returns the appropriate device (GPU if available, otherwise CPU) for the model.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shapley correction function
def apply_shapley_correction(theta: torch.Tensor, H: torch.Tensor, alpha=0.1) -> torch.Tensor:
    """
    Apply Shapley correction to the Hessian matrix.

    :param theta: The weights (parameters) of the module
    :param H: The Hessian matrix
    :param alpha: The correction factor (default is 0.1)
    :return: Corrected Hessian matrix
    """
    eps = 1e-6
    # print("🔥  Shapley correction triggered")
    
    # Transpose theta for easier calculations
    theta = theta.transpose(0, 1)  
    H_diag = torch.diag(H)  # Extract the diagonal of the Hessian
    H_diagV = H_diag.unsqueeze(1)
    
    # Compute the raw Hessian correction
    raw = -0.5 * theta * H_diagV
    raw_sum = raw.sum(dim=1)
    
    # Ensure both tensors have the same dtype for matrix multiplication
    Hv = H.to(theta.dtype) @ theta  # Cast H to match the dtype of theta
    shapley = -0.5 * theta * Hv  # Shapley correction
    shapley_sum = shapley.sum(dim=1)
    
    # Compute the weight for scaling
    weight = torch.abs(shapley_sum) / (torch.abs(raw_sum) + eps)
    
    # Apply the Shapley correction to the diagonal of Hessian
    corrected_diag = alpha * weight * H_diag + (1 - alpha) * H_diag
    
    return torch.diag_embed(corrected_diag)

# Modified GPTQModifier to include Shapley correction
class GPTQModifierWithShapleyCorrection(GPTQModifier):
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
        inp = args[0]

        # Initialize Hessian if not present
        if module not in self._num_samples:
            init_device = (
                "cpu" if self.offload_hessians else get_execution_device(module)
            )
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
        corrected_hessian = apply_shapley_correction(module.weight, self._hessians[module], alpha=0.1)
        
        # Update the Hessian with the corrected version
        self._hessians[module] = corrected_hessian

from llmcompressor.modifiers.factory import ModifierFactory
ModifierFactory._registered_registry["GPTQModifierWithShapleyCorrection"] = GPTQModifierWithShapleyCorrection

# Quantization scheme and model paths
scheme = "W4A16"      # Int 4 quantization (linear layers in int4, activations in float16)
model_path = "/data/Qwen/Qwen2.5-0.5B-Instruct"          # Model load path
quant_path = f"/data/Qwen/Qwen2.5-0.5B-Instruct-ShapleyGPTQ-{scheme}"   # Quantized model save path

# Recipe with SmoothQuant and GPTQ with Shapley correction
recipe = [          
    SmoothQuantModifier(smoothing_strength=0.8),            
    GPTQModifierWithShapleyCorrection(scheme=scheme, targets="Linear", ignore=["lm_head"]),
]

# Run quantization with Shapley correction
oneshot(
    model=model_path,
    dataset="open_platypus",
    recipe=recipe,
    output_dir=quant_path,
    max_seq_length=2048,
    num_calibration_samples=512,
)