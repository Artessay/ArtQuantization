from llmcompressor import oneshot
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

from modifiers import GPTQModifierWithShapleyCorrection

# scheme="FP8"
scheme = "W8A16"
model_path = "/data/Qwen/Qwen2.5-7B-Instruct"
quant_path = f"{model_path}-ShapleyGPTQ-{scheme}"
print(quant_path)

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int4 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifierWithShapleyCorrection(scheme=scheme, targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the built in open_platypus dataset.
#   * See examples for demos showing how to pass a custom calibration set
oneshot(
    model=model_path,
    dataset="open_platypus",
    recipe=recipe,
    output_dir=quant_path,
    max_seq_length=2048,
    num_calibration_samples=512,
)
