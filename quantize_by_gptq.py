from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

# scheme="W8A8"
# model_path = "Qwen/Qwen2.5-32B-Instruct"
# quant_path = "Qwen2.5-32B-Instruct-GPTQ-INT8"

scheme="W4A16"      # Int 4量化 (线性层是int4，激活参数float16)
model_path = "/data/Qwen/Qwen2.5-0.5B-Instruct"          # model load
quant_path = f"/data/Qwen/Qwen2.5-0.5B-Instruct-GPTQ-{scheme}"   # model saver

# Select quantization algorithm. In this case, we:
#   * apply SmoothQuant to make the activations easier to quantize
#   * quantize the weights to int4 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)

recipe = [          # 量化的配置
    SmoothQuantModifier(smoothing_strength=0.8),            
    GPTQModifier(scheme=scheme, targets="Linear", ignore=["lm_head"]),
]

# Apply quantization using the built in open_platypus dataset.
#   * See examples for demos showing how to pass a custom calibration set

# 运行
oneshot(
    model=model_path,
    dataset="open_platypus",
    recipe=recipe,
    output_dir=quant_path,
    max_seq_length=2048,
    num_calibration_samples=512,
)