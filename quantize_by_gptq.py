from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "Qwen/Qwen2.5-32B-Instruct"
quant_path = "Qwen2.5-32B-Instruct-GPTQ-4bit"

calibration_dataset = load_dataset(
    "mit-han-lab/pile-val-backup",
    split="validation"
  ).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

model = GPTQModel.load(model_id, quant_config, device_map="auto", device="cuda")

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)