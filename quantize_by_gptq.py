from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "Qwen/Qwen2.5-32B-Instruct"
quant_path = "Qwen2.5-32B-Instruct-GPTQ-4bit"

calibration_dataset = load_dataset(
    "mit-han-lab/pile-val-backup",
    split="validation"
).select(range(1024))["text"]

quant_config = QuantizeConfig(bits=4, group_size=128)

max_memory = {
    0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB", 
    4: "24GiB", 5: "24GiB", 6: "24GiB", 7: "24GiB", 
    "cpu": "256GiB", 
}
model = GPTQModel.load(model_id, quant_config, device_map="auto", device="cuda", max_memory=max_memory)

# increase `batch_size` to match gpu/vram specs to speed up quantization
model.quantize(calibration_dataset, batch_size=1)

model.save(quant_path)