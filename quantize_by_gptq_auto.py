from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "Qwen/Qwen2.5-32B-Instruct"
quantized_model_dir = "Qwen2.5-32B-Instruct-GPTQ-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)


calibration_dataset = load_dataset(
    "mit-han-lab/pile-val-backup",
    split="validation"
).select(range(1024))["text"]
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

max_memory = {
    0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB", 
    4: "24GiB", 5: "24GiB", 6: "24GiB", 7: "24GiB", 
    "cpu": "256GiB", 
}

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, max_memory=max_memory, device_map="auto")

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir)