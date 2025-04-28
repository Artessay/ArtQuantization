from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

pretrained_model_name = "Qwen/Qwen2.5-32B-Instruct"
quantized_model_name = "Qwen2.5-32B-Instruct-GPTQ-4bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
gptq_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

max_memory = {
    0: "24GiB", 1: "24GiB", 2: "24GiB", 3: "24GiB", 
    4: "24GiB", 5: "24GiB", 6: "24GiB", 7: "24GiB", 
    "cpu": "256GiB", 
}
quantized_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name,
    device_map="auto",
    max_memory=max_memory,
    quantization_config=gptq_config
)

quantized_model.to("cpu")
quantized_model.save_pretrained(quantized_model_name)
tokenizer.save_pretrained(quantized_model_name)