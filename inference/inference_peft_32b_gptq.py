from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/data/Qwen/Qwen2.5-32B-Instruct-Medical-GPTQ-W4A8"

# Device management for model parallelism using accelerate
# You don't need to call model.to() here if using accelerate correctly with device_map="auto"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"  # The device map will handle device placement
)

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(model_name)

for i in range(10):
    print(f"#{i+1}")
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    # Format the conversation for GPT model
    conversation = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            conversation += f"System: {content}\n"
        elif role == "user":
            conversation += f"User: {content}\n"
    
    # Tokenize the input
    model_inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

    # Generate a response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    
    # Decode the response
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(response)
