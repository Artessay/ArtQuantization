import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the quantized model
model_path = "/data/Qwen/Qwen2.5-0.5B-Instruct-ShapleyGPTQ-W4A16"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure the model is in evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=False)
model.to(device)  # Move model to the appropriate device (GPU or CPU)

# Set the padding token explicitly to avoid issues with padding/eos token
tokenizer.pad_token = tokenizer.eos_token  # Ensuring padding and eos tokens are distinct
model.config.pad_token_id = tokenizer.pad_token_id  # Set the pad_token_id to ensure proper padding behavior

def ask_question(question: str) -> str:
    """
    This function takes a question, tokenizes it, and generates an answer from the model.

    :param question: The question to be answered
    :return: The generated response from the model
    """
    # Tokenize the input question and move the tensors to the correct device
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(device)  # Move input_ids to the same device as the model
    attention_mask = inputs['attention_mask'].to(device)  # Move attention_mask to the same device as the model
    
    # Run inference on the model with the attention mask
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=256, num_return_sequences=1)
    
    # Decode the output into a human-readable text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# Example question
question = "What is the capital of France?"

# Get the answer from the model
generated_answer = ask_question(question)

# Print the generated answer
print(generated_answer)
