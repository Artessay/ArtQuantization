from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def merge_peft_model(model_path: str, adapter_path: str, save_path: str):
    """
    Merge the LoRA weights into the base model.
    """
    # Load the base model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

    # Load the PEFT model
    adapter_model = PeftModel.from_pretrained(base_model, adapter_path)

    # Merge the LoRA weights into the base model
    merged_model = adapter_model.merge_and_unload()

    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"model saved to {save_path}")

if __name__ == '__main__':
    model_path = "/data/Qwen/Qwen2.5-32B-Instruct"
    adapter_path = "/data/Qwen/Qwen2.5-32B-Instruct-Medical-Adapter"
    save_path = "/data/Qwen/Qwen2.5-32B-Instruct-Medical"

    merge_peft_model(model_path, adapter_path, save_path)

