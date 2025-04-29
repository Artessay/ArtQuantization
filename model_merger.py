import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

def merge_peft_model(model_path, adapter_path):
    """
    Merge the LoRA weights into the base model.
    """
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

    # Load the PEFT model
    peft_model = AutoPeftModelForCausalLM.from_pretrained(adapter_path, device_map="auto", torch_dtype="auto")

    # Merge the LoRA weights into the base model
    model = peft_model.merge_and_unload()

    return model

