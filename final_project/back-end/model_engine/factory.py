from .t5_adapter import T5SQLAdapter
from .llama_adapter import LlamaQLoRAAdapter

def get_model_engine(model_type: str):
    if model_type in ["t5_small", "t5_base"]:
        return T5SQLAdapter()
    elif model_type == "codellama_7b_QLoRA":
        return LlamaQLoRAAdapter()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
