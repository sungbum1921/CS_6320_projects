import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from .base import BaseSQLModel
import os

class LlamaQLoRAAdapter(BaseSQLModel):
    def __init__(self, base_model_id="codellama/CodeLlama-7b-hf"):
        # Note: base_model_id could also be passed during init if needed, 
        # but usually QLoRA assumes a specific base model.
        self.base_model_id = base_model_id
        self.model = None
        self.tokenizer = None

    def load(self, adapter_path: str):
        print(f"Loading LLaMA QLoRA from {adapter_path}...")
        
        if not os.path.exists(adapter_path):
             raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        # Check for GPU
        if not torch.cuda.is_available():
            print("WARNING: GPU not available. BitsAndBytes 4-bit quantization requires a GPU.")
        
        # 1. Base Model Load (4-bit quant)
        # 1. Base Model Load
        try:
            try:
                print("Attempting to load model with 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id, 
                    quantization_config=bnb_config, 
                    device_map="auto"
                )
            except Exception as e:
                print(f"Quantization failed ({e}). Falling back to standard float16 load.")
                print("WARNING: This may require significantly more GPU VRAM (approx 14GB for 7B model).")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_id,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            base_model.config.use_cache = False
            base_model.config.pretraining_tp = 1
            
            # 2. Load Adapter (QLoRA)
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
            
            # 3. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
        except Exception as e:
            print(f"Error loading LLaMA model: {e}")
            raise e

    def generate_sql(self, input_text: str) -> str:
        if not self.model or not self.tokenizer:
             raise RuntimeError("Model is not loaded.")

        # Prepare Prompt consistent with training formatting
        prompt = f"### Instruction:\nTranslate to SQL: {input_text}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=200)
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response part
        if "### Response:" in decoded:
            return decoded.split("### Response:")[-1].strip()
        return decoded
