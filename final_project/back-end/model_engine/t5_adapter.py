from transformers import T5ForConditionalGeneration, T5Tokenizer
from .base import BaseSQLModel
import torch
import os

class T5SQLAdapter(BaseSQLModel):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str):
        print(f"Loading T5 model from {model_path} on {self.device}...")
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model path not found: {model_path}")

        try:
            # legacy=False avoids warnings for newer transformers versions
            self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True).to(self.device)
        except Exception as e:
            print(f"Error loading T5 model: {e}")
            raise e

    def generate_sql(self, input_text: str) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model is not loaded. Call load() first.")

        input_text_fmt = f"Translate to SQL: {input_text}"
        inputs = self.tokenizer(input_text_fmt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512)
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
