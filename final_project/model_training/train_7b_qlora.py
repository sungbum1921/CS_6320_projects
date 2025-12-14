import torch
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
    logging,
)
from peft import LoraConfig, peft_model
from trl import SFTTrainer

# --- Configuration ---
# Options: "codellama/CodeLlama-7b-hf", "mistralai/Mistral-7B-v0.1"
MODEL_NAME = "codellama/CodeLlama-7b-hf" 
NEW_MODEL_NAME = "codellama-7b-finetuned-sql"

# QLoRA Parameters
LORA_R = 16 # LoRA attention dimension
LORA_ALPHA = 16 # Alpha parameters for LoRA scaling
LORA_DROPOUT = 0.05 

# BitsAndBytes Parameters
USE_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = "float16" # float16 or bfloat16
BNB_4BIT_QUANT_TYPE = "nf4" # fp4 or nf4
USE_NESTED_QUANT = False

# Training Parameters
OUTPUT_DIR = "./results_qlora"
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # Keep small for VRAM
GRADIENT_ACCUMULATION_STEPS = 16 # Changed from 4 to 16 (Crucial for learning stability with Batch 1)
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit" # Paged optimizer to save memory
MAX_GRAD_NORM = 0.3
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "constant"

def train_model():
    print(f"--- Starting QLoRA Training for {MODEL_NAME} ---")
    
    # 1. Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device != "cuda":
        print("WARNING: QLoRA requires a GPU. This will likely fail or be extremely slow on CPU.")

    # 2. Dataset Preparation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, 'NL2SQL_Modified.csv')
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Format for Causal LM (Input + Output in one string)
    # Instruction format: ### Instruction: ... ### Response: ...
    def format_instruction(sample):
        return f"### Instruction:\nTranslate to SQL: {sample['Prompt']}\n\n### Response:\n{sample['Query']}"
    
    df['text'] = df.apply(format_instruction, axis=1)
    dataset = Dataset.from_pandas(df[['text']])

    # 3. Model Loading with Quantization
    compute_dtype = getattr(torch, BNB_4BIT_COMPUTE_DTYPE)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=USE_NESTED_QUANT,
    )

    print(f"Loading base model: {MODEL_NAME}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
    except Exception as e:
        print(f"Error loading model (Ensure bitsandbytes is working): {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Training Configuration (SFTConfig replaces TrainingArguments)
    from trl import SFTConfig
    
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=None, # Will default to tokenizer model max length or 1024
        packing=False,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=25,
        logging_steps=25,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        bf16=False,
        max_grad_norm=MAX_GRAD_NORM,
        max_steps=-1,
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="none"
    )

    # 6. SFT Trainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    # 7. Start Training
    print(">>> Starting Training...")
    trainer.train()
    print(">>> Training Complete!")
    
    # 8. Save Model
    save_path = os.path.join(current_dir, NEW_MODEL_NAME)
    print(f"Saving adapter to: {save_path}")
    trainer.model.save_pretrained(save_path)
    print("Done.")

if __name__ == "__main__":
    train_model()
