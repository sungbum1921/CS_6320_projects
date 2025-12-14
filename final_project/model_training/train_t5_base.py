import pandas as pd
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

# --- Configuration ---
# Options: "t5-base", "Salesforce/codet5-base"
MODEL_NAME = "t5-base" 
# MODEL_NAME = "Salesforce/codet5-base"

# Decrease batch size for Base models (High VRAM usage)
BATCH_SIZE = 4 
GRADIENT_ACCUMULATION_STEPS = 8 # Changed from 2 to 8 (Effective batch size ~32)
EPOCHS = 10 # Changed from 3 to 10 (Base models converge slower)
LEARNING_RATE = 1e-4 # Slightly lower LR for larger model

def train_model():
    print(f"--- Starting Training Script for {MODEL_NAME} ---")

    # 1. Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. File Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, 'NL2SQL_Modified.csv')
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return

    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath)

    # 3. Preprocessing
    print("Preprocessing data...")
    # T5 expects a task prefix. CodeT5 is T5-based so it generally works well with this too.
    prefix = "Translate to SQL: " 
    data['input'] = prefix + data['Prompt']
    data['target'] = data['Query'].str.replace(r'\s+', ' ', regex=True)

    dataset = Dataset.from_pandas(data[['input', 'target']])

    # 4. Load Model & Tokenizer
    print(f"Loading model: {MODEL_NAME}...")
    try:
        if "codet5" in MODEL_NAME:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        else:
            tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
            model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)

    # 5. Tokenization
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples['input'],
            max_length=128,
            truncation=True,
            padding='max_length'
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'],
                max_length=128,
                truncation=True,
                padding='max_length'
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 6. Training Configuration
    safe_model_name = MODEL_NAME.replace("/", "_")
    output_dir = os.path.join(current_dir, f"{safe_model_name}_output")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE, 
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # Simulate larger batch size
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=20,
        report_to="none",
        no_cuda=False if device == "cuda" else True,
        fp16=True if device == "cuda" else False # Enable Mixed Precision if on GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    # 7. Start Training
    print(f"\n>>> Starting Training ({EPOCHS} Epochs, Batch {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS})...")
    trainer.train()
    print(">>> Training Complete!")

    # 8. Save Model
    save_path = os.path.join(current_dir, f"{safe_model_name}_finetuned")
    print(f"Saving model to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # 9. Test Inference
    print("\n>>> Testing Model...")
    test_query = "Show me all students"
    input_text = f"{prefix}{test_query}"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {test_query}")
    print(f"Generated SQL: {result}")
    print("\nDone.")

if __name__ == "__main__":
    train_model()
