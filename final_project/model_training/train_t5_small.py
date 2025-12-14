import pandas as pd
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, AutoConfig
from datasets import Dataset

def train_model():
    print("--- Starting T5-Small Training Script ---")

    # 1. Device Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("WARNING: CUDA (GPU) not detected. Training will use CPU and might be slow.")

    # 2. File Paths
    # Script assumes it's running in the 'model_training' directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, 'NL2SQL_Modified.csv')
    
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return

    print(f"Loading data from: {filepath}")
    data = pd.read_csv(filepath)

    # 3. Preprocessing
    print("Preprocessing data...")
    # Add prefix for T5
    data['input'] = "Translate to SQL: " + data['Prompt']
    # Clean target whitespace
    data['target'] = data['Query'].str.replace(r'\s+', ' ', regex=True)

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(data[['input', 'target']])

    # 4. Load Model & Tokenizer
    model_name = "t5-small"
    print(f"Loading model: {model_name}...")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
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
    output_dir = os.path.join(current_dir, "t5_small_test_output")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch", # Check validation loss every epoch
        learning_rate=3e-4,
        per_device_train_batch_size=16, # Changed from 8 to 16 (Recommended for T5-Small)
        per_device_eval_batch_size=16, # Changed from 8 to 16
        num_train_epochs=10, # Changed from 3 to 10 (Small models need more iterations)
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50,
        report_to="none", # Disable external logging services
        no_cuda=False if device == "cuda" else True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset, # Using same set for quick demo
    )

    # 7. Start Training
    print("\n>>> Starting Training (3 Epochs)...")
    trainer.train()
    print(">>> Training Complete!")

    # 8. Save Model
    save_path = os.path.join(current_dir, "t5_small_finetuned_test")
    print(f"Saving model to: {save_path}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    # 9. Test Inference
    print("\n>>> Testing Model...")
    test_query = "Show me all students"
    input_text = f"Translate to SQL: {test_query}"
    
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input: {test_query}")
    print(f"Generated SQL: {result}")
    print("\nDone.")

if __name__ == "__main__":
    train_model()
