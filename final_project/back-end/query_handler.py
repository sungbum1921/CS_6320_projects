import os
import gc
from model_engine.factory import get_model_engine

# Configuration
# Default model
DEFAULT_MODEL_TYPE = "t5_small"

# Paths configuration
# The 'model_training' directory is a sibling of 'back-end'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
BASE_MODEL_DIR = os.path.join(PARENT_DIR, "model_training")

MODEL_PATHS = {
    "t5_small": os.path.join(BASE_MODEL_DIR, "t5_small"),
    "t5_base": os.path.join(BASE_MODEL_DIR, "t5_base"),
    "codellama_7b_QLoRA": os.path.join(BASE_MODEL_DIR, "codellama_7b_QLoRA")
}

# Global state
current_engine = None
current_model_type = None

def get_or_load_engine(model_type: str):
    global current_engine, current_model_type
    
    if current_engine is not None and current_model_type == model_type:
        return current_engine

    print(f"--- Switching Model Engine to: {model_type} ---")
    
    # Optional: cleanup previous model to free memory?
    # Python's GC might handle it if we drop the reference, but for GPU/PyTorch, explicit cleanup is better.
    # checking if it has a 'unload' or similar? assuming simple overwrite for now.
    current_engine = None
    gc.collect() 

    try:
        engine = get_model_engine(model_type)
        
        target_path = MODEL_PATHS.get(model_type)
        if not target_path:
            raise ValueError(f"No path configuration found for model: {model_type}")
            
        print(f"Loading model from: {target_path}")
        engine.load(target_path)
        print("Model loaded successfully.")
        
        current_engine = engine
        current_model_type = model_type
        return current_engine

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_type}. Details: {e}")
        raise e

# Initialize default model on startup (Success/Failure logs might appear)
try:
    get_or_load_engine(DEFAULT_MODEL_TYPE)
except Exception as e:
    print(f"Startup model load failed: {e}")


def generate_sql(question, model_type=DEFAULT_MODEL_TYPE):
    try:
        engine = get_or_load_engine(model_type)
        return engine.generate_sql(question)
    except Exception as e:
        return f"Error generating SQL: {str(e)}"
