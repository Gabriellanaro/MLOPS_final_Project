import os
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)

MODEL = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL)

# tokenizer_path = "./MLOPS_final_Project/outputs/2024-01-12/13-40-09/models/2024_01_12_13_40_09/tokenizer/"
#model_path = "./MLOPS_final_Project/outputs/2024-01-12/13-40-09/models/2024_01_12_13_40_09/model"
# tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
#print(os.listdir(model_path))
#model = T5ForConditionalGeneration.from_pretrained(model_path)


# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
