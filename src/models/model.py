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
model = T5ForConditionalGeneration.from_pretrained(MODEL,return_dict=True)

input = "Mein Name ist Azeem und ich lebe in Indien."
def translate(text: str):
    input_ids = tokenizer("translate Danish to English: "+input, return_tensors="pt").input_ids  # Batch size 1

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded

print(translate(input))
