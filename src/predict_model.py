import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import glob
from datetime import datetime

# TO FIX TO BE SURE THAT THE PATH OF TTHE DIRECTORY IS CORRECT AND THE NAME OF THE CHECKPOINT IS ALWAYS checkpoint-500

def predict():
    # Look from the most recent train to load the model
    OUT_DIR = f"{os.getcwd()}/models"
    directories = [d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))]
    # Find the directory with the most recent timestamp
    most_recent_directory = max(directories, key=lambda d: datetime.strptime(d, "%Y-%m-%d-%H:%M:%S"))

    model_path = f'{most_recent_directory}/checkpoint-500/'  # CHANCE PATH
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(most_recent_directory) # CHANCE PATH
    input_str = "Lisa Kristine: Billeder der bærer vidne til moderne slaveri"

    # Tokenize the input string and return it as a PyTorch tensor
    input_tensor = tokenizer.encode(input_str, return_tensors='pt')

    # Generate a prediction
    output = model.generate(input_tensor)

    # Decode the prediction
    decoded_output = tokenizer.decode(output[0])

    print(decoded_output)