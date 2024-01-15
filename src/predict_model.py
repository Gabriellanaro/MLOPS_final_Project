import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import glob
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from http import HTTPStatus

# TO FIX TO BE SURE THAT THE PATH OF TTHE DIRECTORY IS CORRECT AND THE NAME OF THE CHECKPOINT IS ALWAYS checkpoint-500

def predict(input_str: str):
    # Look from the most recent train to load the model
    OUT_DIR = f"{os.getcwd()}/models"
    directories = [d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))]
    # Find the directory with the most recent timestamp
    most_recent_directory = max(directories, key=lambda d: datetime.strptime(d, "%Y-%m-%d-%H:%M:%S"))

    model_path = f'{most_recent_directory}/checkpoint-500/'  # CHANCE PATH
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(most_recent_directory) # CHANCE PATH
    #input_str = "Lisa Kristine: Billeder der bærer vidne til moderne slaveri"

    # Tokenize the input string and return it as a PyTorch tensor
    input_tensor = tokenizer.encode(input_str, return_tensors='pt')

    # Generate a prediction
    output = model.generate(input_tensor)

    # Decode the prediction
    translated_text = tokenizer.decode(output[0])

    return {"translated_text": translated_text}




app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


"""class Sentence(BaseModel):
    text: str

@app.post("/translate/")
async def read_sentence(sentence: Sentence):
    prediction = predict(sentence.text)
    return {"sentence": sentence.text, "prediction": prediction}"""
