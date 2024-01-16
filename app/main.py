# main.py
from fastapi import FastAPI, HTTPException
import requests

app = FastAPI()

@app.get("/translate/{text}")
def translate(text: str):
    # Replace 'gcr.io/<project-id>/your-model-image' with the actual path to your model image in GCR
    model_url = 'gcr.io/<project-id>/your-model-image'
    payload = {'text': text}
    
    try:
        response = requests.post(model_url, json=payload)
        translated_text = response.json().get('translated_text')
        return {"original_text": text, "translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with the translation model: {str(e)}")


"""from fastapi import FastAPI
from http import HTTPStatus
#from src.predict_model import predict
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}"""

"""class Sentence(BaseModel):
    text: str

@app.post("/translate/")
async def read_sentence(sentence: Sentence):
    prediction = predict(sentence.text)
    return {"sentence": sentence.text, "prediction": prediction}"""
