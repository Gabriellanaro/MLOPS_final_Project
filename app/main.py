from fastapi import FastAPI
from http import HTTPStatus
#from src.predict_model import predict
from pydantic import BaseModel

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
