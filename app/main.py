from http import HTTPStatus
from fastapi import FastAPI
from MLOPS_final_Project.src.predict_model import predict
from pydantic import BaseModel

app = FastAPI()
class Sentence(BaseModel):
    text: str

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response



@app.post("/translate/")
async def read_sentence(sentence: Sentence):
    prediction = predict(sentence.text)
    return {"sentence": sentence.text, "prediction": prediction}



