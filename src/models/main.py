import os
import torch
from fastapi import FastAPI
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer

#monitoring
from prometheus_fastapi_instrumentator import Instrumentator, metrics

app = FastAPI()

MODEL = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL, return_dict=True)


# just verifying
@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/translate")
def translate_text(text: str):
    input_ids = tokenizer("translate English to French: " + text, return_tensors="pt").input_ids  # Batch size 1

    outputs = model.generate(input_ids)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"translated_text": decoded}

# Track specific metrics
# metrics.track_requests(app)
# metrics.track_exceptions(app)
# metrics.track_latency(app)

Instrumentator().instrument(app).expose(app)


# if __name__ == "__main__":
#     # uvicorn.run(app, host="127.0.0.1", port=8000)
