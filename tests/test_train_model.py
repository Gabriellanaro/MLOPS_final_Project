import pytest
import torch
from datasets import Dataset
import sys
sys.path.insert(0, 'C:/Users/Usuario/dtu/mlopsproj/MLOPS_final_Project/src')

from models.train_model import train_model

def test_train_model():
    # Load a dataset
    dataset = Dataset.from_dict({'text': ['Lisa Kristine: Billeder der bærer vidne til moderne slaveri.']})

    # Train a model
    model = train_model(dataset)

    # Check if the model is trained
    # This could be done by checking if the model's parameters have changed after training,
    # or by checking if the model can make predictions without raising an exception.
    try:
        model(dataset['text'])
        print("Test correctly passed :)")
    except Exception as e:
        pytest.fail(f"Model prediction raised an exception: {e}")