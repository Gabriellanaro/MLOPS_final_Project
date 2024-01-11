import pickle
from datasets import Dataset
import sys
# from tests import _PATH_DATA
import pandas as pd
import os

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_path, ".."))
from src.data.make_dataset import preprocess_function

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
print(_PATH_DATA)

def test_preprocess_function():
    # Load the datasets
    print(f"Current working directory: {os.getcwd()}")
    print(f"Full path to Data folder: {_PATH_DATA}")
    print(f"Contents of {_PATH_DATA}/raw/: {os.listdir(os.path.join(_PATH_DATA, 'raw'))}")
    with open(f'{_PATH_DATA}/raw/train_set.pkl', 'rb') as f:
        train_set = pickle.load(f)
    with open(f'{_PATH_DATA}/raw/test_set.pkl', 'rb') as f:
        test_set = pickle.load(f)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_set))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_set))
    # Apply the preprocess_function
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)
    print(_PATH_DATA)
    # Load the preprocessed data saved by make_dataset.py
    with open(f'{_PATH_DATA}/processed/tokenized_train.pkl', 'rb') as f:
        saved_tokenized_train = pickle.load(f)
    with open(f'{_PATH_DATA}/processed/tokenized_test.pkl', 'rb') as f:
        saved_tokenized_test = pickle.load(f)

    # Convert the Datasets to lists
    list_tokenized_train = list(tokenized_train)
    list_saved_tokenized_train = list(saved_tokenized_train)
    list_tokenized_test = list(tokenized_test)
    list_saved_tokenized_test = list(saved_tokenized_test)

    # Check if the results match
    assert list_tokenized_train == list_saved_tokenized_train
    assert list_tokenized_test == list_saved_tokenized_test

    print("Test passed :)")