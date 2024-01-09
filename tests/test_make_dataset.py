import pickle
from datasets import Dataset
import sys
sys.path.insert(0, 'C:/Users/Usuario/dtu/mlopsproj/MLOPS_final_Project/src')

from data.make_dataset import preprocess_function

def test_preprocess_function():
    # Load the datasets
    with open('data/processed/train_set.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('data/processed/test_set.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    # Apply the preprocess_function
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Load the preprocessed data saved by make_dataset.py
    with open('data/processed/tokenized_train.pkl', 'rb') as f:
        saved_tokenized_train = pickle.load(f)
    with open('data/processed/tokenized_test.pkl', 'rb') as f:
        saved_tokenized_test = pickle.load(f)

    # Check if the results match
    assert tokenized_train == saved_tokenized_train
    assert tokenized_test == saved_tokenized_test