from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import pickle
import os
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import pandas as pd

MODEL = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(MODEL)

def preprocess_function(examples):
    inputs = [f"{da}" for da in examples['da']]
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding='max_length'
    )

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['en'],
            max_length=512,
            truncation=True,
            padding='max_length'
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    print("I am executing make_dataset.py")
    dataset = load_dataset(
        "ted_talks_iwslt", language_pair=("da", "en"), year="2014", trust_remote_code=True
    )
    data = dataset["train"]["translation"]
    print(type(data))
    print(f'Current working directory: {os.getcwd()}')
    # Open a file in binary write mode ('wb')
    with open('data/raw/data.pkl', 'wb') as f:
        # Use pickle.dump() to serialize and save the raw data
        pickle.dump(data, f)
    
    # Split the 'train_data' into train and test sets
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    # Check the lengths of train and test sets
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    with open('data/raw/train_set.pkl', 'wb') as f:
        # Use pickle.dump() to serialize and save the raw data
        pickle.dump(train_set, f)
    with open('data/raw/test_set.pkl', 'wb') as f:
        # Use pickle.dump() to serialize and save the raw data
        pickle.dump(test_set, f)

    # Process data to obtain model inputs
    # Convert the lists to Dataset objects
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_set))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_set))

    # Apply the preprocess_function
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    # Save preprocessed data
    with open('data/processed/tokenized_train.pkl', 'wb') as f:
        pickle.dump(tokenized_train, f)
    with open('data/processed/tokenized_test.pkl', 'wb') as f:
        pickle.dump(tokenized_test, f)


    
