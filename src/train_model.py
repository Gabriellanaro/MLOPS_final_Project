import pickle
from models.model import model, tokenizer
import os
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import torch 
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
import hydra


# writer = SummaryWriter() #  I THINK LATER WE WILL USE WEIGHTS AND BIASES SO LATER WE WILL NEED TO REMOVE TENSORBOARD

@hydra.main(config_name="../config.yaml")
def main(config):
    print("Load data...")
    print(f'Current working directory: {os.getcwd()}')
    with open(f"{os.getcwd()}/data/processed/tokenized_train.pkl", 'rb') as f:
        tokenized_train = pickle.load(f)
    with open(f"{os.getcwd()}/data/processed/tokenized_test.pkl", 'rb') as f:
        tokenized_test = pickle.load(f)

    print("Set parameters...")
    MODEL = 't5-small'
    BATCH_SIZE = config.hp.batch_size
    NUM_PROCS = config.hp.num_procs
    EPOCHS = config.hp.epochs
    #OUT_DIR = 'results_t5small'
    MAX_LENGTH = config.hp.max_length
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Get the current timestamp
    timestamp = datetime.now()
    # Convert the timestamp to a string
    timestamp_str = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
    OUT_DIR = f"{os.getcwd()}/models"   ## However, please note that this path will be relative. If you want to use it in subsequent file operations, you might need to convert it to an absolute path using os.path.abspath(OUT_DIR)

    
    os.makedirs(f'{OUT_DIR}/{timestamp_str}', exist_ok=True)
    print(f"Results will be saved in {OUT_DIR}/{timestamp_str}")
    print("Training...")
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=config.hp.warmup_steps,
        weight_decay=config.hp.weight_decay,
        logging_dir=OUT_DIR,
        logging_steps=10,
        evaluation_strategy=config.hp.evaluation_strategy,
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        save_total_limit=5,
        report_to='tensorboard',
        learning_rate=config.hp.learning_rate,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=config.hp.dataloader_num_workers
    )

    print(torch.cuda.is_available())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )

    trainer.train()
    tokenizer.save_pretrained(f"{OUT_DIR}/{timestamp_str}")
    
    # import subprocess

    # # Define the command
    # cmd = ["tensorboard", "--logdir=runs"]

    # # Run the command
    # process = subprocess.Popen(cmd)

    # # Wait for the command to finish
    # process.wait()

if __name__ == "__main__":
    main()
