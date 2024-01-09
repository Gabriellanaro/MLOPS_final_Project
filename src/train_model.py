import pickle
from src.models.model import model, tokenizer
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

# writer = SummaryWriter() #  I THINK LATER WE WILL USE WEIGHTS AND BIASES SO LATER WE WILL NEED TO REMOVE TENSORBOARD

if __name__ == "__main__":
    print("Load data...")
    print(f'Current working directory: {os.getcwd()}')
    with open(f"{os.getcwd()}/data/processed/tokenized_train.pkl", 'rb') as f:
        tokenized_train = pickle.load(f)
    with open(f"{os.getcwd()}/data/processed/tokenized_test.pkl", 'rb') as f:
        tokenized_test = pickle.load(f)

    print("Set parameters...")
    MODEL = 't5-small'
    BATCH_SIZE = 16
    NUM_PROCS = 16
    EPOCHS = 15
    #OUT_DIR = 'results_t5small'
    MAX_LENGTH = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Get the current timestamp
    timestamp = datetime.now()
    # Convert the timestamp to a string
    timestamp_str = timestamp.strftime("%Y-%m-%d-%H:%M:%S")
    OUT_DIR = f"{os.getcwd()}/models"   ## However, please note that this path will be relative. If you want to use it in subsequent file operations, you might need to convert it to an absolute path using os.path.abspath(OUT_DIR)

    
    os.makedirs(f'{OUT_DIR}/{timestamp_str}', exist_ok=True)
    print(f"Results will be saved in {OUT_DIR}/{timestamp_str}")
    print("Training...")
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=OUT_DIR,
        logging_steps=10,
        evaluation_strategy='steps',
        save_steps=500,
        eval_steps=500,
        load_best_model_at_end=True,
        save_total_limit=5,
        report_to='tensorboard',
        learning_rate=0.0001,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4
    )

    print(torch.cuda.is_available())

# Set up the profiler using torch.profiler.profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{OUT_DIR}/profiling_logs")
) as prof:
    print("Training...")
    # Place your model training code here using Trainer or other methods
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

