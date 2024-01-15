from transformers import T5ForConditionalGeneration
import torch
import os

# Specify the path to the saved model
print(os.getcwd())
saved_model_path = "./outputs/2024-01-12/13-40-09/models/2024_01_12_13_40_09/model"

# Load the model
model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
torch.save(model.state_dict(), f"{saved_model_path}/checkpoint.pt")
torch.save(model.state_dict(), f"{saved_model_path}/checkpoint.pt")
# script_model = torch.jit.script(model)
# script_model.save('deployable_model.pt')

