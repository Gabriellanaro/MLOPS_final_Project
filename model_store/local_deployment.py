from transformers import T5ForConditionalGeneration
import torch

# Specify the path to the saved model
saved_model_path = "/MLOPS_final_Project/outputs/2024-01-12/13-40-09/models/2024_01_12_13_40_09/model"

# Load the model
model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

