from transformers import T5ForConditionalGeneration
import torch
import os

# Specify the path to the saved model
print(os.getcwd())
saved_model_path = "./outputs/model.pt"

# Load the model
MODEL = 't5-small'
model = T5ForConditionalGeneration.from_pretrained(MODEL)
state_dict = torch.load(saved_model_path)
#print(state_dict.keys())
model.load_state_dict(state_dict)
# model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
# torch.save(model.state_dict(), f"{saved_model_path}/checkpoint.pt")
# torch.save(model.state_dict(), f"{saved_model_path}/checkpoint.pt")
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

