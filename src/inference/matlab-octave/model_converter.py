import torch
import torch.nn as nn
import os
import numpy as np
import json

nparams=24
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define a fully connected layers model with three inputs (frequency,T, Bfft(complex numbers)) and one output (power loss).
        self.layers = nn.Sequential(
            nn.Linear(nparams+5,nparams+5),
            nn.ReLU(),
            nn.Linear(nparams+5,15),
            nn.ReLU(),
            nn.Linear(15,15),
            nn.ReLU(),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        """
        """
        return self.layers(x)


net = Net()

def model_converter(filename):
    """ Converter the model into json that can be loaded in matrix.
        Pending ---> cross validation with other implementations
    """
    pretrainedModel =filename
    base_filename=".".join(filename.split(".")[:-1])
    if os.path.isfile(pretrainedModel):
        state_dict=torch.load(pretrainedModel)
        state_dict_={}
        for key in state_dict.keys():
            key_new=key.replace("module.","")
            state_dict_[key_new]=state_dict[key].numpy().tolist()
        file_path = f"{base_filename}.json"
        with open(file_path, "w") as file:
            json.dump(state_dict_, file)

materials = ["3E6", "3F4","77","78","N27", "N30","N49", "N87", "3C90", "3C94","A","B","C","D","E"]

for material in materials:
    model_converter(f"models/Model{material}.sd")
    print(f"converted {material}")
print("done")