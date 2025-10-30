# take argument with data folder to train from and proceed as the streamlit app
# replace streamlit options with argument

# To be run in colab or any other environment!
import pandas as pd
from io import StringIO
from ecce2025 import main_pd
import argparse
import os

pretrained=["",
            
            "3C90",
            "3C94",
            "N87",
            "N49",
            "N30",
            "77",
            "78",
            "3F4"
            ]

# todo , add argument parsing
parser = argparse.ArgumentParser(
                    prog='Model trainer for ECCE 2025',
                    description='Trains the model from command line',
                    epilog='Author: Emmanuel Havugimana <ehavugim@asu.edu>')
parser.add_argument('folder')           # positional argument
parser.add_argument('-t', '--trainer')      # option that takes a value
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag
parser.add_argument('-p', '--patience')

path ="resampled/3F4"
files=list(os.listdir(path))
files

uploaded_files = [f"{path}/{file}" for file in files]
data={"Volumetric_losses[Wm-3].csv":"VL",
      "Temperature[C].csv":"T",
      "H_waveform[Am-1].csv":"H",
      "Frequency[Hz].csv":"F",
      "B_waveform[T].csv":"B",}
dataReady={}
for uploaded_file in uploaded_files:
    dataframe = pd.read_csv(uploaded_file,header=None)
    filename=uploaded_file.split("/")[-1]
    dataReady[data.get(filename,"x")]=dataframe
pretrain=""

trainable=False
patience=50
try:        
    x,fileout, fileout2=main_pd(dataReady["B"],dataReady['F'],
                dataReady['T'],dataReady['VL'],pretrain,
                patience=patience,base_excel="src/train/ecce-train/base.xlsx")
    print(x)
    trainable=True
except Exception as e:
    print(e)
    print("Please upload  data in same format as  \n https://github.com/ehavugi/ECCE-2025-data/tree/main/resampled/3C94")

if trainable:
    print("trained", fileout,fileout2 )