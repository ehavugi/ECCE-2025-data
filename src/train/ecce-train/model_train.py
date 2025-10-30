import streamlit as st
import pandas as pd
from io import StringIO
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
pretrain = st.selectbox(
            f'pretrained-models(a model to transfer learn from):',
            pretrained,
            index=1,
            key=f'material {pretrained}',
            help='select from a list of pretrained  models')
patience= st.slider("patience \n (Number of epochs to wait without improvements)", 0,1000, 50)

from ecce2025 import main_pd
uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)

data={"Volumetric_losses[Wm-3].csv":"VL",
      "Temperature[C].csv":"T",
      "H_waveform[Am-1].csv":"H",
      "Frequency[Hz].csv":"F",
      "B_waveform[T].csv":"B",}
dataReady={}
for uploaded_file in uploaded_files:
    dataframe = pd.read_csv(uploaded_file,header=None)
    filename=uploaded_file.name
    dataReady[data.get(filename,"x")]=dataframe

print(pretrain)
st.write(pretrain)
trainable=False
try:
        
    x,fileout, fileout2=main_pd(dataReady["B"],dataReady['F'],
                dataReady['T'],dataReady['VL'],pretrain,
                patience=patience,base_excel="src/train/ecce-train/base.xlsx")
    st.write(x)
    trainable=True
except Exception as e:
    st.write("Please upload  data in same format as  \n https://github.com/ehavugi/ECCE-2025-data/tree/main/resampled/3C94")


if trainable:
    try:
        with open(fileout, "rb") as fp:
            btn = st.download_button(
                label="Download trained model (SD)",
                data=fp,
                file_name="model.sd" # Any file name
            )
    except:
        st.write("Error while generating model.sd")
    try:
        with open(fileout2, "rb") as fp:
            btn = st.download_button(
                label="Download trained model(XLSX)",
                data=fp,
                file_name=fileout2 # Any file name
            )
            # st.download_button('Download file', x)  # Defaults to 'application/octet-stream'
    except:
        st.write("Error while generating xlsx model format")
    #     st.write("Upload data:  ", data)