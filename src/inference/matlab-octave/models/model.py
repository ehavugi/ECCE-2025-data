#!/usr/bin/env python
# coding: utf-8
## Train and test loop
## 
# In[2]:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy 

#Listed in tutorial : https://mag-net.princeton.edu/

# Import necessary packages
# https://github.com/minjiechen/magnetchallenge/blob/main/tutorials/Core%20Loss/Demo_FNN_Train.ipynb
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
# import numpy as np
import datetime
import shutil
import json
import math
from sklearn import preprocessing
import sys
import os

nparams = 24


# Model run time for log folders
now = datetime.datetime.now()
nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")
logpath = "logs/{}".format(nowstr)
isExist = os.path.exists(logpath)
if not isExist:
    os.mkdir(logpath)

def MagNetStats(data,material = "3C94"):
    """
    Plot graphs that are similar to ones required by competition.
    Will need some tweaking and embedding in a pdf
    """
    plt.clf()
    plt.hist(data, weights=np.ones(len(data)) / len(data), bins=50)
    # Highlight the 95th percentile and max
    plt.axvline(np.percentile(data, 95), color='red', linestyle='dashed')
    plt.axvline(data.max(), color='red', linestyle='dashed')
    plt.axvline(data.mean(), color='red', linestyle='dashed')

    x95 = np.percentile(data, 95)
    xmax = data.max()
    labelH = plt.ylim()[1]/2
    xmean = data.mean()
    x99 = np.percentile(data, 99)
    # Annotate the lines with text

    plt.annotate('95 percentile, {:0.2f}%'.format(x95), xy=(x95,plt.ylim()[1]/2), xytext=(np.percentile(data, 95) + 0.01, labelH), color='red')
    plt.annotate('max, {:0.2f}'.format(xmax), xy=(data.max(), 0.001), xytext=(xmax + 0.01,labelH/2), color='green')
    plt.annotate('mean, {:0.2f}'.format(xmean), xy=(xmean, 0.001), xytext=(xmean + 0.01,labelH*1.8), color='green')

    # Show the plot
    plt.title("Error Distribution for {}\n Avg={:0.3f} %, 95-Pct: {:.2f}%, 99-Prc={:0.2f}%, Max = {:0.2f}%".format(material, xmean,x95,x99, xmax))
    # plt
    # plt.show()
    plt.xlabel("Relative Error of Core Loss [%]")
    plt.ylabel("Ratio of Data Points")

    return plt


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
def count_parameters(model):
    """Count number of parameters for model comparisons
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def rotate2D(y, degrees,fast=True):
    points = int(degrees/360*y.shape[1])
    array = []

    part2 = y[:, :-points]
    part1 = y[:, -points:]
    # y[:len(y)-points] = y[-points:]
    # y[points:] =y[:-points]
    if fast:
        return  np.concatenate([part1,part2], axis=1)

    for  i in range(points, len(y)+points):
        array.append(y[i-len(y)])
    return np.array(array)

def tempEncoder(T):
    """
    For consistent Temperature encoding. 
        + Ensure one encoder for validation and test training set
    **To change in future away from onehot encoding it.

    """
    T = T.values
    # Encode Temp as 4 variables
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(T)

    # 3. Transform
    Tlabels = enc.transform(T).toarray()
def get_dataset(material, baseFolder="data"):
    """Data loader for values. 
        material input (folder name): example: "N87"
        basefolder input(main folder a material folder): example: data
    """
    if "valid" in baseFolder: # format for validation dataset is different
        B = pd.read_csv(f"{baseFolder}/{material}/B_waveform.csv", header=None)
        H = pd.read_csv(f"{baseFolder}/{material}/H_Waveform.csv", header=None)
        F = pd.read_csv(f"{baseFolder}/{material}/Frequency.csv", header=None)
        VL = pd.read_csv(f"{baseFolder}/{material}/Volumetric_Loss.csv", header=None)
        T  = pd.read_csv(f"{baseFolder}/{material}/Temperature.csv", header=None)
    else:
        B = pd.read_csv(f"{baseFolder}/{material}/B_waveform[T].csv", header=None)
        H = pd.read_csv(f"{baseFolder}/{material}/H_waveform[Am-1].csv", header=None)
        F = pd.read_csv(f"{baseFolder}/{material}/Frequency[Hz].csv", header=None)
        VL = pd.read_csv(f"{baseFolder}/{material}/Volumetric_losses[Wm-3].csv", header=None)
        T  = pd.read_csv(f"{baseFolder}/{material}/Temperature[C].csv", header=None)
    Freq = F.values
    Flux  = B.values # The sign does not matter?
    Power = VL.values
    shifts = True # False if don't want rotations
    T = T.values
    # Encode Temp as 4 variables
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(T)

    # 3. Transform
    Tlabels = enc.transform(T).toarray()
    T = Tlabels
    # Compute labels
    # There's approximalely an exponential relationship between Loss-Freq and Loss-Flux. 
    # Using logarithm may help to improve the training.
    fft_data = np.fft.fft(Flux, axis=1)

    Flux = np.abs(fft_data[:, :nparams]) # Frequency decomposi
    # return to time domain
    # Flux = np.abs(np.fft.ifft(Flux))

    # uncomment for using signal space(not Freque)
    Flux =(scipy.fft.ifft(fft_data[:, :nparams],n=nparams, axis=1)) 
    Flux = scipy.signal.resample(B.values,nparams, axis=1) # enable new transform
    Fluxx = scipy.signal.resample(Flux, 1024, axis=1)
    Error = np.sum(np.abs(B.values-Fluxx),axis=1)/np.sum(np.abs(B.values),axis=1)

    Flux = np.abs(Flux)

    Freq = np.log10(Freq)
    Flux = np.log10(Flux)
    Power = np.log10(Power)
    
    # Reshape data
    Freq = Freq.reshape((-1,1))
    Flux = Flux.reshape((-1,nparams))
    T = T.reshape((-1,4))

    print(np.shape(Freq))
    print(np.shape(Flux))
    print(np.shape(T))
    print(np.shape(Power))
    if shifts:
        for phase in [40,80,160]:

            temp1 = np.concatenate((Freq,rotate2D(Flux,phase),T),axis=1)

            temp = np.concatenate((Freq,Flux,T),axis=1)

            temp = np.concatenate((temp,temp1), axis=0)
            Power = np.concatenate((Power,Power), axis=0)
            Freq = np.concatenate((Freq,Freq), axis=0)
            T = np.concatenate((T,T), axis=0)
            Flux= np.concatenate((Flux,Flux), axis=0)
            Error = np.concatenate((Error,Error), axis=0)
    else:
        temp = np.concatenate((Freq,Flux,T),axis=1)

    # log data 

    print(np.shape(Freq))
    print(np.shape(Flux))
    print(np.shape(T))
    print(np.shape(Power))
    try:
        os.mkdir('{}/{}'.format(logpath,material))
    except:
        pass
    np.savetxt('{}/{}/inputs.csv'.format(logpath, material),temp, delimiter= ", ")
    np.savetxt('{}/{}/outputs.csv'.format(logpath, material), Power, delimiter= ", ")
    np.savetxt('{}/{}/RecError.csv'.format(logpath, material), Error, delimiter= ", ")

    
    in_tensors = torch.from_numpy(temp).view(-1, nparams + 1 +T.shape[1])
    out_tensors = torch.from_numpy(Power).view(-1, 1)

    # # # Save dataset for future use
    # np.save("{}/{}/dataset.fc.in.npy".format(logpath, material), in_tensors.numpy())
    # np.save("{}/{}/dataset.fc.out.npy".format(logpath, material), out_tensors.numpy())

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)

class percentile95(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # calculate the loss
        loss =  torch.std((target-input)/abs(target))
#         loss =  torch.mean(loss)#, 0.95)

        return loss
# Config the model training

def main(material,pretrainer="N87"):
    """
    Train and test sequence for a single material. 
    """
    print("material ", material)
    global pretrain
    
    # Reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

    # Hyperparameters
    NUM_EPOCH = 1000
    BATCH_SIZE = 128
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.5
    LR_INI = 0.025

    # Select GPU as default device
#     device = torch.device("cuda")
    device = torch.device("cpu")

    # Load dataset
    dataset = None
    dataset = get_dataset(material)
    # Split the dataset
    train_size = int(0.6* len(dataset))
    # valid_size = len(dataset) - train_size
    valid_size = int(0.2* len(dataset))

    test_size = len(dataset) - train_size - valid_size ## to be loaded seperated
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
    # test_dataset = get_dataset(material, "valid")  # Note: the data will be used for testing as provided in magnet Nov1 update
    kwargs = {'num_workers': 0, 'pin_memory_device': "cpu"}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network
    # It was found starting new model with previous model weight can help less data materials
    
    net = Net().double().to(device) # to avoid reseting parameters?, can be commented out
    pretrainedModel = f"pretrained/Model{pretrainer}.sd"
    if os.path.isfile(pretrainedModel) and pretrain==True:
        net.load_state_dict(torch.load(pretrainedModel)) # Model parameter initialize
        net.eval()
        # i = 0
        # print(net.features)
        print(net.parameters())
        print(net)
        i = 0
        for param in net.parameters():
            if i<1:
                param.requires_grad = True
            print(i, param.numel())
            i += 1
        param.requires_grad = True
        print("Loaded Master net")
    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer
    
    
    criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    # criterion = percentile95()
#     criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 
    # Train the network
    oldEpoch_loss = 10e8
    for epoch_i in range(NUM_EPOCH):

        # Train for one epoch
        epoch_train_loss = 0
        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI* (DECAY_RATIO ** (0+ epoch_i // DECAY_EPOCH))

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # Compute Validation Loss
        with torch.no_grad():
            epoch_valid_loss = 0
            for inputs, labels in valid_loader:
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))

                epoch_valid_loss += loss.item()
        
        if (epoch_i+1)%50 == 0:
          # print(f"Epoch {epoch_i+1:2d} "
          #       f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
          #       f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f}")
          newEpoch_loss = epoch_valid_loss / len(valid_dataset)
        
          change = np.abs((newEpoch_loss-oldEpoch_loss)/oldEpoch_loss)*100
          oldEpoch_loss = newEpoch_loss
          print(f"Epoch {epoch_i+1:2d} "
                f"Train {epoch_train_loss / len(train_dataset) * 1e5:.5f} "
                f"Valid {epoch_valid_loss / len(valid_dataset) * 1e5:.5f} " 
                f"Change {change:.2f}")
          # if change<0.1:
          #     break
    # Save the model parameters
    torch.save(net.state_dict(),f"{logpath}/Model{material}.sd")
    print("Training finished! Model is saved!")

    # Evaluation
    net.eval()
    y_meas = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            y_pred.append(net(inputs.to(device)))
            y_meas.append(labels.to(device))

    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    print(f"Test Loss: {F.mse_loss(y_meas, y_pred).item() / len(test_dataset) * 1e5:.5f}")

    yy_pred = 10**(y_pred.cpu().numpy())
    yy_meas = 10**(y_meas.cpu().numpy())
    # yy_pred = (y_pred.cpu().numpy())
    # yy_meas = (y_meas.cpu().numpy())
    # Relative Error
    Error_re = abs(yy_pred-yy_meas)/abs(yy_meas)*100
    Error_re_avg = np.mean(Error_re)
    percentile95_error = np.percentile(Error_re,95)
    Error_re_rms = np.sqrt(np.mean(Error_re ** 2))
    Error_re_max = np.max(Error_re)
    print(f"Relative Error: {Error_re_avg:.8f}")
    print(f"RMS Error: {Error_re_rms:.8f}")
    print(f"MAX Error: {Error_re_max:.8f}")
    print(f"95 % error Error: {percentile95_error:.8f}")
    plotx = MagNetStats(Error_re, material)
    plotx.savefig('{}/{}/Error.png'.format(logpath, material))

    dataFrame = {"meas":yy_meas.flatten(), "pred":yy_pred.flatten(),"error":Error_re.flatten()}

    data = pd.DataFrame(dataFrame)
    ## Reduce data size of log
    data['meas'] = data['meas'].astype('int')
    data['pred'] = data['pred'].astype('int')
    data['error'] = data['error'].round(1)
    data.to_csv("{}/{}_last_stats.csv".format(logpath,material), index=False)
    return data
def modelRUN():
    global net
    global pretrain
    global logpath
    """
    Loop through all materials. 
        1. call main for each material
    """

    old_stdout = sys.stdout

    log_file = open("{}/{}.csv".format(logpath,"logs.log"),"a+")

    # sys.stdout = log_file

    

    now = datetime.datetime.now()
    print(now)

    materials = ["3E6", "3F4","77","78","N27", "N30","N49", "N87", "3C90", "3C94"]
    pretrained_materials = ["N87","3E6", "3F4","77","78","N27", "N30","N49", "3C90", "3C94"]
    pretrained_materials = ["Nxx"]
    # materials = "ABCDE0"
    Outside = False  #If true I would  be doing pre-training or transfer learning

    # Config the model training
    device = torch.device("cpu")
    data=None


    pretrain = False

    for pretrainer in pretrained_materials:
        net = Net().double().to(device)
        torch.set_num_threads(10)
        net = torch.nn.DataParallel(net)
        # net = Net().double().to(device)

        # nowstr = now.strftime("%Y-%m-%d_%H-%M-%S")
        logpath = "logs/{}/{}".format(nowstr, pretrainer)
        isExist = os.path.exists(logpath)
        if not isExist:
            os.mkdir(logpath)
        for material in materials:
                # if ~Outside:
            #     net = Net().double().to(device)
        
            main(material,pretrainer)
            print(materials)
    

if __name__ == "__main__":
    shutil.copyfile(__file__, '{}/model.py'.format(logpath))
    modelRUN()
