def get_dataset(material, baseFolder="training"):
    """Data loader for values.
        material input (folder name): example: "N87"
        basefolder input(main folder a material folder): example: data
    """
    resampled=False
    if "resampled" in baseFolder: # format for resampled dataset is different
        resampled=True
        B = pd.read_csv(f"{baseFolder}/{material}/B_waveform[T].csv", header=None)
        H = pd.read_csv(f"{baseFolder}/{material}/H_waveform[Am-1].csv", header=None)
        F = pd.read_csv(f"{baseFolder}/{material}/Frequency[Hz].csv", header=None)
        VL = pd.read_csv(f"{baseFolder}/{material}/Volumetric_losses[Wm-3].csv", header=None)
        T  = pd.read_csv(f"{baseFolder}/{material}/Temperature[C].csv", header=None)

    elif "valid" in baseFolder: # format for validation dataset is different
        B = pd.read_csv(f"{baseFolder}/{material}/B_waveform.csv", header=None)
        H = pd.read_csv(f"{baseFolder}/{material}/H_Waveform.csv", header=None)
        F = pd.read_csv(f"{baseFolder}/{material}/Frequency.csv", header=None)
        VL = pd.read_csv(f"{baseFolder}/{material}/Volumetric_Loss.csv", header=None)
        T  = pd.read_csv(f"{baseFolder}/{material}/Temperature.csv", header=None)
    else:
        B = pd.read_csv(f"{baseFolder}/Material {material}/B_Field.csv", header=None)
        H = pd.read_csv(f"{baseFolder}/Material {material}/H_Field.csv", header=None)
        F = pd.read_csv(f"{baseFolder}/Material {material}/Frequency.csv", header=None)
        VL = pd.read_csv(f"{baseFolder}/Material {material}/Volumetric_Loss.csv", header=None)
        T  = pd.read_csv(f"{baseFolder}/Material {material}/Temperature.csv", header=None)

    Freq = F.values
    Flux  = B.values*1024/24
    Power = VL.values
    shifts = False
    T = T.values
    
    # Encode Temp as 4 variables
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(T)

    # 3. Transform
    Tlabels = enc.transform(T).toarray()
    T = Tlabels
    # Compute labels
    if not resampled:
        fft_data = np.fft.fft(Flux, axis=1)
        Flux = scipy.signal.resample(B.values,nparams, axis=1)
        Flux_upsampled = scipy.signal.resample(Flux, 1024, axis=1)
        Error = np.sum(np.abs(B.values-Fluxx),axis=1)/np.sum(np.abs(B.values),axis=1)

    Flux = np.abs(Flux)
    Freq = np.log10(Freq)
    Flux = np.log10(Flux)
    Power = np.log10(Power)

    # Reshape data
    Freq = Freq.reshape((-1,1))
    Flux = Flux.reshape((-1,nparams))
    T = T.reshape((-1,4))

    printe(np.shape(Freq))
    printe(np.shape(Flux))
    printe(np.shape(T))
    printe(np.shape(Power))
    if shifts:
        for phase in [40,80,160]:

            temp1 = np.concatenate((Freq,rotate2D(Flux,phase),T),axis=1)

            temp = np.concatenate((Freq,Flux,T),axis=1)

            temp = np.concatenate((temp,temp1), axis=0)
            Power = np.concatenate((Power,Power), axis=0)
            Freq = np.concatenate((Freq,Freq), axis=0)
            T = np.concatenate((T,T), axis=0)
            Flux= np.concatenate((Flux,Flux), axis=0)
    else:
        temp = np.concatenate((Freq,Flux,T),axis=1)

    # log data
    try:
        os.mkdir('{}/{}'.format(logpath,material))
    except:
        pass
    np.savetxt('{}/{}/inputs.csv'.format(logpath, material),temp, delimiter= ", ")
    np.savetxt('{}/{}/outputs.csv'.format(logpath, material), Power, delimiter= ", ")
    if not resampled:
      np.savetxt('{}/{}/RecError.csv'.format(logpath, material), Error, delimiter= ", ")

    in_tensors = torch.from_numpy(temp).view(-1, nparams + 1 +T.shape[1])
    out_tensors = torch.from_numpy(Power).view(-1, 1)

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)
