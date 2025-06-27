import numpy as np
import scipy
import os
def resample_save(material, baseFolder="data",outfolder="resampled"):
    """Data loader for values. 
        material input (folder name): example: "N87"
        basefolder input(main folder a material folder): example: data
    """
    if "valid" in baseFolder: # format for validation dataset is different
        B = np.genfromtxt(f"{baseFolder}/{material}/{material}/B_waveform.csv",delimiter=",")
        H = np.genfromtxt(f"{baseFolder}/{material}/{material}/H_Waveform.csv",delimiter=",")
        F = np.genfromtxt(f"{baseFolder}/{material}/{material}/Frequency.csv",delimiter= ",")
        VL = np.genfromtxt(f"{baseFolder}/{material}/{material}/Volumetric_Loss.csv",delimiter= ",")
        T  = np.genfromtxt(f"{baseFolder}/{material}/{material}/Temperature.csv",delimiter=",")
    else:
        B = np.genfromtxt(f"{baseFolder}/{material}/B_waveform[T].csv",delimiter=",")
        H = np.genfromtxt(f"{baseFolder}/{material}/H_waveform[Am-1].csv",delimiter=",")
        F = np.genfromtxt(f"{baseFolder}/{material}/Frequency[Hz].csv", delimiter=",")
        VL = np.genfromtxt(f"{baseFolder}/{material}/Volumetric_losses[Wm-3].csv",delimiter=",")
        T  = np.genfromtxt(f"{baseFolder}/{material}/Temperature[C].csv",delimiter=",")
    data_F=F
    data_T=T
    data_B=B
    data_H=H
    length = len(data_F)
    nparams = 24

    B = scipy.signal.resample(B,nparams, axis=1)
    H= scipy.signal.resample(H,nparams, axis=1)
    base_outfolder=f"{outfolder}/{material}"
    if os.path.exists(base_outfolder):
        pass
    else:
        os.mkdir( base_outfolder)
    np.savetxt(f'{base_outfolder}/H_waveform[Am-1].csv', H, delimiter=',')
    np.savetxt(f'{base_outfolder}/B_waveform[T].csv', B, delimiter=',')
    np.savetxt(f'{base_outfolder}/Temperature[C].csv', T, delimiter=',')
    np.savetxt(f'{base_outfolder}/Volumetric_losses[Wm-3].csv', VL, delimiter=',')
    np.savetxt(f'{base_outfolder}/Frequency[Hz].csv', F, delimiter=',')

    print("passed")

if __name__=="__main__":
    print("hello")  
    materials = ["N87","3E6", "3F4","77","78","N27", "N30","N49", "3C90", "3C94"]
    for material in materials:
        resample_save(material, baseFolder="C:/Users/ehavugim/Downloads/ASUMagNet2023/data",outfolder="resampled")
