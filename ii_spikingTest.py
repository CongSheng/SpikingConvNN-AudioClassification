from utils import spikingNeuron

import matplotlib.pyplot as plt
import torch
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

def plotMultiSpikes(memVoltage, spkTrain, figSize=(20,12), figPath="featuresFigure/"):
    timesteps, nrow, ncol = memVoltage.shape
    nSubplots = nrow * ncol
    memVoltage = memVoltage.reshape(timesteps, nSubplots)
    print(f"Edited shape: {memVoltage.shape}")
    spkTrain = spkTrain.reshape(timesteps, nSubplots)
    fig, axs = plt.subplots(nSubplots, figsize=figSize)
    for i in range(nSubplots):
        axs[i].plot(memVoltage[:,i])
        axs[i].set_xlabel('Timesteps')
        axs[i].set_ylabel('Membrane voltage (mV)')
        ax2 = axs[i].twinx()
        ax2.set_ylabel("Spikes")
        ax2.set_ylim(0, 1)
        ax2.plot(spkTrain[:,i], color = 'red')
    plt.show()
    plt.savefig(figPath)    
    return

def plotSingleNeuron(memVoltage, spkTrain,figSize=(20,12)):
    timesteps, nSubplots = memVoltage.shape
    memVoltage = memVoltage.reshape(timesteps, nSubplots)
    spkTrain = spkTrain.reshape(timesteps, nSubplots)
    fig, ax = plt.subplots(figsize= figSize)
    ax.plot(memVoltage)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Membrane voltage (mV)')
    ax2 = ax.twinx()
    ax2.set_ylabel("Spikes")
    ax2.set_ylim(0, 1)
    ax2.plot(spkTrain, color = 'red')
    plt.xticks(np.arange(0, timesteps+1, 2.0))
    ax.tick_params(axis='x', rotation=45)
    plt.show()
    return

if __name__ == '__main__':
    inputShape = (1,)
    x = torch.tensor([220], device=device)
    print(f"Tensor to inject: \n{x}")
    spkLayer = spikingNeuron.LIFNeuron(inputShape, 100, device, tauMem=12, resetMode="rest", encodingMode="count")
    hist, spkHist, mem, encoded = spkLayer(x)

    memVolt = hist.cpu().detach().numpy()
    spkHist = spkHist.cpu().detach().numpy()
    #plotMultiSpikes(memVolt, spkHist, figPath = "featuresFigure/Test.png")
    plotSingleNeuron(memVolt, spkHist)
    print(encoded)

