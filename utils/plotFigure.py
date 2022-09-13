import matplotlib.pyplot as plt
import numpy as np
import os

def plotTrainingProgTwin(hist, loss, figsaveDir, iterCount=0, 
                        figsizeIn=(10, 5), spiking=False):
    fig, ax = plt.subplots(figsize=figsizeIn)
    ax2 = ax.twinx()
    
    if spiking:
        x_axis = np.arange(1, len(hist)/iterCount+1, 1)
        ax.plot(x_axis, hist[::iterCount], 'b-o', label="Accuracy")
        ax2.plot(x_axis, loss[::iterCount], 'g-o', label="Loss")
        ax.set_xlabel("Epoch")
    else:
        x_axis = np.arange(1, len(hist)+1, 1)
        ax.plot(x_axis, hist, 'b-o', label="Accuracy")
        ax2.plot(x_axis, loss, 'g-o', label="Loss")
        ax.set_xlabel('Epoch Number')

    ax.set_ylabel("Accuracy", color='b')
    ax2.set_ylabel("Loss", color='g')
    plt.tight_layout()
    plt.savefig(figsaveDir)
    plt.close()

def plotFeature(feature, figsaveDir, featureName="MFCC", label=None):
    fig, ax = plt.subplots()
    #mfcc= np.swapaxes(mfcc, 0 ,1)
    cax = ax.imshow(feature, interpolation='nearest', origin='lower')
    fig.colorbar(cax, ax=ax, orientation='vertical')
    
    if label is None:
        ax.set_title(featureName)
    else:
        ax.set_title("{} for label \"{}\"".format(featureName, label))
    plt.tight_layout()
    plt.savefig(figsaveDir)
    plt.close()

def plotWave(wave, figsaveDir, label=None):
    fig, ax = plt.subplots()
    ax.plot(wave)
    if label is None:
        ax.set_title("Waveform")
    else:
        ax.set_title("Waveform for label \"{}\"".format(label))
        plt.tight_layout()
        plt.savefig(figsaveDir)
    plt.close()

def plotFilter(layerWeight, figName, figPath, nnMode):
    fig = plt.figure(figsize=(10, 8))
    plt.rcParams['font.size'] = '16'
    for filter in range(layerWeight.shape[0]):
        kernelWeight = np.squeeze(layerWeight[filter], axis=0)
        xlim, ylim = kernelWeight.shape
        plt.imshow(kernelWeight, cmap='hot', vmin=0, vmax=1, extent=(0, xlim, 0, ylim))
        plt.colorbar()
        plt.savefig(os.path.join(figPath, f"{nnMode}_{figName}filter{filter}_weights.png"))
        plt.clf()
    return


