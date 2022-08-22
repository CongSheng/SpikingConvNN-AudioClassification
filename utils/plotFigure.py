import matplotlib.pyplot as plt
import numpy as np

def plotTrainingProgTwin(hist, loss, figsaveDir, figsizeIn=(10, 5)):
    x_axis = np.arange(0, len(hist), 1)
    fig, ax = plt.subplots(figsize=figsizeIn)
    ax2 = ax.twinx()
    ax.plot(x_axis, hist, 'b-o', label="Accuracy")
    ax2.plot(x_axis, loss, 'g-o', label="Loss")
    ax.set_xlabel('Epoch Number')
    ax.set_ylabel("Accuracy", color='b')
    ax2.set_ylabel("Loss", color='g')
    plt.tight_layout()
    plt.savefig(figsaveDir)
    plt.close()

def plotMfcc(mfcc, figsaveDir, label=None):
    fig, ax = plt.subplots()
    #mfcc= np.swapaxes(mfcc, 0 ,1)
    cax = ax.imshow(mfcc, interpolation='nearest', origin='lower')
    fig.colorbar(cax, ax=ax, orientation='vertical')
    
    if label is None:
        ax.set_title("MFCC")
    else:
        ax.set_title("MFCC for label \"{}\"".format(label))
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

