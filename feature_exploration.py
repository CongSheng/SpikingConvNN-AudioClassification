import os
import numpy as np
import librosa
import argparse
import utils.plotFigure as plotFigure
import utils.audioProcessing as audioProcessing
from utils import spikingNeuron
import matplotlib.pyplot as plt
import torch

FILTER_NUM = 16
FILTER_ORDER = 2

def singleFileSweepMFCC(audioFile, label, sr, saveDir, maxLen, 
                    hopLenRange, NsampleRange,
                    audioSave=False):
    # Standard loading and preprocessing
    audio, _ = librosa.load(audioFile, mono=True, sr=sr)
    audio = audioProcessing.remove_silence(audio)
    if len(audio) < maxLen:
        audio = audioProcessing.pad_audio_both(audio, max_length=maxLen)
    else:
        audio = audio[:maxLen]

    if audioSave:
        wavePath = os.path.join(saveDir, "audioWave{}.png".format(label))
        plotFigure.plotWave(audio, wavePath, "MFCC", label)
    
    # Parameter sweep across MFCC
    for sampleN in NsampleRange:
        for hopLen in hopLenRange:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hopLen, n_mfcc=sampleN)
            labelMfcc = f"{label}_{hopLen}hop_{sampleN}n"
            mfccPath = os.path.join(saveDir, "mfcc_{}.png".format(labelMfcc))
            plotFigure.plotFeature(mfcc, mfccPath, label)

def singleFileSweepEnergy(audioFile, label, sr, saveDir, maxLen, 
                    hopLenRange, frameLenRange,
                    audioSave=False):
    # Standard loading and preprocessing
    audio, _ = librosa.load(audioFile, mono=True, sr=sr)
    audio = audioProcessing.remove_silence(audio)
    sosFB = audioProcessing.generateIIRFilter(sr, 
                                            FILTER_NUM, 
                                            FILTER_ORDER)
    if len(audio) < maxLen:
        audio = audioProcessing.pad_audio_both(audio, max_length=maxLen)
    else:
        audio = audio[:maxLen]

    if audioSave:
        wavePath = os.path.join(saveDir, "audioWave{}.png".format(label))
        plotFigure.plotWave(audio, wavePath, label)

    for frameWin in frameLenRange:
        for hopLen in hopLenRange:
            rmse = audioProcessing.filterEnergy(audio, sosFB, frameWin, hopLen)
            labelEnergy = f"{label}_hopLen{hopLen}_frame{frameWin}_{FILTER_NUM}filter"
            featPath = os.path.join(saveDir, "rmse_{}.png".format(labelEnergy))
            plotFigure.plotFeature(rmse, featPath, "RMSE", label)
    
def singleEnergySpike(audioFile, label, sr, saveDir, maxLen, 
                    hopLen, frameLen):
    # Standard loading and preprocessing
    audio, _ = librosa.load(audioFile, mono=True, sr=sr)
    audio = audioProcessing.remove_silence(audio)
    sosFB = audioProcessing.generateIIRFilter(sr, 
                                            FILTER_NUM, 
                                            FILTER_ORDER)
    if len(audio) < maxLen:
        audio = audioProcessing.pad_audio_both(audio, max_length=maxLen)
    else:
        audio = audio[:maxLen]

    rmse = audioProcessing.filterEnergy(audio, sosFB, frameLen, hopLen)
    labelEnergy = f"{label}_hopLen{hopLen}_frame{frameLen}_{FILTER_NUM}filter"
    featPath = os.path.join(saveDir, "rmse_{}.png".format(labelEnergy))
    plotFigure.plotFeature(rmse, featPath, "RMSE", label)

    filterN, time = rmse.shape
    plt.imshow(rmse)
    plt.colorbar()
    plt.show()
    rmse = librosa.util.normalize(rmse, axis=1)
    rmse = normalize2D(rmse, 300, 750)
    plt.imshow(rmse)
    plt.colorbar()
    plt.show()
    memHist = []
    spkHistory = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    spkLayer = spikingNeuron.LIFNeuron((1,), 1, device, tauMem=12, resetMode="rest", encodingMode="count")
    for step in range(time):
        hist, spkHist, mem, encoded = spkLayer(rmse[0,step])
        memHist.append(hist.cpu().detach().numpy())
        spkHistory.append(spkHist.cpu().detach().numpy())
    print(memHist)
    print(spkHistory)

def normalize2D(featureMap, lower, upper):
    maxValue = np.amax(featureMap)
    minValue = np.amin(featureMap)
    newRange = upper-lower
    oldRange = maxValue-minValue
    featureMap = lower + (featureMap - minValue) * (newRange/oldRange)
    return featureMap



def main(args):
    dataPath = args.dataPath
    samplingRate = args.samplingRate
    sinlgeAudioPath = args.audioName

    modeParameter = args.modePara
    figurePath = args.savePath

    label = int(sinlgeAudioPath[0])
    fullPath = os.path.join(dataPath, sinlgeAudioPath)
    audioSaveStr = args.audioSave
    if audioSaveStr == "True":
        audioSave = True
    else:
        audioSave = False

    if modeParameter == "single_mfcc_sweep":
        maxLength = int(args.maxLength * samplingRate)
        hopRange = range(args.hopRangeStart, args.hopRangeEnd, args.hopRangeStep)
        nRange = range(args.nRangeStart, args.nRangeEnd, args.nRangeStep)
        singleFileSweepMFCC(fullPath, label, samplingRate, figurePath, maxLength, hopRange, nRange, audioSave=audioSave)
    if modeParameter == "single_energy_sweep":
        maxLength = int(args.maxLength * samplingRate)
        hopRange = range(args.hopRangeStart, args.hopRangeEnd, args.hopRangeStep)
        frameRange = range(args.fRangeStart, args.fRangeEnd, args.fRangeStep)
        singleFileSweepEnergy(fullPath, label, samplingRate, figurePath, maxLength, hopRange, frameRange, audioSave=audioSave)
    if modeParameter == "single_energy_spike":
        maxLength = int(args.maxLength * samplingRate)
        hopLen = 256
        frameLen = 160
        singleEnergySpike(fullPath, label, samplingRate, figurePath, maxLength, hopLen, frameLen)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default="free-spoken-digit-dataset-v1.0.8/FSDD/recordings/", help="Path containing a single audio data")
    parser.add_argument('--savePath', type=str, default="featuresFigure/", help="Path for plots")
    parser.add_argument('--audioName', type=str, default="0_jackson_0.wav", help="Name of sinlge audio file for exploration")
    parser.add_argument('--samplingRate', type=int, default=8000, help="Sampling rate of audio file")
    parser.add_argument('--maxLength', type=float, default=0.8, help="Max time of audio file")
    parser.add_argument('--modePara', type=str, default="single_mfcc_sweep", help="Config for exploration")
    parser.add_argument('--audioSave', type=str, default="True", help="Set false to inhibit waveform printing")
    parser.add_argument('--hopRangeStart', type=int, default=128, help="Enter start of range")
    parser.add_argument('--hopRangeEnd', type=int, default=512, help="Enter end of range")
    parser.add_argument('--hopRangeStep', type=int, default=64, help="Enter step of range")
    parser.add_argument('--nRangeStart', type=int, default=2, help="Enter start of range")
    parser.add_argument('--nRangeEnd', type=int, default=32, help="Enter end of range")
    parser.add_argument('--nRangeStep', type=int, default=2, help="Enter step of range")
    parser.add_argument('--fRangeStart', type=int, default=32, help="Enter start of range")
    parser.add_argument('--fRangeEnd', type=int, default=257, help="Enter end of range")
    parser.add_argument('--fRangeStep', type=int, default=32,help="Enter step of range")
    args = parser.parse_args()

    print(args)
    main(args)