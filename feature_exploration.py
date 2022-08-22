import os
from xml.etree.ElementInclude import default_loader
import numpy as np
import librosa
import argparse
import utils.plotFigure as plotFigure
import utils.audioProcessing as audioProcessing

def singleFileSweep(audioFile, label, sr, saveDir, maxLen, 
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
        plotFigure.plotWave(audio, wavePath, label)
    
    # Parameter sweep across MFCC
    for sampleN in NsampleRange:
        for hopLen in hopLenRange:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hopLen, n_mfcc=sampleN)
            labelMfcc = f"{label}_{hopLen}hop_{sampleN}n"
            mfccPath = os.path.join(saveDir, "mfcc_{}.png".format(labelMfcc))
            plotFigure.plotMfcc(mfcc, mfccPath, label)

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
        singleFileSweep(fullPath, label, samplingRate, figurePath, maxLength, hopRange, nRange, audioSave=audioSave)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataPath', type=str, default="free-spoken-digit-dataset-v1.0.8/FSDD/recordings/", help="Path containing a single audio data")
    parser.add_argument('--savePath', type=str, default="featuresFigure/", help="Path for plots")
    parser.add_argument('--audioName', type=str, default="0_jackson_0.wav", help="Name of sinlge audio file for exploration")
    parser.add_argument('--samplingRate', type=int, default=8000, help="Sampling rate of audio file")
    parser.add_argument('--maxLength', type=float, default=0.8, help="Max time of audio file")
    parser.add_argument('--modePara', type=str, default="single_mfcc_sweep", help="Config for exploration")
    parser.add_argument('--audioSave', type=str, default="True", help="Set false to inhibit waveform printing")
    parser.add_argument('--hopRangeStart', type=int, help="Enter start of range")
    parser.add_argument('--hopRangeEnd', type=int, help="Enter end of range")
    parser.add_argument('--hopRangeStep', type=int, help="Enter step of range")
    parser.add_argument('--nRangeStart', type=int, help="Enter start of range")
    parser.add_argument('--nRangeEnd', type=int, help="Enter end of range")
    parser.add_argument('--nRangeStep', type=int, help="Enter step of range")
    args = parser.parse_args()

    print(args)
    main(args)