import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import utils.audioProcessing as audioProcessing

class MFCCDataset(Dataset):
    def __init__(self, data_path, sample_rate, max_shape,
                hop_length=512, n_samples=16, channel_in=1, max_length=None):
        print("-----MFCC Dataset creation-----\n")
        self.data_path = str(data_path)
        self.data = []
        self.labels = []
        for src_file in os.listdir(data_path):
            full_path = os.path.join(data_path, src_file)
            audio, _ = librosa.load(full_path, mono=True, sr=sample_rate)
            # Remove silence
            audio = audioProcessing.remove_silence(audio)
            # Trimming or padding
            if max_length is not None:
                if len(audio) < max_length:
                    audio = audioProcessing.pad_audio_both(audio, 
                                                            max_length=max_length)
                else:
                    audio = audio[:max_length]

            mfcc = librosa.feature.mfcc(y=audio, 
                                        sr=sample_rate, 
                                        hop_length=hop_length, 
                                        n_mfcc=n_samples)

            #mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
            mfcc = torch.from_numpy(mfcc).float()
             # TODO (Cong Sheng): Refactor for flexibility
            if channel_in==1: 
                mfcc = self.padMFCC(mfcc, max_shape)
                mfcc = torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))
            if channel_in==2:
                mfcc = self.padMFCC(mfcc, max_shape)
                mfcc = torch.stack([mfcc, mfcc], dim=0)
            if channel_in==3:
                mfcc = self.padMFCC(mfcc, max_shape)
                mfcc = torch.stack([mfcc, mfcc, mfcc], dim=0)

            self.data.append(mfcc)
            self.labels.append(int(src_file[0]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index]
        return [X, Y]
    
    def padMFCC(self, mfcc, shapeDesired, padMode='constant', valuePad=0):
        lenMFCC, widthMFCC = mfcc.shape[0], mfcc.shape[1]
        (lenGoal, widthGoal) = shapeDesired
        assert lenMFCC <= lenGoal or widthMFCC <= widthGoal, "MFCC too large, consider reducing n_mfcc or increasing hop length"
        rightPad = int((widthGoal - widthMFCC)/2)
        topPad = int((lenGoal - lenMFCC)/2)
        leftPad = widthGoal - widthMFCC - rightPad
        btmPad = lenGoal - lenMFCC - topPad
        padDim = (leftPad, rightPad, topPad, btmPad)
        mfccGoal = F.pad(mfcc, padDim, mode=padMode, value=valuePad)
        return mfccGoal

class MFCCDatasetv2(Dataset):
    def __init__(self, data_path, sample_rate, max_shape,
                hop_length=512, n_samples=16, channel_in=1, max_length=None):
        print("-----MFCC Dataset creation-----\n")
        self.data_path = str(data_path)
        self.data = []
        self.labels = []
        self.labelList = sorted(os.listdir(data_path))
        print(f"Label List: {self.labelList}")
        for label in tqdm(os.listdir(data_path)):
            if "_" in label:
                continue
            sub_path = os.path.join(data_path, label)
            if os.path.isfile(sub_path):
                continue
            print(f"\nReading from subpath: {sub_path}")
            for fileName in os.listdir(sub_path):
                fileName = os.path.join(sub_path, fileName)
                audio, _ = librosa.load(fileName, mono=True, sr=sample_rate)
                # Remove silence
                audio = audioProcessing.remove_silence(audio)
                # Trimming or padding
                if max_length is not None:
                    if len(audio) < max_length:
                        audio = audioProcessing.pad_audio_both(audio, 
                                                                max_length=max_length)
                    else:
                        audio = audio[:max_length]
                mfcc = librosa.feature.mfcc(y=audio, 
                                            sr=sample_rate, 
                                            hop_length=hop_length, 
                                            n_mfcc=n_samples)
                #mfcc = mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1])
                mfcc = torch.from_numpy(mfcc).float()
                # TODO (Cong Sheng): Refactor for flexibility
                if channel_in==1: 
                    mfcc = self.padMFCC(mfcc, max_shape)
                    mfcc = torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))
                if channel_in==2:
                    mfcc = self.padMFCC(mfcc, max_shape)
                    mfcc = torch.stack([mfcc, mfcc], dim=0)
                if channel_in==3:
                    mfcc = self.padMFCC(mfcc, max_shape)
                    mfcc = torch.stack([mfcc, mfcc, mfcc], dim=0)
                self.data.append(mfcc)
                self.labels.append(self.labelList.index(label))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index]
        return [X, Y]
    
    def padMFCC(self, mfcc, shapeDesired, padMode='constant', valuePad=0):
        lenMFCC, widthMFCC = mfcc.shape[0], mfcc.shape[1]
        (lenGoal, widthGoal) = shapeDesired
        assert lenMFCC <= lenGoal and widthMFCC <= widthGoal, f"MFCC too large {lenMFCC} x {widthMFCC}, consider reducing n_mfcc or increasing hop length"
        rightPad = int((widthGoal - widthMFCC)/2)
        topPad = int((lenGoal - lenMFCC)/2)
        leftPad = widthGoal - widthMFCC - rightPad
        btmPad = lenGoal - lenMFCC - topPad
        padDim = (leftPad, rightPad, topPad, btmPad)
        mfccGoal = F.pad(mfcc, padDim, mode=padMode, value=valuePad)
        return mfccGoal

if __name__ == '__main__':
    sr = 8000
    max_len = int(0.8 * sr)
    record_path = r"free-spoken-digit-dataset-v1.0.8\FSDD\recordings"
    ds = MFCCDataset(record_path, sr, max_len)
    sample_data, sample_label = ds.__getitem__(0)
    print(f"Number of data: {ds.__len__()}")
    print(f"Sample data's shape: {sample_data.shape}")
    print(f"Sample's label: {sample_label}")



