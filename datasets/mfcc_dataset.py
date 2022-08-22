import os
import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import utils.audioProcessing as audioProcessing

class MFCCDataset(Dataset):
    def __init__(self, data_path, sample_rate, max_length,
                hop_length=512, n_samples=16, channel_in=1):
        print("-----MFCC Dataset creation-----\n")
        self.data_path = str(data_path)
        self.data = []
        self.labels = []
        for src_file in os.listdir(data_path):
            full_path = os.path.join(data_path, src_file)
            audio, _ = librosa.load(full_path, mono=True, sr=sample_rate)
            audio = audioProcessing.remove_silence(audio)
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
                mfcc = torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))
            if channel_in==3:
                l_mfcc, w_mfcc = mfcc.shape[0], mfcc.shape[1]
                r_pad = int((227 - w_mfcc) / 2)
                l_pad = 227 - r_pad - w_mfcc
                t_pad = int((227 - l_mfcc) / 2)
                b_pad = 227 - t_pad - l_mfcc
                pad_dim = (l_pad, r_pad, t_pad, b_pad)
                mfcc = F.pad(mfcc, pad_dim, mode='constant', value=0) 
                mfcc = torch.stack([mfcc, mfcc, mfcc], dim=0)

            self.data.append(mfcc)
            self.labels.append(int(src_file[0]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index]
        return [X, Y]

if __name__ == '__main__':
    sr = 8000
    max_len = int(0.8 * sr)
    record_path = r"free-spoken-digit-dataset-v1.0.8\FSDD\recordings"
    ds = MFCCDataset(record_path, sr, max_len)
    sample_data, sample_label = ds.__getitem__(0)
    print(f"Number of data: {ds.__len__()}")
    print(f"Sample data's shape: {sample_data.shape}")
    print(f"Sample's label: {sample_label}")



