import os
import numpy as np
import pandas as pd
import librosa

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import utils.audioProcessing as audioProcessing

class RMSEDataset(Dataset):
    def __init__(self, data_path, sample_rate, max_shape, filter_order=2,
                frame_length=256, hop_length=512, n_filters=16, channel_in=1, max_length=int(0.8*8000)):
        print("-----RMSE creation-----\n")
        self.data_path = str(data_path)
        self.data = []
        self.labels = []
        sosFB = audioProcessing.generateIIRFilter(sample_rate, 
                                                    n_filters, 
                                                    filter_order)
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
            rmse = audioProcessing.filterEnergy(audio, sosFB, frame_length, hop_length)
            rmse = librosa.util.normalize(rmse, axis=1)
            #rmse = rmse.reshape(1, rmse.shape[0], rmse.shape[1])
            rmse = torch.from_numpy(rmse).float()

            if channel_in==1: 
                rmse = audioProcessing.padFeature(rmse, max_shape)
                rmse = torch.reshape(rmse, (1, rmse.shape[0], rmse.shape[1]))
            if channel_in==2:
                rmse = audioProcessing.padFeature(rmse, max_shape)
                rmse = torch.stack([rmse, rmse], dim=0)
            if channel_in==3:
                rmse = audioProcessing.padFeature(rmse, max_shape)
                rmse = torch.stack([rmse, rmse, rmse], dim=0)

            self.data.append(rmse)
            self.labels.append(int(src_file[0]))
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        X = self.data[index]
        Y = self.labels[index]
        return [X, Y]

class fetchData(Dataset):
    def __init__(self, featurePath, channelIn=1):
        self.featurePath = featurePath
        self.len = len(os.listdir(featurePath))
        self.channelIn = channelIn
    
    def __getitem__(self, index):
        listFeature = os.listdir(self.featurePath)
        singleFeatPath = os.path.join(self.featurePath, listFeature[index])
        data = torch.load(singleFeatPath)
        label = int(listFeature[index][0])
        return data, label
    
    def __len__(self):
        return self.len