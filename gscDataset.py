# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:50:18 2022

@author: user
"""
import torch, torchaudio
import librosa
import os
import logging
import utils.audioProcessing as audioProcessing
import torch.nn.functional as F
from torchaudio.datasets import SPEECHCOMMANDS
from utils import plotFigure
from torch import nn, optim, Tensor
from models import CustomCNN
from snntorch import functional as SF
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

config_list = {
    'model_name': "Model_B_DiffThres",
    'hop_len': 512,
    'n_samples': 16,
    'channel_in': 1,
    'og_sr': 16000,
    'desired_sr': 8000,
    'max_shape': (32, 32),
    'batch_size': 64,
    'epoch_num': 20,
    'num_steps': 10,
    'lr': 0.001,
    'log_path':"gscTest.log",
    'profile_path':"gscGlop.log",
    'chktpt_path':"gcsChktpt",
    'image_path': "GCS_Images",
    }
label_list = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 
              'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 
              'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 
              'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 
              'visual', 'wow', 'yes', 'zero']
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./datasets", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
            
def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(label_list.index(word))


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels_list[index]

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

class PadMFCC(torch.nn.Module):
    def __init__(self, output_size, padMode='constant', padValue=0) -> None:
        super(PadMFCC, self).__init__()
        self.output_size = output_size
        self.padMode = padMode
        self.padValue = padValue
    
    def forward(self, sample: Tensor) -> Tensor:
        length, width = sample.shape[2], sample.shape[3]
        new_length, new_width = self.output_size
        assert length <= new_length or width <= new_width, "MFCC too large, consider reducing n_mfcc or increasing hop length"
        rightPad = int((new_width - width)/2)
        topPad = int((new_length - length)/2)
        leftPad = new_width - width - rightPad
        btmPad = new_length - length - topPad
        padded = F.pad(sample, (leftPad, rightPad, topPad, btmPad), 
                       mode=self.padMode, value=self.padValue)
        return padded
    
    
def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, 
                                            batch_first=True, 
                                            padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

def create_loaders(batch_size, num_workers, pin_memory):
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )      
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader

def trainSNetGSC(device, model, train_dl, transforms, epoch_num, optimizer, 
                 loss_fn, num_steps, train_loss_hist, train_accu_hist, 
                 checkpoint_path, modelName):
    for epoch in range(epoch_num):
        iterCount = 0
        for i, (data, targets) in enumerate(iter(train_dl)):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            data = transforms(data)

            model.train()
            spk_rec, mem_rec = model(data)

            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            train_loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_rec, targets) 
            train_accu_hist.append(acc)
            iterCount +=1
        print(f' Epoch: {epoch} | Train Loss: {train_loss_hist[-1]:.3f} | Accuracy: {train_accu_hist[-1]:.3f} | Iteration: {iterCount}')

    print("-----Finished Training-----")
    torch.save(model.state_dict(), os.path.join(
        checkpoint_path, 'train--{}-{}.chkpt'.format(modelName, epoch_num)
    ))
    return model, train_loss_hist, train_accu_hist, iterCount

def testSNetGCS(sModel, testDataLoader, device, transforms,
            lossFn, numSteps, testNum, epochNum, 
            modelName, addInfo,
            testLogger, profLogger=None, chkPtPath=None, logSparse=False):
    testLoss = torch.zeros((1), dtype=torch.float, device=device)
    correct = 0
    testLossHist = []
    sparseHist = []
    sModel.eval()
    with torch.no_grad():
        for _, (X, Y) in enumerate(testDataLoader):
                X, Y = X.to(device), Y.to(device)
                X = transforms(X)
                testSpk, testMem = sModel(X)
                if logSparse:
                    sparseHist.append(sModel.get_sparsity())
                _, pred = testSpk.sum(dim=0).max(1)
                for step in range(numSteps):
                    testLoss += lossFn(testMem[step], Y)
                testLossHist.append(testLoss.item())
                correct += (pred==Y).type(torch.float).sum().item()
    
    correct /= testNum
    testLoss = testLossHist[-1]/testNum
    logMessage= f'Model:{modelName}, EpochTrained:{epochNum}, ' \
                f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                f'AddInfo: {addInfo}'
    if logSparse:
        logMessage = f"{logMessage}, Avg sparsity: {sum(sparseHist)/len(sparseHist)}"
    print(logMessage)
    testLogger.info(logMessage)
    if profLogger is not None:
        flops = FlopCountAnalysis(sModel, X)
        profLog = f"Model:{modelName}, AddInfo: {addInfo}\n {flop_count_table(flops)}"
        profLogger.info(profLog)
    if chkPtPath is not None:
        torch.save(sModel.state_dict(), os.path.join(
        chkPtPath, 'test-{}-{}{}.chkpt'.format(modelName, epochNum, addInfo)))
    return

def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def main():
    logPath = config_list["log_path"]
    profilePath = config_list['profile_path']
    if not os.path.exists(logPath):
        open(logPath, 'a').close()
    if not os.path.exists(profilePath):
        open(profilePath, 'a').close()
    logger = setupLogger('ResultsLogger', logPath)
    profLogger = setupLogger("ProfileLogger", profilePath)
    
    checkpoint_path = config_list['chktpt_path']
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} with {torch.cuda.get_device_name()}")

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
        
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")
    print(f"List of labels: {label_list}")
    print("Training and test set initialized")
    print(f"Train set: {len(train_set)}, Test set: {len(test_set)}")

    train_dl, test_dl = create_loaders(config_list['batch_size'], num_workers, pin_memory)
    train_sample, train_sample_label = next(iter(train_dl))
    print(f"Checking sample: {train_sample.shape}, and labels {train_sample_label.shape}")
    
    transforms = torch.nn.Sequential(
        torchaudio.transforms.Resample(orig_freq=config_list['og_sr'], 
                                       new_freq=config_list['desired_sr']),
        torchaudio.transforms.MFCC(sample_rate=config_list['desired_sr'], 
                                   n_mfcc=config_list['n_samples'],
                                   melkwargs={
                                       "hop_length": config_list['hop_len'],
                                       "mel_scale": "htk",
                                       },),
        PadMFCC(config_list["max_shape"]),
    ).to(device)
    
    model = CustomCNN.ModelB(config_list['num_steps'], 0.5, num_class = len(label_list)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config_list['lr'], betas=(0.9, 0.999))
    train_loss_hist = []
    train_accu_hist = []
    iterCount = 0
    
    log_name = f"SCNN-GSC-{config_list['model_name']}"
    model, train_loss_hist, train_accu_hist, iterCount =trainSNetGSC(device, model, train_dl, transforms, 
                                                        config_list['epoch_num'], optimizer, criterion, config_list['num_steps'],
                                                        train_loss_hist, train_accu_hist, 
                                                        checkpoint_path, log_name)
    
    imgPath = os.path.join(config_list['image_path'], 'train--{}-{}{}.png'.format(config_list['model_name'], config_list['epoch_num'], log_name))
    plotFigure.plotTrainingProgTwin(train_accu_hist, train_loss_hist, imgPath, iterCount, spiking=True)
    
    testSNetGCS(model, test_dl, device, transforms, criterion, config_list['num_steps'], len(test_set), config_list['epoch_num'], config_list['model_name'], "", logger, profLogger, checkpoint_path)
    
    
if __name__ == '__main__':
    main()