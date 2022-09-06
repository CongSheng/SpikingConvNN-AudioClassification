import os
import torch
import argparse
import logging
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from snntorch import surrogate
from models import AlexCNN, CustomCNN, train, test
from utils import loggingFn

device = torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

class savedData(Dataset):
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

class customNetQT(nn.Module):
    ## Set max shape = (32, 32)
    def __init__(self):
        super(customNetQT, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 3) # (6 ,30, 30)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 15, 15)
        self.conv2 = nn.Conv2d(6, 16, 3) # (16, 13, 13)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def qtMain(args):
    logPath = args.logPath
    if not os.path.exists(logPath):
        open(logPath, 'a').close()
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logger = loggingFn.setupLogger('ResultsLogger', logPath, formatter)
    modelName = args.modelName

    # Load data
    datasetTrain = savedData(args.dataPathTrg)
    datasetTest = savedData(args.dataPathTest)
    trainNum = datasetTrain.__len__()
    testNum = datasetTest.__len__()
    print(f"Train data: {trainNum}")
    print(f"Test data: {testNum}")
    trgLoader = DataLoader(datasetTrain, batch_size=32, shuffle=True, num_workers=2)
    testLoader = DataLoader(datasetTest, batch_size=16, shuffle=False, num_workers=2)
    addInfo = f"quantized_result_{modelName}"

    # Load Model
    if modelName == "CustomSCNN":
        model = CustomCNN.customSNet(args.num_steps, 0.5).to(device)
    else:
        model = customNetQT().to(device)

    lossFn = nn.CrossEntropyLoss()
    
    # Load state dictionary
    checkpoint = torch.load(args.chkPtPath)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    epoch = args.numEpoch

    # Quantize
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    print(datasetTrain.__getitem__(0)[0].shape)
    model_prepared(datasetTrain.__getitem__(0)[0].unsqueeze(0))
    qtModel= torch.quantization.convert(model_prepared)

    # Test quantized model
    testLoss, correct = 0, 0
    qtModel.eval()

    if modelName == "CustomSCNN":
        testLoss = torch.zeros((1), dtype=torch.float, device=device)
        correct = 0
        testLossHist = []
        with torch.no_grad():
            for _, (X, Y) in enumerate(testLoader):
                    X, Y = X.to(device), Y.to(device)      
                    testSpk, testMem = qtModel(X)
                    _, pred = testSpk.sum(dim=0).max(1)
                    for step in range(args.numSteps):
                        testLoss += lossFn(testMem[step], Y)
                    testLossHist.append(testLoss.item())
                    correct += (pred==Y).type(torch.float).sum().item()
        correct /= testNum
        testLoss = testLossHist[-1]/testNum
        logMessage= f'Model:{modelName}, EpochTrained:{epoch}, ' \
                    f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                    f'AddInfo: {addInfo}'
        print(logMessage)
        logger.info(logMessage)
    else:
        with torch.no_grad():
            for _, (X, Y) in enumerate(testLoader):
                X, Y = X.to(device), Y.to(device) 
                pred = model(X)
                testLoss += lossFn(pred, Y).item()
                correct += (pred.argmax(1)==Y).type(torch.float).sum().item()        
        testLoss /= testNum
        correct /= testNum
        logMessage= f'Model:{modelName}, EpochTrained:{epoch}, ' \
                    f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                    f'AddInfo: {addInfo}'
        print(logMessage)
        logger.info(logMessage)
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default="AlexCNN", help="name of model")
    parser.add_argument('--chkPtPath', type=str, default="checkpoints/")
    parser.add_argument('--dataPathTrg', type=str, default="transformedData/mfcc/trg", help="Path containing train data")
    parser.add_argument('--dataPathTest', type=str, default="transformedData/mfcc/trg", help="Path containing test data")
    parser.add_argument('--numEpoch', type=int, default=20)
    parser.add_argument('--numSteps', type=int, default=10, help="Number of time steps for spiking version")
    parser.add_argument('--logPath', type=str, default='test.log', help="Directory of file to log results")
    args = parser.parse_args()
    print(args)
    qtMain(args)