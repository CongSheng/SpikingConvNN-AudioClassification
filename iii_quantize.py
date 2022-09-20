import os
import torch
import argparse
import logging
import torch.nn.functional as F
import snntorch as snn
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from snntorch import surrogate
from models import test, train
from models.CustomCNN import qtCNet, qtSNet
from datasets.customDataset import fetchData
from utils import loggingFn

CONFIG_TYPE = "6C3-P2-128F-10F-10Steps"
CHECKPT_PATH = "Expt/checkpoints/QT/"
device = torch.device("cpu")

def staticQtModel(model, config, singleData):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(config)
    model_prepared = torch.quantization.prepare(model)
    print(f"Calibrating with data of size: {singleData.shape}")
    model_prepared(singleData.unsqueeze(0))
    quantizedModel = torch.quantization.convert(model_prepared)
    return quantizedModel

def qtMain(args):
    logPath = args.logPath
    if not os.path.exists(logPath):
        open(logPath, 'a').close()
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logger = loggingFn.setupLogger('ResultsLogger', logPath, formatter)
    modelName = args.modelName

    # Load data
    datasetTrain = fetchData(args.dataPathTrg)
    datasetTest = fetchData(args.dataPathTest)
    trainNum = datasetTrain.__len__()
    testNum = datasetTest.__len__()
    print(f"Train data: {trainNum}")
    print(f"Test data: {testNum}")
    trgLoader = DataLoader(datasetTrain, batch_size=32, shuffle=True, num_workers=2)
    testLoader = DataLoader(datasetTest, batch_size=64, shuffle=False, num_workers=2)
    addInfo = f"quantized_result_{modelName}_{CONFIG_TYPE}"

    # Load Model
    lossFn = nn.CrossEntropyLoss()
    epochNum = args.numEpoch
    # Load state dictionary
    checkpoint = torch.load(args.chkPtPath)
    if modelName == "CustomSCNN":
        numSteps = args.numSteps
        model = qtSNet(args.numSteps, 0.5).to(device)
        print("Spikey QAT!")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                            T_max=4690, 
                                                            eta_min=0, 
                                                            last_epoch=-1)
        qatModel, lossHist, lrHist = train.qatrainSNet(model, epochNum, numSteps, trgLoader, lossFn, optimizer, addInfo=CONFIG_TYPE, scheduler=scheduler, checkpoint_path=CHECKPT_PATH)
        test.testSNet(qatModel, testLoader, device, lossFn, numSteps, testNum, epochNum, modelName, addInfo, logger)
    else:
        model = qtCNet().to(device)
        model.load_state_dict(checkpoint)
        
        # Quantize
        qtModel = staticQtModel(model, 'fbgemm', datasetTrain.__getitem__(2)[0])    

        # Test quantized model
        test.testNet(qtModel, testLoader, device, lossFn, testNum, epochNum, modelName, addInfo, logger)
    
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', type=str, default="AlexCNN", help="name of model")
    parser.add_argument('--chkPtPath', type=str, default="checkpoints/")
    parser.add_argument('--dataPathTrg', type=str, default="transformedData/mfcc/trg", help="Path containing train data")
    parser.add_argument('--dataPathTest', type=str, default="transformedData/mfcc/test", help="Path containing test data")
    parser.add_argument('--numEpoch', type=int, default=20)
    parser.add_argument('--numSteps', type=int, default=10, help="Number of time steps for spiking version")
    parser.add_argument('--logPath', type=str, default='test.log', help="Directory of file to log results")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    qtMain(args)