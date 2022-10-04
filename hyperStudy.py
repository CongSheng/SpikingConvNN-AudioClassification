from email.policy import default
from xml.etree.ElementInclude import default_loader
import torch
from torch import nn, optim 
import os
import logging
from main import MAX_SHAPE
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import plotFigure
from snntorch import surrogate

from datasets import customDataset
from models import CustomCNN, train, test

EXPERIMENT_NAME = "Threshold_Sparse"
MODEL_NAME = "SCNN"
LOG_PATH = "Expt/expt.log"
PROFILE_LOG = "Expt/exptProfile.log"
CHECKPOINT_PATH = "Expt/checkpoints"
MAX_SHAPE = (32,32)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

# Search Space
T_SPACE = [1, 8, 10, 15, 18, 20, 50, 100]
THRESHOLD_SPACE = [0.0, 0.5, 1.0, 5.0, 10.0, 15.0]
BETA_SPACE = [0, 0.2, 0.5, 0.8, 1.0]

# Device config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def hyperStudyMain():
    # Check logging and directories
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, 'a').close()
    if not os.path.exists(PROFILE_LOG):
        open(PROFILE_LOG, 'a').close()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    logger = setupLogger('ResultsLogger', LOG_PATH)
    profLogger = setupLogger("ProfileLogger", PROFILE_LOG)

    # Parameters of interest
    defaultParam = {
        "batchsize": 16,
        "timestep": 10,
        "threshold": 1,
        "beta":0.5,
        "epochNum":20,
        "lossFn": nn.CrossEntropyLoss(),
        "learningRate": 0.001,
    }

    # Import data
    train_ds = customDataset.fetchData("transformedData/mfcc/trg")
    test_ds = customDataset.fetchData("transformedData/mfcc/test")
    train_num = train_ds.__len__()
    test_num = test_ds.__len__()
    full_ds_len = train_num + test_num
    print(f"Train data: {train_num}")
    print(f"Test data: {test_num}")
    num_classes = 10
    train_dl = DataLoader(train_ds, batch_size=defaultParam["batchsize"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=defaultParam["batchsize"], shuffle=True)
    print("-----Loaded-----\n")

    # Model
    train_loss_hist = []
    train_accu_hist = []

    sparseMode = False
    if "sparse" in EXPERIMENT_NAME.lower():
        sparseMode = True

    if "time" in EXPERIMENT_NAME.lower():
        for timestep in tqdm(T_SPACE):
            model = CustomCNN.customSNetv2(timestep, defaultParam["beta"], num_class=num_classes).to(device)
            logInfo = EXPERIMENT_NAME + str(timestep)
            optimizer = optim.Adam(model.parameters(), lr=defaultParam["learningRate"], betas=(0.9, 0.999))
            model, train_loss_hist, train_accu_hist, iterCount = train.trainSNet(device, model, train_dl, 
                                                            defaultParam["epochNum"], optimizer, defaultParam["lossFn"], timestep,
                                                            train_loss_hist, train_accu_hist, 
                                                            CHECKPOINT_PATH, MODEL_NAME)
            test.testSNet(model, test_dl, device, defaultParam["lossFn"], timestep, test_num, defaultParam["epochNum"], MODEL_NAME, logInfo, logger, profLogger, CHECKPOINT_PATH, sparseMode)
    elif "threshold" in EXPERIMENT_NAME.lower():
        for currThres in tqdm(THRESHOLD_SPACE):
            model = CustomCNN.customSNetv2(defaultParam["timestep"], defaultParam["beta"], num_class=num_classes, threshold=currThres).to(device)
            logInfo = EXPERIMENT_NAME + str(currThres)
            optimizer = optim.Adam(model.parameters(), lr=defaultParam["learningRate"], betas=(0.9, 0.999))
            model, train_loss_hist, train_accu_hist, iterCount = train.trainSNet(device, model, train_dl, 
                                                            defaultParam["epochNum"], optimizer, defaultParam["lossFn"], defaultParam["timestep"],
                                                            train_loss_hist, train_accu_hist, 
                                                            CHECKPOINT_PATH, MODEL_NAME)
            test.testSNet(model, test_dl, device, defaultParam["lossFn"], defaultParam["timestep"], test_num, defaultParam["epochNum"], MODEL_NAME, logInfo, logger, profLogger, CHECKPOINT_PATH, sparseMode)
    elif "beta" in EXPERIMENT_NAME.lower():
        for beta in tqdm(BETA_SPACE):
            model = CustomCNN.customSNetv2(defaultParam["timestep"], beta, num_class=num_classes, threshold=defaultParam["threshold"]).to(device)
            logInfo = EXPERIMENT_NAME + str(beta)
            optimizer = optim.Adam(model.parameters(), lr=defaultParam["learningRate"], betas=(0.9, 0.999))
            model, train_loss_hist, train_accu_hist, iterCount = train.trainSNet(device, model, train_dl, 
                                                            defaultParam["epochNum"], optimizer, defaultParam["lossFn"], defaultParam["timestep"],
                                                            train_loss_hist, train_accu_hist, 
                                                            CHECKPOINT_PATH, MODEL_NAME)
            test.testSNet(model, test_dl, device, defaultParam["lossFn"], defaultParam["timestep"], test_num, defaultParam["epochNum"], MODEL_NAME, logInfo, logger, profLogger, CHECKPOINT_PATH, sparseMode)
    else:
        print("Design space not set up yet")
    
if __name__=='__main__':
    hyperStudyMain()
        
                                            
