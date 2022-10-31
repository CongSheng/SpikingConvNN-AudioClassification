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

MODEL_NAME = "SCNN_B_CHECK"
TRAIN_LOG_PATH = "paperRun/train.log"
TEST_LOG_PATH = "paperRun/test.log"
PROFILE_LOG = "paperRun/profile.log"
CHECKPOINT_PATH = "paperRun/checkpoints"
MAX_SHAPE = (32,32)
NUM_CLASSES = 10
NUM_ITER = 50
formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s, %(message)s')

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

def scriptRunMain():
# Check logging and directories
    if not os.path.exists(TRAIN_LOG_PATH):
        open(TRAIN_LOG_PATH, 'a').close()
    if not os.path.exists(TEST_LOG_PATH):
        open(TEST_LOG_PATH, 'a').close()
    if not os.path.exists(PROFILE_LOG):
        open(PROFILE_LOG, 'a').close()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    trainlogger = setupLogger('trainLogger',TRAIN_LOG_PATH)
    testlogger = setupLogger("Testlogger", TEST_LOG_PATH)
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
    num_classes = NUM_CLASSES
    train_dl = DataLoader(train_ds, batch_size=defaultParam["batchsize"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=defaultParam["batchsize"], shuffle=True)
    print("-----Loaded-----\n")

    # Model
    train_loss_hist = []
    train_accu_hist = []

    sparseMode = True
    
    for i in range(NUM_ITER):
        if "A" in MODEL_NAME:
            print("MODEL A SELECTED!")
            model = CustomCNN.ModelA(defaultParam["timestep"], 0.5, num_class = num_classes).to(device)
        elif "B" in MODEL_NAME:
            print("MODEL B SELECTED!")
            model = CustomCNN.ModelB(defaultParam["timestep"], 0.5, num_class = num_classes).to(device)
        elif "C" in MODEL_NAME:
            print("MODEL C SELECTED!")
            model = CustomCNN.ModelC(defaultParam["timestep"], 0.5, num_class = num_classes).to(device)
        else:
            print("Invalid model")

        optimizer = optim.Adam(model.parameters(), lr=defaultParam["learningRate"], betas=(0.9, 0.999))
        model, train_loss_hist, train_accu_hist, iterCount, avg_loss= train.trainSNet(device, model, train_dl, 
                                                            defaultParam["epochNum"], optimizer, defaultParam["lossFn"], defaultParam["timestep"],
                                                            train_loss_hist, train_accu_hist, 
                                                            CHECKPOINT_PATH, MODEL_NAME)
        
        trainMessage = f"Train Loss, {train_loss_hist[-1]:.3f},  Avg Loss, {avg_loss:.3f}, Accuracy, {train_accu_hist[-1]:.3f}"
        trainlogger.info(trainMessage)
        test.testSNet(model, test_dl, device, defaultParam["lossFn"], defaultParam["timestep"], test_num, defaultParam["epochNum"], MODEL_NAME, "scriptRun", testlogger, profLogger, CHECKPOINT_PATH, sparseMode)
        
    
if __name__=='__main__':
    scriptRunMain()
        