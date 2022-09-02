import profile
from re import L
from urllib.response import addinfo
import torch
from torch import nn, optim
import argparse
import os
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import plotFigure
from snntorch import surrogate

from datasets import customDataset, mfcc_dataset
from models import AlexCNN, CustomCNN, train, test

# Device config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

SAMPLE_RATE = 8000
MAX_SHAPE = (32, 32)
HOP_LENGTH = 64
FRAME_LENGTH = 32
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

def main(args):
    # Logging
    logPath = args.logPath
    profilePath = args.profilePath
    if not os.path.exists(logPath):
        open(logPath, 'a').close()
    if not os.path.exists(profilePath):
        open(profilePath, 'a').close()
    logger = setupLogger('ResultsLogger', logPath)
    profLogger = setupLogger("ProfileLogger", profilePath)

    # Checkpoint directory
    checkpoint_path = args.chkpt_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Init dataset
    datasetType = args.dataset_type
    assert datasetType=="mfcc" or datasetType=="rmse", 'Invalid dataset type'
    if datasetType == "mfcc":
        full_ds = mfcc_dataset.MFCCDataset(args.data_path, 
                                            sample_rate=SAMPLE_RATE,
                                            max_shape = MAX_SHAPE,
                                            channel_in=args.channel_in,
                                            hop_length=HOP_LENGTH)
        full_ds_len = full_ds.__len__()
    if datasetType == "rmse":
        full_ds = customDataset.RMSEDataset(args.data_path,
                                            sample_rate=SAMPLE_RATE,
                                            max_shape = MAX_SHAPE,
                                            channel_in = args.channel_in,
                                            frame_length=FRAME_LENGTH,
                                            hop_length = HOP_LENGTH)
        full_ds_len = full_ds.__len__()
    
    # Split dataset and implement dataloader
    train_num = int(args.train_partition * full_ds_len)
    test_num = int(args.test_partition * full_ds_len)
    assert (train_num + test_num) == full_ds_len, "Invalid partitioning"
    train_ds, test_ds = torch.utils.data.random_split(full_ds, 
                                                    [train_num, test_num])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    print("-----Loaded-----\n")
    
    # Setting up neural net
    spikingMode = False
    iterCount = 0
    modelName = args.model_name
    if modelName=="AlexCNN":
        model = AlexCNN.AlexNet().to(device)
    elif modelName == "AlexSCNN":
        model_full = AlexCNN.AlexSpikingNet(device, 0.5, surrogate.fast_sigmoid(slope=0.75))
        model = model_full.net
    elif modelName == "CustomSCNN":
        model = CustomCNN.customSNet(args.num_steps, 0.5).to(device)
    else:
        model = CustomCNN.customNet().to(device)
    
    # Setting up optimizers and tracking
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    train_loss_hist = []
    train_accu_hist = []
    epoch_num = args.num_epochs
    
    # Print plot feature if stated
    addInfo = f"hopLen{HOP_LENGTH}_{datasetType}"

    if args.plot_feature == 'Y':
        sample_point, sample_label = next(iter(train_dl))
        mfccPath = os.path.join(args.img_path, '{}_{}{}.png'.format(sample_label[0], datasetType, addInfo))
        plotFigure.plotMfcc(sample_point[0][0], mfccPath, sample_label[0])

    # Train
    if modelName == "AlexSCNN" or modelName == "CustomSCNN":
        spikingMode = True
        numSteps = args.num_steps
        addInfo = f"{addInfo}_{numSteps}Steps"
        model, train_loss_hist, train_accu_hist, iterCount = train.trainSNet(device, model, train_dl, 
                                                            epoch_num, optimizer, criterion, args.num_steps,
                                                            train_loss_hist, train_accu_hist, 
                                                            checkpoint_path, modelName)
    else:
        model, train_loss_hist, train_accu_hist = train.trainNet(device, model, train_dl, 
                                                            epoch_num, optimizer, criterion, 
                                                            train_loss_hist, train_accu_hist, 
                                                            checkpoint_path, modelName)

    imgPath = os.path.join(args.img_path, 'train--{}-{}{}.png'.format(modelName, epoch_num, addInfo))
    plotFigure.plotTrainingProgTwin(train_accu_hist, train_loss_hist, imgPath, iterCount, spiking=spikingMode)
    
    # Test
    if args.test=="Y":
        if modelName == "AlexSCNN" or modelName == "CustomSCNN":
            test.testSNet(model, test_dl, device, criterion, args.num_steps, test_num, epoch_num, checkpoint_path, modelName, addInfo, logger, profLogger)
        else:
            test.testNet(model, test_dl, device, criterion, test_num, epoch_num, checkpoint_path, modelName, addInfo, logger, profLogger)
        # model.eval()
        # test_loss, correct = 0, 0
        # with torch.no_grad():
        #     for _, (X, Y) in enumerate(test_dl):
        #         X, Y = X.to(device), Y.to(device)
        #         pred = model(X)

        #         test_loss += criterion(pred, Y).item()
        #         correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
        # test_loss /= test_num
        # correct /= test_num
        # logMessage = f'Model:{modelName}, EpochTrained:{epoch_num}, Acc: {(100*correct):>0.1f}%, AvgLoss: {test_loss:>8f}, AddInfo: {addInfo}'
        # logging.info(logMessage)
        # torch.save(model.state_dict(), os.path.join(
        # checkpoint_path, 'test-{}-{}{}.chkpt'.format(modelName, epoch_num, addInfo))
        #)

def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="AlexCNN", help="name of model")
    parser.add_argument('--data_path', type=str, default="free-spoken-digit-dataset-v1.0.8/FSDD/recordings/", help="Path containing audio data")
    parser.add_argument('--img_path', type=str, default="figures/", help="Path for plots")
    parser.add_argument('--chkpt_path', type=str, default="checkpoints/")
    parser.add_argument('--dataset_type', type=str, default="mfcc", help="Type of dataset to load")
    parser.add_argument('--train_partition', type=float, default=0.8, help="Fraction of dataset for training (0-1)")
    parser.add_argument('--test_partition', type=float, default=0.2, help="Fraction of dataset for testing (0-1)")
    parser.add_argument('--channel_in', type=int, default=1, help="Number of channel for model input")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_steps', type=int, default=10, help="Number of time steps for spiking version")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--test', type=str, default="N")
    parser.add_argument('--plot_feature', type=str, default="N")
    parser.add_argument('--logPath', type=str, default='test.log', help="Directory of file to log results")
    parser.add_argument('--profilePath', type=str, default='flopLog.log', help="Directory of file to log profile")
    args = parser.parse_args()
    print(args)
    main(args)


    
