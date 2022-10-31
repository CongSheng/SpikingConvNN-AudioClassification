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
HOP_LENGTH = 512
FRAME_LENGTH = 256
N_MFCC = 16
CONFUSE_PATH = "hyperTuning/confuseMatrix"
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

def trainTestSplit(trainSplit, testSplit, fullDataset):
    fullDsLen = fullDataset.__len__()
    trainLen = int(trainSplit * fullDsLen)
    # testLen = int(testSplit * fullDsLen)
    testLen = fullDsLen - trainLen
    print(f"Total Number:{fullDsLen} , Training Number:{trainLen}, Test Number:{testLen}")
    trainDs, testDs = torch.utils.data.random_split(fullDataset, [trainLen, testLen])
    return trainDs, testDs, fullDsLen, trainLen, testLen

def main(args):
    global SAMPLE_RATE
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
    num_classes = args.num_class
    # assert datasetType=="mfcc" or datasetType=="rmse", 'Invalid dataset type'
    if datasetType == "mfcc":
        full_ds = mfcc_dataset.MFCCDataset(args.data_path, 
                                            sample_rate=SAMPLE_RATE,
                                            max_shape = MAX_SHAPE,
                                            channel_in=args.channel_in,
                                            hop_length=HOP_LENGTH, 
                                            n_samples=N_MFCC)
        train_ds, test_ds, full_ds_len, train_num, test_num = trainTestSplit(args.train_partition, args.test_partition, full_ds)
        addInfo = f"hopLen{HOP_LENGTH}_nMFCC{N_MFCC}_{datasetType}"
    elif datasetType == "MFCC_fixed":
        train_ds = customDataset.fetchData("transformedData/mfcc/trg")
        test_ds = customDataset.fetchData("transformedData/mfcc/test")
        train_num = train_ds.__len__()
        test_num = test_ds.__len__()
        full_ds_len = train_num + test_num
        print(f"Train data: {train_num}")
        print(f"Test data: {test_num}")
        addInfo = f"hopLen{HOP_LENGTH}_nMFCC{N_MFCC}_{datasetType}"
    elif datasetType == "rmse":
        full_ds = customDataset.RMSEDataset(args.data_path,
                                            sample_rate=SAMPLE_RATE,
                                            max_shape = MAX_SHAPE,
                                            channel_in = args.channel_in,
                                            frame_length=FRAME_LENGTH,
                                            hop_length = HOP_LENGTH)
        train_ds, test_ds, full_ds_len, train_num, test_num = trainTestSplit(args.train_partition, args.test_partition, full_ds)
        addInfo = f"hopLen{HOP_LENGTH}_frameLen{FRAME_LENGTH}_{datasetType}"
    elif datasetType == "speechcommand":
        SAMPLE_RATE = 16000
        # full_ds = mfcc_dataset.MFCCDatasetv2("speechcommand/SpeechCommands/speech_commands_v0.02/",
        #                                     sample_rate=SAMPLE_RATE,
        #                                     max_shape = MAX_SHAPE,
        #                                     channel_in=args.channel_in,
        #                                     hop_length=HOP_LENGTH, 
        #                                     n_samples=N_MFCC)
        train_ds = customDataset.fetchData("transformedData/speechcommand/trg")
        test_ds = customDataset.fetchData("transformedData/speechcommand/test")
        train_num = train_ds.__len__()
        test_num = test_ds.__len__()
        full_ds_len = train_num + test_num
        print(f"Train data: {train_num}")
        print(f"Test data: {test_num}")
        num_classes = 35
        addInfo = f"hopLen{HOP_LENGTH}_nMFCC{N_MFCC}_{datasetType}"
    elif datasetType =="mswc":
        SAMPLE_RATE = 48000
        full_ds = mfcc_dataset.MFCCDatasetv2("mswc/EN/", 
                                            sample_rate=SAMPLE_RATE,
                                            max_shape = MAX_SHAPE,
                                            channel_in=args.channel_in,
                                            hop_length=HOP_LENGTH, 
                                            n_samples=N_MFCC, max_length = SAMPLE_RATE)
        train_ds, test_ds, full_ds_len, train_num, test_num = trainTestSplit(args.train_partition, args.test_partition, full_ds)
        num_classes = 31
        addInfo = f"hopLen{HOP_LENGTH}_nMFCC{N_MFCC}_{datasetType}"
    else:
        print("Invalid dataset")
        raise RuntimeError

    if args.extraInfo != "None":
        addInfo = addInfo + args.extraInfo

    # assert (train_num + test_num) == full_ds_len, "Invalid partitioning"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True)
    print("-----Loaded-----\n")

    if args.datasetSaveDir != "None":
        print(f"dataset dir: {args.datasetSaveDir}")
        savePathDS = f"{args.datasetSaveDir}/{datasetType}/"
        trgPath = os.path.join(savePathDS, "trg/")
        testPath = os.path.join(savePathDS, "test/")
        print(f"Train data path: {trgPath}")
        print(f"Test data path: {testPath}")
        if not os.path.exists(trgPath):
            os.makedirs(trgPath)
        if not os.path.exists(testPath):
            os.makedirs(testPath)
        for i, (data, label) in enumerate((train_ds)):
            torch.save(data, os.path.join(trgPath, f"{label}_{datasetType}_{i}.pt"))
        for i, (data, label) in enumerate((test_ds)):
            torch.save(data, os.path.join(testPath, f"{label}_{datasetType}_{i}.pt"))


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
        model = CustomCNN.ModelB(args.num_steps, 0.5, num_class = num_classes).to(device)
    elif modelName == "CustomSCNN2":
        model = CustomCNN.customSNetv2(args.num_steps, 0.5, num_class = num_classes).to(device)
    else:
        model = CustomCNN.customNet(num_class = num_classes).to(device)
    
    # Setting up optimizers and tracking
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    train_loss_hist = []
    train_accu_hist = []
    epoch_num = args.num_epochs

    if args.plot_feature == 'Y':
        sample_point, sample_label = next(iter(train_dl))
        featurePath = os.path.join(args.img_path, '{}_{}{}.png'.format(sample_label[0], datasetType, addInfo))
        plotFigure.plotFeature(sample_point[0][0], 
                                featurePath,
                                featureName=datasetType, 
                                label=sample_label[0])

    # Train
    if modelName == "AlexSCNN" or modelName == "CustomSCNN" or modelName=="CustomSCNN2":
        spikingMode = True
        numSteps = args.num_steps
        addInfo = f"{addInfo}_{numSteps}Steps"
        model, train_loss_hist, train_accu_hist, iterCount, avg_loss = train.trainSNet(device, model, train_dl, 
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
        if modelName == "AlexSCNN" or modelName == "CustomSCNN" or modelName=="CustomSCNN2":
            test.testSNet_confuse(model, test_dl, device, criterion, args.num_steps, test_num, epoch_num, modelName, addInfo, logger, CONFUSE_PATH, profLogger, checkpoint_path)
            # test.testSNet(model, test_dl, device, criterion, args.num_steps, test_num, epoch_num, modelName, addInfo, logger, profLogger, checkpoint_path)
        else:
            test.testNet(model, test_dl, device, criterion, test_num, epoch_num, modelName, addInfo, logger, profLogger, checkpoint_path)
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
    parser.add_argument('--datasetSaveDir', type=str, default="None", help="Insert dataset path if you wish to save the dataset")
    parser.add_argument('--img_path', type=str, default="Expt/figures/", help="Path for plots")
    parser.add_argument('--chkpt_path', type=str, default="Expt/checkpoints/")
    parser.add_argument('--dataset_type', type=str, default="mfcc", help="Type of dataset to load")
    parser.add_argument('--train_partition', type=float, default=0.8, help="Fraction of dataset for training (0-1)")
    parser.add_argument('--test_partition', type=float, default=0.2, help="Fraction of dataset for testing (0-1)")
    parser.add_argument('--channel_in', type=int, default=1, help="Number of channel for model input")
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_class', type=int, default=10, help="Number of classes in dataset.")
    parser.add_argument('--num_steps', type=int, default=10, help="Number of time steps for spiking version")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--test', type=str, default="N")
    parser.add_argument('--plot_feature', type=str, default="N")
    parser.add_argument('--logPath', type=str, default='expTest.log', help="Directory of file to log results")
    parser.add_argument('--profilePath', type=str, default='expFlopLog.log', help="Directory of file to log profile")
    parser.add_argument('--extraInfo', type=str, default="None")
    args = parser.parse_args()
    print(args)
    main(args)


    
