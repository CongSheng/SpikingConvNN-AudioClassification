import torch
from torch import nn, optim 
import os
import logging
import models
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import plotFigure
from snntorch import surrogate

from datasets import mfcc_dataset
from models import CustomCNN, train, test
from torch.utils.data import random_split
from snntorch import functional as SF

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import numpy as np
from ray.air import session
from ray.air.checkpoint import Checkpoint

EXPERIMENT_NAME = "Threshold_Sparse"
MODEL_NAME = "SCNN"
LOG_PATH = "Expt/expt.log"
PROFILE_LOG = "Expt/exptProfile.log"
CHECKPOINT_PATH = "Expt/checkpoints"
NUM_CLASS= 10
MAX_SHAPE = (32,32)
HOP_LENGTH = 512
FRAME_LENGTH = 256
N_MFCC = 16
MAX_EPOCH = 25
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

# Parameters of interest
defaultParam = {
    "batchsize": 16,
    "timestep": 10,
    "threshold": 1,
    "beta":0.5,
    "epochNum":20,
    "learningRate": 0.001,
    "checkpt":"/hyperTuning/"
}

# Data
fullDataset = mfcc_dataset.MFCCDataset("free-spoken-digit-dataset-v1.0.8/FSDD/recordings/",
                                    sample_rate=8000,
                                    max_shape = MAX_SHAPE,
                                    channel_in=1,
                                    hop_length=HOP_LENGTH, 
                                    n_samples=N_MFCC)
fullDsLen = fullDataset.__len__()
trainLen = int(0.6 * fullDsLen)
testLen = int(0.2 * fullDsLen)
valLen = fullDsLen - trainLen - testLen
print(f"Total Number:{fullDsLen} , Training Number:{trainLen}, Val Number: {valLen}, Test Number:{testLen}")
train_ds, val_ds, test_ds = random_split(fullDataset, [trainLen, valLen, testLen])                                    
train_dl = DataLoader(train_ds, defaultParam["batchsize"], shuffle=True)
val_dl = DataLoader(val_ds, batch_size=defaultParam["batchsize"], shuffle=True)
test_dl = DataLoader(test_ds, defaultParam["batchsize"], shuffle=True)


def setupLogger(name, logPath, level=logging.INFO):
    handler = logging.FileHandler(logPath)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def trainValTestSplit(trainSplit, testSplit, fullDataset):
    fullDsLen = fullDataset.__len__()
    trainLen = int(trainSplit * fullDsLen)
    testLen = int(testSplit * fullDsLen)
    valLen = fullDsLen - trainLen - testLen
    print(f"Total Number:{fullDsLen} , Training Number:{trainLen}, Val Number: {valLen}, Test Number:{testLen}")
    trainDs, valDs, testDs = random_split(fullDataset, [trainLen, valLen, testLen])
    return trainDs, valDs, testDs, fullDsLen, trainLen, valLen, testLen

def trainValSNet(device, model, train_dl, val_dl, epoch_num, optimizer, loss_fn, num_steps,
            train_loss_hist, train_accu_hist, checkpoint_path, modelName):
    train_loss_hist = []
    train_accu_hist = []
    for epoch in range(epoch_num):
        iterCount = 0
        for i, (data, targets) in enumerate(iter(train_dl)):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            model.train()
            spk_rec, mem_rec = model(data)

            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)
                print(loss_val)

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            train_loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_rec, targets) 
            train_accu_hist.append(acc)
            iterCount +=1
        print(f' Epoch: {epoch} | Train Loss: {train_loss_hist[-1]:.3f} | Accuracy: {train_accu_hist[-1]:.3f} | Iteration: {iterCount}')

    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(val_dl, 0):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            spk_rec, mem_rec = model(data)
            _, pred = spk_rec.sum(dim=0).max(1)
            for step in range(num_steps):
                    val_loss += loss_fn(spk_rec[step], labels)
            total += labels.size(0)
            correct += (pred==labels).type(torch.float).sum().item()
            val_steps += 1
    # print("-----Finished Training-----")
    torch.save(model.state_dict(), os.path.join(
        checkpoint_path, 'valtrain--{}-{}.chkpt'.format(modelName, epoch_num)
    ))
    checkpoint = Checkpoint.from_directory("my_model")
    session.report({"loss": (val_loss / val_steps), "accuracy": correct / total}, checkpoint=checkpoint)
    return model, train_loss_hist, train_accu_hist, iterCount

def rayTuneTrain(config):
    # Model
    model = CustomCNN.customSNetv2(num_steps=config["timestep"],
                                    beta=config["beta"],
                                    threshold=config["threshold"], 
                                    num_class=NUM_CLASS).to(device)
    loss_fn=nn.CrossEntropyLoss(),
    optimizer= optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model, train_loss_hist, train_accu_hist, iterCount = trainValSNet(device, model, train_dl, val_dl, defaultParam["epochNum"], optimizer, loss_fn, config["timestep"], train_loss_hist, train_accu_hist, defaultParam["checkpt"], "hyperTuneSCNN")

def hyperTuneMain():
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, 'a').close()
    if not os.path.exists(PROFILE_LOG):
        open(PROFILE_LOG, 'a').close()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    logger = setupLogger('ResultsLogger', LOG_PATH)
    profLogger = setupLogger("ProfileLogger", PROFILE_LOG)

    sparseMode = False
    if "sparse" in EXPERIMENT_NAME.lower():
        sparseMode = True

    # Config
    config = {
        "threshold": tune.sample_from(lambda _: np.random.randint(1, 9)),
        "timestep":  tune.sample_from(lambda _: np.random.randint(0, 25)),
        "beta":  tune.uniform(0, 1),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = ASHAScheduler(
        max_t=MAX_EPOCH,
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(rayTuneTrain),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=10,
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_model = CustomCNN.customSNetv2(num_steps=best_result["timestep"],
                                    beta=best_result["beta"],
                                    threshold=best_result["threshold"], 
                                    num_class=NUM_CLASS).to(device)
    test.testSNet(test_model, test_dl, device, nn.CrossEntropyLoss(), defaultParam["timestep"], len(test_dl), defaultParam["epochNum"], MODEL_NAME, "hyperStudy", logger, profLogger, CHECKPOINT_PATH, sparseMode)

if __name__ == '__main__':
    hyperTuneMain()