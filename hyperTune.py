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
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
import numpy as np
from ray.air import session
from ray.air.checkpoint import Checkpoint

MODEL_NAME = "SCNN_A"
LOG_PATH = "hyperTuning/resultsScriptRun.log"
PROFILE_LOG = "hyperTuning/profileScriptRun.log"
PARA_LOG = "hyperTuning/parameterScriptRun.log"
CHECKPOINT_PATH = "hyperTuning/checkpoints"
ADD_INFO = "hyperStudy-ModelC-random"
NUM_CLASS= 10
MAX_SHAPE = (32,32)
HOP_LENGTH = 512
FRAME_LENGTH = 256
N_MFCC = 16
MAX_EPOCH = 25
formatter = logging.Formatter('%(asctime)s, %(levelname)s, %(name)s, %(message)s')

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
    "checkpt":"hyperTuning/"
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
                 checkpoint_path, modelName):
    train_loss_hist = []
    train_accu_hist = []
    loss_fn = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epoch_num)):
        running_loss = 0.0
        correct = 0
        total = 0
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

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            train_loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_rec, targets) 
            train_accu_hist.append(acc)
            iterCount +=1
        print(f' Epoch: {epoch} | Train Loss: {train_loss_hist[-1]:.3f} | Accuracy: {train_accu_hist[-1]:.3f} | Iteration: {iterCount}')

    #Val Loop
    val_loss = torch.zeros((1), dtype=torch.float, device=device)
    correct = 0
    val_loss_hist = []
    val_steps = 0
    with torch.no_grad():
        for _, (X, Y) in enumerate(val_dl):
            X, Y = X.to(device), Y.to(device)      
            val_spk, val_mem = model(X)
            _, pred = val_spk.sum(dim=0).max(1)
            for step in range(num_steps):
                val_loss += loss_fn(val_mem[step], Y)
            val_loss_hist.append(val_loss.item())
            correct += (pred==Y).type(torch.float).sum().item()
        val_steps+=1
    loss_from_val = val_loss_hist[-1] / valLen
    acc_from_val = correct / valLen
    print(f"Loss is {loss_from_val} and acc is {acc_from_val}")
    os.makedirs(checkpoint_path, exist_ok=True)
    pathName = os.path.join(checkpoint_path, "checkpt.pt")
    torch.save(model.state_dict(), pathName)
    checkpoint = Checkpoint.from_directory(checkpoint_path)
    session.report({"loss": loss_from_val, "accuracy": acc_from_val}, checkpoint=checkpoint)
    return model, train_loss_hist, train_accu_hist, val_loss_hist

def rayTuneTrain(configTune):
    # Model
    if "A" in MODEL_NAME:
        model = CustomCNN.ModelA(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    elif "B" in MODEL_NAME:
        model = CustomCNN.ModelB(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    elif "C" in MODEL_NAME:
        model = CustomCNN.ModelC(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    else:
        print("Invalid Model")
    loss_fn=nn.CrossEntropyLoss(),
    optimizer= optim.Adam(model.parameters(), lr=defaultParam['learningRate'], betas=(0.9, 0.999))
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpt.pt"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model, train_loss_hist, train_accu_hist, _ = trainValSNet(device, 
                                                            model, 
                                                            train_dl, 
                                                            val_dl, 
                                                            defaultParam["epochNum"], 
                                                            optimizer, 
                                                            loss_fn, 
                                                            int(configTune["timestep"]),  
                                                            CHECKPOINT_PATH, 
                                                            "hyperTuneSCNN")

def hyperTuneMain():
    if not os.path.exists(LOG_PATH):
        open(LOG_PATH, 'a').close()
    if not os.path.exists(PARA_LOG):
        open(PARA_LOG, 'a').close()
    if not os.path.exists(PROFILE_LOG):
        open(PROFILE_LOG, 'a').close()
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    logger = setupLogger('ResultsLogger', LOG_PATH)
    profLogger = setupLogger("ProfileLogger", PROFILE_LOG)
    paraLogger = setupLogger("ParaLogger", PARA_LOG)

    sparseMode = True

    # Config
    configTune = {
        "threshold": tune.sample_from(lambda _: np.random.randint(1, 9)),
        "timestep":  tune.sample_from(lambda _: np.random.randint(0, 25)),
        # "threshold": tune.uniform(1, 9),
        # "timestep": tune.uniform(0, 25),
        "beta":  tune.uniform(0, 1),
        # "lr": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = ASHAScheduler(
        max_t=MAX_EPOCH,
        grace_period=1,
        reduction_factor=2)

    algo = BayesOptSearch(random_search_steps=4)
    # algo = HyperOptSearch(metric="loss", mode="min")

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(rayTuneTrain),
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=10,
            # search_alg=algo,
        ),
        param_space=configTune,
    )
    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "max")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    if "A" in MODEL_NAME:
        test_model = CustomCNN.ModelA(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    elif "B" in MODEL_NAME:
        test_model = CustomCNN.ModelB(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    elif "C" in MODEL_NAME:
        test_model = CustomCNN.ModelC(num_steps=int(configTune["timestep"]),
                                        beta=configTune["beta"],
                                        threshold=int(configTune["threshold"]), 
                                        num_class=NUM_CLASS).to(device)
    else:
        print("Invalid Model")
    
    state_dict = torch.load(os.path.join(best_result.checkpoint.to_directory(), "checkpt.pt"))
    test_model.load_state_dict(state_dict)
    paraLogger.info(best_result.config)
    test.testSNet(test_model, test_dl, device, nn.CrossEntropyLoss(), int(best_result.config["timestep"]), testLen, defaultParam["epochNum"], MODEL_NAME, ADD_INFO, logger, profLogger, CHECKPOINT_PATH, sparseMode)

if __name__ == '__main__':
    hyperTuneMain()