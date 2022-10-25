import torch
import logging
import os
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from models.CustomCNN import customSNet

def testNet(model, testDataLoader, device, 
            lossFn, testNum, epochNum, 
            modelName, addInfo,
            testLogger, profLogger=None, chkPtPath=None):
    testLoss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for _, (X, Y) in enumerate(testDataLoader):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                testLoss += lossFn(pred, Y).item()
                correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
    testLoss /= testNum
    correct /= testNum
    logMessage= f'Model:{modelName}, EpochTrained:{epochNum}, ' \
                f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                f'AddInfo: {addInfo}'
    print(logMessage)
    testLogger.info(logMessage)
    if profLogger is not None:
        flops = FlopCountAnalysis(model, X)
        profLog = f"Model:{modelName}, AddInfo: {addInfo}\n {flop_count_table(flops)}"
        profLogger.info(profLog)
    if chkPtPath is not None:
        torch.save(model.state_dict(), os.path.join(
        chkPtPath, 'test-{}-{}{}.chkpt'.format(modelName, epochNum, addInfo)))
    return

def testSNet(sModel, testDataLoader, device,
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

def testSNet_confuse(sModel, testDataLoader, device,
            lossFn, numSteps, testNum, epochNum, 
            modelName, addInfo,
            testLogger, confusePath, profLogger=None, chkPtPath=None, logSparse=False):
    testLoss = torch.zeros((1), dtype=torch.float, device=device)
    correct = 0
    testLossHist = []
    sparseHist = []
    y_pred = []
    y_true =[]
    sModel.eval()
    with torch.no_grad():
        for _, (X, Y) in enumerate(testDataLoader):
                X, Y = X.to(device), Y.to(device)      
                testSpk, testMem = sModel(X)
                if logSparse:
                    sparseHist.append(sModel.get_sparsity())
                _, pred = testSpk.sum(dim=0).max(1)
                for step in range(numSteps):
                    testLoss += lossFn(testMem[step], Y)
                testLossHist.append(testLoss.item())
                correct += (pred==Y).type(torch.float).sum().item()
                pred_out = pred.data.cpu().numpy()
                true_out = Y.data.cpu().numpy()
                y_pred.extend(pred_out)
                y_true.extend(true_out)
    correct /= testNum
    testLoss = testLossHist[-1]/testNum
    logMessage= f'Model:{modelName}, EpochTrained:{epochNum}, ' \
                f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                f'AddInfo: {addInfo}'
    if logSparse:
        logMessage = f"{logMessage}, Avg sparsity: {sum(sparseHist)/len(sparseHist)}"
    print(logMessage)
    testLogger.info(logMessage)
    #classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    classes = ("0", "1")
    class_str = ("no", "yes")
    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
    print(np.sum(cf_matrix))
    classCount = [y_true.count(int(i)) for i in classes]
    print(classCount)
    cf_matrix_df = pd.DataFrame(cf_matrix/(classCount), index = [i for i in class_str],
                     columns = [i for i in class_str])
    plt.figure(figsize=(12,7))
    plt.rcParams['font.size'] = '18'   
    sn.heatmap(cf_matrix_df, annot=True)
    plt.savefig(os.path.join(confusePath, addInfo))
    print("SAVE FIG OI")
    if profLogger is not None:
        flops = FlopCountAnalysis(sModel, X)
        profLog = f"Model:{modelName}, AddInfo: {addInfo}\n {flop_count_table(flops)}"
        profLogger.info(profLog)
    if chkPtPath is not None:
        torch.save(sModel.state_dict(), os.path.join(
        chkPtPath, 'test-{}-{}{}.chkpt'.format(modelName, epochNum, addInfo)))
    return

    