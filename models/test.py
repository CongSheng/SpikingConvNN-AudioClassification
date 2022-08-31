import torch
import logging
import os
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

def testNet(model, testDataLoader, device, 
            lossFn, testNum, epochNum, 
            chkPtPath, modelName, addInfo,
            testLogger, profLogger):
    testLoss, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for _, (X, Y) in enumerate(testDataLoader):
                X, Y = X.to(device), Y.to(device)
                pred = model(X)
                testLoss += lossFn(pred, Y).item()
                correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
    flops = FlopCountAnalysis(model, X)            
    testLoss /= testNum
    correct /= testNum
    logMessage= f'Model:{modelName}, EpochTrained:{epochNum}, ' \
                f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                f'AddInfo: {addInfo}'
    profLog = f"Model:{modelName}, AddInfo: {addInfo}\n {flop_count_table(flops)}"
    print(logMessage)
    testLogger.info(logMessage)
    profLogger.info(profLog)
    torch.save(model.state_dict(), os.path.join(
    chkPtPath, 'test-{}-{}{}.chkpt'.format(modelName, epochNum, addInfo)))
    return

def testSNet(sModel, testDataLoader, device,
            lossFn, numSteps, testNum, epochNum, 
            chkPtPath, modelName, addInfo,
            testLogger, profLogger):
    testLoss = torch.zeros((1), dtype=torch.float, device=device)
    correct = 0
    testLossHist = []
    sModel.eval()
    with torch.no_grad():
        for _, (X, Y) in enumerate(testDataLoader):
                X, Y = X.to(device), Y.to(device)      
                testSpk, testMem = sModel(X)
                _, pred = testSpk.sum(dim=0).max(1)
                for step in range(numSteps):
                    testLoss += lossFn(testMem[step], Y)
                testLossHist.append(testLoss.item())
                correct += (pred==Y).type(torch.float).sum().item()
    flops = FlopCountAnalysis(sModel, X)
    correct /= testNum
    testLoss = testLossHist[-1]/testNum
    logMessage= f'Model:{modelName}, EpochTrained:{epochNum}, ' \
                f'Acc: {(100*correct):>0.1f}%, AvgLoss: {testLoss:>8f}, ' \
                f'AddInfo: {addInfo}'
    profLog = f"Model:{modelName}, AddInfo: {addInfo}\n {flop_count_table(flops)}"
    print(logMessage)
    testLogger.info(logMessage)
    profLogger.info(profLog)
    torch.save(sModel.state_dict(), os.path.join(
    chkPtPath, 'test-{}-{}{}.chkpt'.format(modelName, epochNum, addInfo)))
    return