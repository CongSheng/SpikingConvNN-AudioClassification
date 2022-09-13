import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
from models.CustomCNN import customNet, customSNet
from utils.plotFigure import plotFilter

# Device config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")
INPUT_SHAPE = (32, 32)

def countSparsity(arrayIn):
    nonZeroCount = np.count_nonzero(arrayIn)
    zeroCount = np.count_nonzero(arrayIn==0)
    nonZeroPercent = nonZeroCount / arrayIn.size * 100
    zeroPercent = zeroCount / arrayIn.size * 100
    return nonZeroCount, zeroCount, nonZeroPercent, zeroPercent

def plotFilterOut(layer, input, label, figPath, nnMode):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(input.cpu().numpy().squeeze(0))
    plt.colorbar()
    plt.rcParams['font.size'] = '16'
    plt.savefig(os.path.join(figPath, f"{nnMode}_input{label}.png"))
    # output = layer(input).cpu().detach().numpy()
    if nnMode == "SCNN":
        output = layer(input)
        lif1 = snn.Leaky(beta=0.9, init_hidden=True)
        output = lif1.forward(output).cpu().detach().numpy()
        plotType = "Spikes"
    else:
        plotType = "MFCC"
        output = layer(input).cpu().detach().numpy()
    print(f"Output Shape: {output.shape}")
    for i in range(output.shape[0]):
        currOut = output[i]
        _, _, percentNonZero, percentZero = countSparsity(currOut)
        titlePlot = f"{plotType} for \"{label}\" - Zeros: {percentZero:.2f} %, Nonzeros: {percentNonZero:.2f} %"
        plt.clf()
        fileName = f"{nnMode}_{label}output_filter{i}_{plotType}.png"
        print(fileName)
        plt.imshow(currOut)
        plt.colorbar()
        plt.title(titlePlot)
        plt.savefig(os.path.join(figPath, fileName))
    return

def getSpikeMembrane(model, input, label, device, plotting=False, savePath="featuresFigure/featureVisual"):
    model.to(device).eval()
    plt.rcParams['font.size'] = '16'
    with torch.no_grad():
        spikeRecord, memVoltage = model(input)
        _, pred = spikeRecord.sum(dim=0).max(1)
        print(f"Predicted: {pred.cpu().detach().numpy()}, Actual: {label}")
        spikeRecord = spikeRecord.cpu().detach().numpy().squeeze(1)
        memRecord = memVoltage.cpu().detach().numpy().squeeze(1)
        neuronList = range(0, 10)

        print(f"Record of spikes: {spikeRecord.shape} and voltage: {memRecord.shape})")
        if plotting:
            fig = plt.subplots(figsize=(10, 8))
            plt.plot(np.transpose(spikeRecord))
            plt.title(f"Spikes over time for \"{label}\"")
            plt.legend(neuronList)
            plt.savefig(os.path.join(savePath, f"spikingRecord{label}.png"))
            memFig = plt.figure(figsize=(10,8))
            plt.plot(np.transpose(memRecord))
            plt.title(f"Membrane voltage over time for \"{label}\"")  
            plt.legend(neuronList)      
            plt.savefig(os.path.join(savePath, f"membraneRecord{label}.png"))
        return spikeRecord, memRecord

def layerMain(args):
    # Loading model with state_dir
    checkpoint = torch.load(args.checkptPath)
    nnMode = args.nnMode
    if nnMode == "SCNN":
        net = customSNet(num_steps=10, beta=0.5)
        print("SNN Mode activated.\n")
    elif nnMode =="CNN":
        net = customNet()
        print("CNN Mode activated.\n")
    else:
        print("Invalid model, SNN or CNN")
    net.load_state_dict(checkpoint)
    modelChildren = list(net.children())
    print(modelChildren)

    # Prepare data 
    onesInput = torch.full(INPUT_SHAPE, 1).to(device)
    dataPath = args.dataPath
    if dataPath != "None":
        realInput = torch.load(dataPath)
        labelName = dataPath.split('/')[-1].split('_')[0]
        realInput = realInput.to(device)
    
    # Plot specific layers and output
    layerSelected = modelChildren[args.layerType]
    print(f"Layer type: {type(layerSelected)}")
    layerWeight = layerSelected.weight.detach().numpy()
    figName = str({layerSelected})
    plotFilter(layerWeight, figName, args.figPath, nnMode)
    plotFilterOut(layerSelected.to(device), realInput, labelName, args.figPath, nnMode)
    
    ## Membrane analysis for SCNN
    if nnMode == "SCNN":
        spikes, mem = getSpikeMembrane(net, realInput, labelName, device, True)
        print(f"Shape of spikes {spikes.shape} and mem {mem.shape}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figPath', type=str, default="featuresFigure/featureVisual", help="Path to save the figures.")
    parser.add_argument('--checkptPath', type=str, default="checkpoints/train--CustomCNN-20.chkpt", help="Path of the checkpoint.")
    parser.add_argument('--dataPath', type=str, default="None", help="Input path of data to use for visualization.")
    parser.add_argument('--nnMode', type=str, default="CNN", help="Mode of neural network.")
    parser.add_argument('--layerType', type=int, default=0, help="Layer to examine")
    args = parser.parse_args()
    print(args)
    layerMain(args)