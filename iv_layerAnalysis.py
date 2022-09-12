import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import snntorch as snn
from models.CustomCNN import customNet, customSNet

# Device config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")
INPUT_SHAPE = (32, 32)

def plotFilter(layerWeight, figName, figPath, nnMode):
    fig = plt.figure(figsize=(10, 8))
    for filter in range(layerWeight.shape[0]):
        kernelWeight = np.squeeze(layerWeight[filter], axis=0)
        xlim, ylim = kernelWeight.shape
        plt.imshow(kernelWeight, cmap='hot', vmin=0, vmax=1, extent=(0, xlim, 0, ylim))
        plt.colorbar()
        plt.savefig(os.path.join(figPath, f"{nnMode}_{figName}filter{filter}_weights.png"))
        plt.clf()
    return

def plotFilterOut(layer, input, label, figPath, nnMode):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(input.cpu().numpy().squeeze(0))
    plt.colorbar()
    plt.savefig(os.path.join(figPath, f"{nnMode}_input{label}.png"))
    # output = layer(input).cpu().detach().numpy()
    output = layer(input)
    lif1 = snn.Leaky(beta=0.9, init_hidden=True)
    output = lif1.forward(output).cpu().detach().numpy()
    print(f"Output Shape: {output.shape}")
    for i in range(output.shape[0]):
        plt.clf()
        fileName = f"{nnMode}_{label}output_filter{i}_spikes.png"
        plt.imshow(output[i])
        plt.colorbar()
        plt.savefig(os.path.join(figPath, fileName))
    return

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

    onesInput = torch.full(INPUT_SHAPE, 1).to(device)
    dataPath = args.dataPath
    if dataPath != "None":
        realInput = torch.load(dataPath)
        labelName = dataPath.split('/')[-1].split('_')[0]
        realInput = realInput.to(device)
    
    layerSelected = modelChildren[args.layerType]
    print(f"Layer type: {type(layerSelected)}")
    layerWeight = layerSelected.weight.detach().numpy()
    figName = str({layerSelected})
    plotFilter(layerWeight, figName, args.figPath, nnMode)
    plotFilterOut(layerSelected.to(device), realInput, labelName, args.figPath, nnMode)
    
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