from utils import spikingNeuron

import matplotlib.pyplot as plt
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} on {torch.cuda.get_device_name(0)} :D ")

if __name__ == '__main__':
    inputShape = (2,2)
    x = torch.tensor([[220, 75],[0, 500]], device=device)
    print(f"Tensor to inject: \n{x}")
    spkLayer = spikingNeuron.LIFNeuron(inputShape, 100, device, tauMem=12, resetMode="rest")
    hist, spkHist, mem = spkLayer(x)

    memVolt = hist.cpu().detach().numpy()
    spkHist = spkHist.cpu().detach().numpy()
    fig, ax1 = plt.subplots()
    print(memVolt.shape)
    ax1.plot(memVolt[:,1, 1])
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Membrane voltage (mV)')
    ax2 = ax1.twinx() 
    ax2.plot(spkHist[:,1, 1], color = 'red')
    ax2.set_ylabel("Spikes")  
    plt.show()