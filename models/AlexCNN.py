## AlexNet architecture obtained from https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html
## Serve as main reference for spiking conversion
import snntorch as snn
from snntorch import utils as sutils
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexSpikingNet:
    def __init__(self, device, beta, spike_grad, num_classes: int = 10, dropout: float = 0.5) -> None:
        self.net= nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(36, 4096),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Linear(4096, num_classes),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        ).to(device)
    
    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        spk_rec = []
        mem_rec = []
        sutils.reset(self.net)

        for step in range(x.size(0)):
            print(x.size(0))
            spk_out, mem_out = self.net(x[step])
            print(f"Shape of spike: {spk_out.shape}")
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        return torch.stack(spk_rec), torch.stack(mem_rec)