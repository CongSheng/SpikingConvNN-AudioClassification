import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import utils as sutils
from snntorch.functional import quant
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat

class customNet(nn.Module):
    ## Set max shape = (32, 32)
    def __init__(self, num_class = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # (6 ,30, 30)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 15, 15)
        self.conv2 = nn.Conv2d(6, 16, 3) # (16, 13, 13)
        self.fc1 = nn.Linear(576, 128) #576 for full, 1350 for topconv only
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class customSNet(nn.Module):
    def __init__(self, num_steps, beta, spike_grad=snn.surrogate.fast_sigmoid(slope=25), num_class=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(12544, 128) # 12544 for no pooling, 2704 for 1 pooling, 576 for 2 poolings
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(64, num_class)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky() 
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []

        for step in range(self.num_steps):
            # cur1 = self.pool(self.conv1(x))
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # cur2 = self.pool(self.conv2(spk1))
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.view(batch_size_curr, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)

class customSNetv2(nn.Module):
    def __init__(self, num_steps, beta, spike_grad=snn.surrogate.fast_sigmoid(slope=25), num_class=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(5400, 128) # 12544 for no pooling, 2704 for 1 pooling, 576 for 2 poolings
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(128, 64)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(128, num_class)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky() 
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []

        for step in range(self.num_steps):
            # cur1 = self.pool(self.conv1(x))
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # cur2 = self.pool(self.conv2(spk1))
            # cur2 = self.conv2(spk1)
            # spk2, mem2 = self.lif2(cur1, mem2)
            cur3 = self.fc1(spk1.view(batch_size_curr, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            # cur4 = self.fc2(spk3)
            # spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk3)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)

        return torch.stack(spk5_rec), torch.stack(mem5_rec)

### Quantized versions of the CNN ###
class qtCNet(nn.Module):
    ## Set max shape = (32, 32)
    def __init__(self, num_class = 10):
        super(qtCNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(1, 6, 3) # (6 ,30, 30)
        self.pool = nn.MaxPool2d(2, 2)  # (6, 15, 15)
        self.conv2 = nn.Conv2d(6, 16, 3) # (16, 13, 13)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_class)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x= self.dequant(x)
        return x

class qtSNetBasic(nn.Module):
    def __init__(self, num_steps, beta, nBits=8, spike_grad=snn.surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        self.num_steps = num_steps
        self.nBits = nBits
        self.spike_grad = spike_grad
        self.conv1 = qnn.QuantConv2d(1, 6, 3, weight_bit_width=self.nBits)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = qnn.QuantConv2d(6, 16, 3, weight_bit_width=self.nBits)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = qnn.QuantLinear(12544, 128, bias=False, weight_bit_width=self.nBits) # 12544 for no pooling, 2704 for 1 pooling, 576 for 2 poolings
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = qnn.QuantLinear(128, 64, bias=False, weight_bit_width=self.nBits)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = qnn.QuantLinear(64, 10, bias=False, weight_bit_width=self.nBits)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky() 
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []
        for step in range(self.num_steps):
            # cur1 = self.pool(self.conv1(x))
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # cur2 = self.pool(self.conv2(spk1))
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.view(batch_size_curr, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
        return torch.stack(spk5_rec), torch.stack(mem5_rec)

class qtSNet(nn.Module):
    def __init__(self, num_steps, beta, nBits=8, spike_grad=snn.surrogate.fast_sigmoid(slope=25)):
        super().__init__()
        # Attributes declaration
        self.num_steps = num_steps
        self.nBits = nBits
        self.spike_grad = spike_grad
        self.qState = quant.state_quant(nBits, uniform=True)
        # Layers definition
        self.conv1 = qnn.QuantConv2d(1, 6, 3, weight_bit_width=self.nBits)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, state_quant=self.qState)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = qnn.QuantConv2d(6, 16, 3, weight_bit_width=self.nBits)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, state_quant=self.qState)
        self.fc1 = qnn.QuantLinear(12544, 128, bias=False, weight_bit_width=self.nBits) # 12544 for no pooling, 2704 for 1 pooling, 576 for 2 poolings
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, state_quant=self.qState)
        self.fc2 = qnn.QuantLinear(128, 64, bias=False, weight_bit_width=self.nBits)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, state_quant=self.qState)
        self.fc3 = qnn.QuantLinear(64, 10, bias=False, weight_bit_width=self.nBits)
        self.lif5 = snn.Leaky(beta=beta, spike_grad=spike_grad, state_quant=self.qState)

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        batch_size_curr = x.shape[0]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky() 
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()

        # Record the final layer
        spk5_rec = []
        mem5_rec = []
        for step in range(self.num_steps):
            # cur1 = self.pool(self.conv1(x))
            cur1 = self.conv1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            # print(f"First layer OK, cur1 is: {cur1.is_valid}")
            # cur2 = self.pool(self.conv2(spk1))
            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            # print(f"Second layer OK, cur2 is {cur2.is_valid}")
            cur3 = self.fc1(spk2.view(batch_size_curr, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            cur5 = self.fc3(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)
            
            spk5_rec.append(spk5)
            mem5_rec.append(mem5)
        return torch.stack(spk5_rec), torch.stack(mem5_rec)