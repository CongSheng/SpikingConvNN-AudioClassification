import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    def __init__(
        self,
        inputShape,
        timeSteps,
        device,
        tauMem  = 10,
        uRest = -75,
        refTime = 2,
        resMem = 0.1,
        threshold = -55,
        resetMode = "subtract",
        tensorType = torch.float32
    ) -> None:
        super(LIFNeuron, self).__init__()
        #self.decay = decay
        self.tauMem = tauMem
        self.uRest = uRest
        self.threshold = threshold
        self.resetMode = resetMode
        self.timeSteps = timeSteps
        self.refTime = refTime
        self.resMem = resMem
        self.tensorType = tensorType
        self.mem = torch.full(inputShape, self.uRest, dtype= tensorType, device=device)
        self.memHist = []
        self.spkHist = []
        self.refCounter = torch.zeros(inputShape, dtype=tensorType, device=device)
        self.oneDecrem = torch.ones(inputShape, dtype=tensorType, device=device)

    def fire(self):
        memDiff = self.mem - self.threshold
        spike = torch.where(memDiff >= 0, 1, 0)
        return memDiff, spike

    def resetMem(self, memDiff, spk):
        if self.resetMode == "subtract":
            self.mem = self.mem - spk * self.threshold
        if self.resetMode == "zero":
            self.mem[spk==1] = 0
        if self.resetMode == "rest":
            self.mem[spk==1] = self.uRest
        return 

    def updateMemV(self, x, refMask):
        dv = (1/self.tauMem) * (-(self.mem - self.uRest) + x*self.resMem*refMask)
        self.mem = self.mem + dv
        return
    
    def inject(self, x):
        refMask = torch.where(self.refCounter>0, 0, 1 )
        refMaskInv = torch.where(self.refCounter>0, 1, 0 )
        #self.mem = self.mem * self.decay + (1 - self.decay) * x * refMask
        self.updateMemV(x, refMask)
        memDiff, spk = self.fire()
        self.refCounter[spk==1] = self.refTime
        self.resetMem(memDiff, spk)
        self.memHist.append(self.mem)
        self.spkHist.append(spk)
        self.refCounter = self.refCounter - self.oneDecrem * refMaskInv
        return

    def forward(self, x):
        for step in range(self.timeSteps):
            self.inject(x)
            #print(self.mem)
        return torch.stack(self.memHist), torch.stack(self.spkHist), self.mem
    
    
    
    

