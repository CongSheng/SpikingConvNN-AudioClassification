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
        encodingMode = "rate",
        tensorType = torch.float32
    ) -> None:
        super(LIFNeuron, self).__init__()
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
        self.encodingMode = encodingMode

    def forward(self, x):
        for step in range(self.timeSteps):
            self.inject(x)
        spkHistStacked = torch.stack(self.spkHist)
        encoded = self.spikeEncode(spkHistStacked)
        return torch.stack(self.memHist), spkHistStacked, self.mem, encoded

    def fire(self):
        memDiff = self.mem - self.threshold
        spike = torch.where(memDiff >= 0, 1, 0)
        return memDiff, spike

    def resetMem(self, spk):
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
        self.updateMemV(x, refMask)
        memDiff, spk = self.fire()
        self.refCounter[spk==1] = self.refTime
        # print(f"Membrane voltage: {self.mem}, spike: {spk} ")
        self.memHist.append(self.mem)
        self.spkHist.append(spk)
        self.resetMem(spk)
        self.refCounter = self.refCounter - self.oneDecrem * refMaskInv
        return

    def spikeEncode(self, spkHist):
        spkCount = torch.sum(spkHist, 0) 
        if self.encodingMode == "rate":
            return spkCount/self.timeSteps
        if self.encodingMode == "count":
            return spkCount
    
    
    
    
    

