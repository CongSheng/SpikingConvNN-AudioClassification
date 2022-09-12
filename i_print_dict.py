import os
import argparse
import torch

def printDict(stateDict):
    for key in stateDict:
        print(key, "\t", stateDict[key].size())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--NAME', type=str, default="test-CustomNet-5.chkpt")
    args = parser.parse_args()
    print(args)
    statePath = os.path.join('checkpoints/', args.NAME)
    stateDict = torch.load(statePath)
    printDict(stateDict)

