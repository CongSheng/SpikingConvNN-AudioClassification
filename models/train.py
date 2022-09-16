from cProfile import label
from tqdm import tqdm
import torch
import os
from snntorch import functional as SF

def trainNet(device, model, train_dl, epoch_num, optimizer, criterion, 
            train_loss_hist, train_accu_hist, checkpoint_path, modelName):
    for epoch in tqdm(range(epoch_num)):
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (X, Y) in enumerate(train_dl):
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
       
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += Y.size(0)
            correct += predicted.eq(Y).sum().item()

        train_loss = running_loss/len(train_dl)
        train_accu = 100* correct/total
        train_loss_hist.append(train_loss)
        train_accu_hist.append(train_accu)
        print(f' Epoch: {epoch} | Train Loss: {train_loss:.3f} | Accuracy: {train_accu:.3f}')
    print("-----Finished Training-----")
    torch.save(model.state_dict(), os.path.join(
        checkpoint_path, 'train--{}-{}.chkpt'.format(modelName, epoch_num)
    ))
    return model, train_loss_hist, train_accu_hist

def trainSNet(device, model, train_dl, epoch_num, optimizer, loss_fn, num_steps,
            train_loss_hist, train_accu_hist, checkpoint_path, modelName):
    for epoch in range(epoch_num):
        iterCount = 0
        for i, (data, targets) in enumerate(iter(train_dl)):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            model.train()
            spk_rec, mem_rec = model(data)

            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(num_steps):
                loss_val += loss_fn(mem_rec[step], targets)

            # Gradient calculation + weight update
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            train_loss_hist.append(loss_val.item())
            acc = SF.accuracy_rate(spk_rec, targets) 
            train_accu_hist.append(acc)
            iterCount +=1
        print(f' Epoch: {epoch} | Train Loss: {train_loss_hist[-1]:.3f} | Accuracy: {train_accu_hist[-1]:.3f} | Iteration: {iterCount}')

    print("-----Finished Training-----")
    torch.save(model.state_dict(), os.path.join(
        checkpoint_path, 'train--{}-{}.chkpt'.format(modelName, epoch_num)
    ))
    return model, train_loss_hist, train_accu_hist, iterCount
        
def qatrainSNet(net, epochNum, stepNum, trainloader, criterion, optimizer, addInfo=None, gradClip=False, weightClip=False, device="cpu", scheduler=None, checkpoint_path=None):
    """Complete one epoch of training."""
    net.train()
    lossHist = []
    lrHist = []
    accuHist = []
    correct = 0
    for epoch in tqdm(range(epochNum)):
        i = 0
        for data, labels in trainloader:
            i+=1
            data, labels = data.to(device), labels.to(device)
            spk_rec, mem_rec = net(data)
            loss_val = torch.zeros((1), dtype=torch.float, device=device)
            for step in range(stepNum):
                loss_val += criterion(mem_rec[step], labels)
            optimizer.zero_grad()
            loss_val.backward()

            ## Enable gradient clipping
            if gradClip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)

            ## Enable weight clipping
            if weightClip:
                with torch.no_grad():
                    for param in net.parameters():
                        param.clamp_(-1, 1)
            optimizer.step()
            scheduler.step()
            lossHist.append(loss_val.item())
            lrHist.append(optimizer.param_groups[0]["lr"])
        acc = SF.accuracy_rate(spk_rec, labels) 
        accuHist.append(acc)
        print(f' Epoch: {epoch} | Train Loss: {lossHist[-1]:.3f} | Accuracy: {acc:.3f}')
    print("-----Finished Training-----")
    if checkpoint_path is not None:
        torch.save(net.state_dict(), os.path.join(checkpoint_path, f'train-qtModel-{epochNum}-{addInfo}.chkpt'))
    return net, lossHist, lrHist
    