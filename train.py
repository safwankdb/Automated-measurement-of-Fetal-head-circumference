# import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
# import os
import copy
from unet_model import UNet
from data_loader import HC18

train_set = HC18('train')
print('Train Set loaded')
val_set = HC18('val')
print('Validation Set loaded')
test_set = HC18('test')
print('Test Set loaded')


dataset = {0: train_set, 1: val_set}

dataloaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=4, shuffle=True, num_workers=0)
               for x in range(2)}
# print(dataloaders[0])

dataset_sizes = {x: len(dataset[x]) for x in range(2)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch ' + str(epoch) + ' running')

        for phase in range(2):
            if phase == 0:
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            # print(len(dataloaders[phase]))
            for i, Data in enumerate(dataloaders[phase]):
                inputs, labels = Data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 0):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 0:
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 1 and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('End of epoch')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


model = UNet(1, 1)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model_optim = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(model_optim, step_size=7, gamma=0.1)
model = train_model(model, criterion, model_optim,
                    exp_lr_scheduler, num_epochs=25)
