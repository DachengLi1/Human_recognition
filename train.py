import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from datasets import HumanRecognitionDataset
import torchvision.transforms as transforms
from model import SqueezeNet
from torchvision.models import resnet18
import random
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models

transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = HumanRecognitionDataset(transform = transform,train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

testset = HumanRecognitionDataset(transform = transform,train=False)
testloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

random.seed(0)
torch.manual_seed(0)
cudnn.benchmark = True

def train():   
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print(loss)
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d/%5d] loss: %.3f' %
                  (epoch + 1, i + 1, len(trainloader), running_loss / 10))
            running_loss = 0.0
    
def test():
    for i, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print(loss)
        if i % 10 == 9:    # print every 10 mini-batches
            print('[%d, %5d/%5d] loss: %.3f' %
                  (epoch + 1, i + 1, len(trainloader), running_loss / 10))
            running_loss = 0.0
    

device = torch.device('cuda:0')
criterion = F.binary_cross_entropy_with_logits 
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1)
net = net.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9,0.999))

scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(10):
    train()
    test()
    schedule.step()

print('Finished Training')
