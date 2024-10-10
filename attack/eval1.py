import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from loader import get_loaders
# from resnet import PreActResNet18
from resnet import ResNet18_FSR
import torch.nn.functional as F
x=["cln","FGSM","BIM","PGD","RFGSM","MIFGSM","NIFGSM","SINIFGSM","DIFGSM","VNIFGSM","VMIFGSM"]
TRAINED_MODEL_PATH="/home/workspace/wcl/FSR-main/weights/cifar100/resnet18/"
filename="cifar100_resnet18.pth"
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18_FSR(num_classes=100)
model.load_state_dict(torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
model.to(device)
model.eval()
test_list=[]
for t in range(len(x)):
    loaded_data = torch.load('/home/workspace/wcl/try-cifar100/attack/data/{}lte.pth'.format(x[t]))
    tedata_list = loaded_data['advdata_list']
    telabel_list = loaded_data['advdata_labellt']
    test_list.append([tedata_list,telabel_list])

for i in range(len(x)):
    test_clnloss = 0
    clncorrect = 0
    print(x[i])
    for batch_idx, (clndata, target) in enumerate(zip(test_list[i][0], test_list[i][1])):
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        # print(clndata.size())
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()
    test_clnloss /= 10000
    print('\nTASK{} Test set: avg cln loss: {:.4f},'
        ' cln acc: {}/{} ({:.0f}%)'.format(
            i,test_clnloss, clncorrect, 10000,
            100. * clncorrect / 10000))