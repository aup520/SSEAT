import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from loader import get_loaders
from resnet import ResNet18
import torchattacks


TRAINED_MODEL_PATH="./model"
filename="clntraincifar100.pt4"
torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = ResNet18()
model.load_state_dict(
    torch.load(os.path.join(TRAINED_MODEL_PATH, filename)))
model.to(device)
model.eval()

data_list=[]
data_labellt=[]
# train_loader, test_loader= get_loaders(dir_="./cifar-data",batch_size=64)
# for cln_data, true_label in train_loader:
#     cln_data, true_label = cln_data.to(device), true_label.to(device)
#     data_list.append(cln_data)
#     data_labellt.append(true_label)
#     # print(true_label)
# torch.save({'advdata_list': data_list, 'advdata_labellt': data_labellt}, './data/clntr.pth')
# data_list=[]
# data_labellt=[]
# for cln_data, true_label in test_loader:
#     cln_data, true_label = cln_data.to(device), true_label.to(device)
#     data_list.append(cln_data)
#     data_labellt.append(true_label)
#     # print(true_label)
# torch.save({'advdata_list': data_list, 'advdata_labellt': data_labellt}, './data/clnte.pth')
# print("ready")
loaded_data = torch.load('./data/clnmtr.pth')
trdata_list =  loaded_data['advdata_list']
trlabel_list = loaded_data['advdata_labellt']

loaded_data = torch.load('./data/clnlte.pth')
tedata_list = loaded_data['advdata_list']
telabel_list = loaded_data['advdata_labellt']
ak1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
ak2 = torchattacks.VNIFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak3 = torchattacks.FGSM(model, eps=8/255)

ak4 = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
ak5 = torchattacks.RFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak6 = torchattacks.MIFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak7 = torchattacks.DIFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak8 = torchattacks.NIFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak9 = torchattacks.SINIFGSM(model, eps=8/255, alpha=2/255, steps=10)
ak10 = torchattacks.VMIFGSM(model, eps=8/255, alpha=2/255, steps=10)

advdata_list=[]
advdata_list1=[]
advdata_list2=[]
advdata_list3=[]
advdata_labellt=[]
advdata_labellt1=[]
advdata_labellt2=[]
advdata_labellt3=[]
advdata_list4=[]
advdata_labellt4=[]
advdata_list5=[]
advdata_labellt5=[]
advdata_list6=[]
advdata_labellt6=[]
advdata_list7=[]
advdata_labellt7=[]
advdata_list8=[]
advdata_labellt8=[]
advdata_list9=[]
advdata_labellt9=[]
advdata_labellt10=[]
advdata_list10=[]
nu=0
for cln_data, true_label in zip(trdata_list,trlabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    advdata1 = ak1(cln_data, true_label)
    advdata_list1.append(advdata1)
    advdata_labellt1.append(true_label)
    
    advdata2 = ak2(cln_data, true_label)
    advdata_list2.append(advdata2)
    advdata_labellt2.append(true_label)

    advdata3 = ak3(cln_data, true_label)
    advdata_list3.append(advdata3)
    advdata_labellt3.append(true_label)

    advdata4 = ak4(cln_data, true_label)
    advdata_list4.append(advdata4)
    advdata_labellt4.append(true_label)
    
    advdata5 = ak5(cln_data, true_label)
    advdata_list5.append(advdata5)
    advdata_labellt5.append(true_label)

    advdata6 = ak6(cln_data, true_label)
    advdata_list6.append(advdata6)
    advdata_labellt6.append(true_label)

    advdata7 = ak7(cln_data, true_label)
    advdata_list7.append(advdata7)
    advdata_labellt7.append(true_label)

    advdata8 = ak8(cln_data, true_label)
    advdata_list8.append(advdata8)
    advdata_labellt8.append(true_label)

    advdata9 = ak9(cln_data, true_label)
    advdata_list9.append(advdata9)
    advdata_labellt9.append(true_label)

    advdata10 = ak10(cln_data, true_label)
    advdata_list10.append(advdata10)
    advdata_labellt10.append(true_label)
    nu=nu+1
    print(nu)
    
    
x=["PGD","VNIFGSM","FGSM","BIM","RFGSM","MIFGSM","DIFGSM","NIFGSM","SINIFGSM","VMIFGSM"]
torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/PGDltr.pth')
torch.save({'advdata_list': advdata_list2, 'advdata_labellt': advdata_labellt2}, './data/VNIFGSMltr.pth')
torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/FGSMltr.pth')
torch.save({'advdata_list': advdata_list4, 'advdata_labellt': advdata_labellt4}, './data/BIMltr.pth')
torch.save({'advdata_list': advdata_list5, 'advdata_labellt': advdata_labellt5}, './data/RFGSMltr.pth')
torch.save({'advdata_list': advdata_list6, 'advdata_labellt': advdata_labellt6}, './data/MIFGSMltr.pth')
torch.save({'advdata_list': advdata_list7, 'advdata_labellt': advdata_labellt7}, './data/DIFGSMltr.pth')
torch.save({'advdata_list': advdata_list8, 'advdata_labellt': advdata_labellt8}, './data/NIFGSMltr.pth')
torch.save({'advdata_list': advdata_list9, 'advdata_labellt': advdata_labellt9}, './data/SINIFGSMltr.pth')
torch.save({'advdata_list': advdata_list10, 'advdata_labellt': advdata_labellt10}, './data/VMIFGSMltr.pth')

advdata_list=[]
advdata_list1=[]
advdata_list2=[]
advdata_list3=[]
advdata_labellt=[]
advdata_labellt1=[]
advdata_labellt2=[]
advdata_labellt3=[]
advdata_list4=[]
advdata_labellt4=[]
advdata_list5=[]
advdata_labellt5=[]
advdata_list6=[]
advdata_labellt6=[]
advdata_list7=[]
advdata_labellt7=[]
advdata_list8=[]
advdata_labellt8=[]
advdata_list9=[]
advdata_labellt9=[]
advdata_labellt10=[]
advdata_list10=[]

for cln_data, true_label in zip(tedata_list,telabel_list):
    cln_data, true_label = cln_data.to(device), true_label.to(device)
    advdata1 = ak1(cln_data, true_label)
    advdata_list1.append(advdata1)
    advdata_labellt1.append(true_label)
    
    advdata2 = ak2(cln_data, true_label)
    advdata_list2.append(advdata2)
    advdata_labellt2.append(true_label)

    advdata3 = ak3(cln_data, true_label)
    advdata_list3.append(advdata3)
    advdata_labellt3.append(true_label)

    advdata4 = ak4(cln_data, true_label)
    advdata_list4.append(advdata4)
    advdata_labellt4.append(true_label)
    
    advdata5 = ak5(cln_data, true_label)
    advdata_list5.append(advdata5)
    advdata_labellt5.append(true_label)

    advdata6 = ak6(cln_data, true_label)
    advdata_list6.append(advdata6)
    advdata_labellt6.append(true_label)

    advdata7 = ak7(cln_data, true_label)
    advdata_list7.append(advdata7)
    advdata_labellt7.append(true_label)

    advdata8 = ak8(cln_data, true_label)
    advdata_list8.append(advdata8)
    advdata_labellt8.append(true_label)

    advdata9 = ak9(cln_data, true_label)
    advdata_list9.append(advdata9)
    advdata_labellt9.append(true_label)

    advdata10 = ak10(cln_data, true_label)
    advdata_list10.append(advdata10)
    advdata_labellt10.append(true_label)
    print(1)

torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/PGDlte.pth')
torch.save({'advdata_list': advdata_list2, 'advdata_labellt': advdata_labellt2}, './data/VNIFGSMlte.pth')
torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/FGSMlte.pth')
torch.save({'advdata_list': advdata_list4, 'advdata_labellt': advdata_labellt4}, './data/BIMtle.pth')
torch.save({'advdata_list': advdata_list5, 'advdata_labellt': advdata_labellt5}, './data/RFGSMlte.pth')
torch.save({'advdata_list': advdata_list6, 'advdata_labellt': advdata_labellt6}, './data/MIFGSMlte.pth')
torch.save({'advdata_list': advdata_list7, 'advdata_labellt': advdata_labellt7}, './data/DIFGSMlte.pth')
torch.save({'advdata_list': advdata_list8, 'advdata_labellt': advdata_labellt8}, './data/NIFGSMlte.pth')
torch.save({'advdata_list': advdata_list9, 'advdata_labellt': advdata_labellt9}, './data/SINIFGSMlte.pth')
torch.save({'advdata_list': advdata_list10, 'advdata_labellt': advdata_labellt10}, './data/VMIFGSMlte.pth')



