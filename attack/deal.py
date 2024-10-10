
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from wideresnet import WideResNet
# from advertorch.context import ctx_noparamgrad_and_eval
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import torchattacks
# from autoaugment import CIFAR10Policy
from resnet import ResNet18
class ImageDataset(Dataset):
    
    def __init__(self, data_frame: pd.DataFrame, dataset: str, transform=None):
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_frame.iloc[idx]["file_name"]
        imten = self.data_frame.iloc[idx]["data"]
        label = self.data_frame.iloc[idx].get("label", -1)
        data=imten
        label=label
        # arr=data.permute(1, 2, 0).numpy()
        # image = Image.fromarray((arr * 255).astype(np.uint8))
        # img_path = os.path.join("dataset", self.dataset, img_name)
        # image = PIL.Image.open(img_path).convert("RGB")
        # if self.transform:
        #     image = self.transform(image)
        image=transforms.ToPILImage()(data).convert('RGB')
        sample["image1"] = self.transform(image)
        sample["image2"] = self.transform(image)
        # sample["image"] = self.transform(image)
        sample["label"] = label
        sample["image_name"] = img_name
        return sample
model = ResNet18()
model.load_state_dict(torch.load('/home/workspace/wcl/try-cifar100/attack/model/clntraincifar100.pt4'))
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
device = torch.device("cuda")
x=["FGSM","BIM","PGD","RFGSM","MIFGSM","NIFGSM","SINIFGSM","DIFGSM","VNIFGSM","VMIFGSM"]
trainlt=[]
loaded_data = torch.load('/home/workspace/wcl/try-cifar100/attack/data/clntr.pth')
trdata_list =  loaded_data['advdata_list']
trlabel_list = loaded_data['advdata_labellt']
train_list=list(zip(trdata_list, trlabel_list))
alist=[]
# cls=["truck","automobile","frog","airplane","cat","bird","dog","horse","deer","ship"]
cls = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
nu=0
for batch_idx,(data, target) in enumerate(train_list):
    for k in range(data.size()[0]):
        alist.append(dict(klass=cls[target[k]],file_name=nu,data=data[k].cpu(),label=target[k].cpu(),task=0))
        nu+=1
df=pd.DataFrame(alist)
for j in range(100):
    random_rows=df[df["label"]==j].sample(n=10, random_state=42)
    trainlt+=random_rows.to_dict(orient="records")
tr_dataset = ImageDataset(
                pd.DataFrame(trainlt),
                dataset="cifar100",
                transform=transforms.Compose([transforms.ToTensor()])
            )            
tr_loader = DataLoader(     
                tr_dataset,
                shuffle=True,
                batch_size=64,
                num_workers=0,
            )
print(1)
for dt in tr_loader:
    cln_data,true_label = dt["image1"],dt["label"]
    cln_data=cln_data.to(device)
    # data = data.to(device)
    true_label = true_label.to(device)
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
x=["PGD","VNIFGSM","FGSM","BIM","RFGSM","MIFGSM","DIFGSM","NIFGSM","SINIFGSM","VMIFGSM"]
torch.save({'advdata_list': advdata_list1, 'advdata_labellt': advdata_labellt1}, './data/PGDlltr.pth')
torch.save({'advdata_list': advdata_list2, 'advdata_labellt': advdata_labellt2}, './data/VNIFGSMlltr.pth')
torch.save({'advdata_list': advdata_list3, 'advdata_labellt': advdata_labellt3}, './data/FGSMlltr.pth')
torch.save({'advdata_list': advdata_list4, 'advdata_labellt': advdata_labellt4}, './data/BIMlltr.pth')
torch.save({'advdata_list': advdata_list5, 'advdata_labellt': advdata_labellt5}, './data/RFGSMlltr.pth')
torch.save({'advdata_list': advdata_list6, 'advdata_labellt': advdata_labellt6}, './data/MIFGSMlltr.pth')
torch.save({'advdata_list': advdata_list7, 'advdata_labellt': advdata_labellt7}, './data/DIFGSMlltr.pth')
torch.save({'advdata_list': advdata_list8, 'advdata_labellt': advdata_labellt8}, './data/NIFGSMlltr.pth')
torch.save({'advdata_list': advdata_list9, 'advdata_labellt': advdata_labellt9}, './data/SINIFGSMlltr.pth')
torch.save({'advdata_list': advdata_list10, 'advdata_labellt': advdata_labellt10}, './data/VMIFGSMlltr.pth')  




    

        
