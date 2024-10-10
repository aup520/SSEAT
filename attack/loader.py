import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
# from autoaugment import CIFAR10Policy
class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def get_loaders(dir_, batch_size, DATASET='CIFAR100'):
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # CIFAR10Policy(),
            transforms.ToTensor(),
        ])
    # train_transform.transforms.append(CutoutDefault(int(32 / 2)))
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    num_workers = 1

    if DATASET == 'CIFAR10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif DATASET == 'CIFAR100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )
    return train_loader, test_loader