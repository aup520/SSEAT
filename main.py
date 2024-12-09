
import logging.config
import os
import random
from collections import defaultdict

import numpy as np
import torch
from randaugment import RandAugment
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd

from configuration import config
from utils.augment import Cutout, select_autoaugment
from utils.data_loader import get_test_datalist, get_statistics
from utils.data_loader import get_train_datalist
from utils.method_manager import select_method
from utils.augment       import CIFAR10Policy
class CutoutDefault(object):
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

def main():
    args = config.base_parser()

    # Save file name
    tr_names = ""
    for trans in args.transforms:
        tr_names += "_" + trans
    save_path = f"{args.dataset}/{args.mode}_{args.mem_manage}_{args.stream_env}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter("tensorboard")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")

    
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)

    mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset)
    train_transform = []
    if "cutout" in args.transforms:
        train_transform.append(Cutout(size=16))
    if "randaug" in args.transforms:
        train_transform.append(RandAugment())
    if "autoaug" in args.transforms:
        train_transform.append(select_autoaugment(args.dataset))
    
   
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
    ])
    train_transform.transforms.append(CutoutDefault(int(32 / 2)))
    
    logger.info(f"Using train-transforms {train_transform}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((inp_size, inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    method = select_method(
        args, criterion, device, train_transform, test_transform, n_classes
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)

    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return

        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")

        task_acc = 0.0
        eval_dict = dict()

        logger.info("[2-2] Set environment for the current task")
        
        method.before_task(cur_iter, args.init_model, args.init_opt)

 
        logger.info(f"[2-3] Start to train under {args.stream_env}")

        if args.stream_env == "offline" or args.mode == "joint" or args.mode == "gdumb":
            
            method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            torch.save(method.model.state_dict(),os.path.join("/home/workspace/wcl/atcode/model", "basedatamintrain.pt3"))
           
        print(cur_iter)
       

if __name__ == "__main__":
    main()
