
import logging
import random
import torch.optim as optim
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from methods.finetune import Finetune
from utils.data_loader import cutmix_data, ImageDataset
from methods.resnet import ResNet18
logger = logging.getLogger()
writer = SummaryWriter("tensorboard")
def kdloss(logit1, logit2, T=1.):
    loss=(- F.softmax(logit1 / T, dim=-1).detach() * F.log_softmax(logit2 / T, dim=-1)).mean(0).sum() * T ** 2
    return loss
def _kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')
def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5

def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class SSEAT(Finetune):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )

        self.batch_size = kwargs["batchsize"]
        self.n_worker = kwargs["n_worker"]
        self.exp_env = kwargs["stream_env"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "uncertainty"

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=0):
        if len(self.memory_list) > 0:
            mem_dataset = ImageDataset(
                pd.DataFrame(self.memory_list),
                dataset=self.dataset,
                transform=self.train_transform,
            )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=8,
                num_workers=n_worker,
            )
            # stream_batch_size = batch_size - batch_size // 2
        else:
            memory_loader = None
            # stream_batch_size = batch_size
        
        # train_list == streamed_list in RM
        # train_list = self.streamed_list
        # test_list = self.test_list
        # random.shuffle(train_list)
        # Configuring a batch with streamed and memory data equally.
        # train_loader, test_loader = self.get_dataloader(
        #     stream_batch_size, n_worker, train_list, test_list,cur_iter
        # )
        train_loader, test_loader,alist = self.get_dataloader(cur_iter)
        self.streamed_list=alist
        # logger.info(f"Streamed samples: {len(self.streamed_list)}")
        # logger.info(f"In-memory samples: {len(self.memory_list)}")
        # logger.info(f"Train samples: {len(train_list)+len(self.memory_list)}")
        # logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        self.model = self.model.to(self.device)
        best_adv=0
        if cur_iter==0:
            return
        
        for epoch in range(n_epoch):
            # # initialize for each task
            # if epoch <= 0:  # Warm start of 1 epoch
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr * 0.1
            # elif epoch == 1:  # Then set to maxlr
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr
            # else:  # Aand go!
            #     self.scheduler.step()
            # if epoch <= 0:
            #     for param_group in self.optimizer.param_groups:
            #         param_group["lr"] = self.lr
            # else:
            #     self.scheduler.step()
            adv=0

            # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            # scheduler = MultiStepLR(self.optimizer, milestones=[10], gamma=0.1)
            self._train(train_loader=train_loader, memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion,cur_iter=cur_iter)
            self.model.eval()
            for i in range(cur_iter+1):
                test_clnloss = 0
                clncorrect = 0
                for batch_idx, (clndata, target) in enumerate(zip(self.test_list[i][0], self.test_list[i][1])):
                    clndata, target = clndata.to(self.device), target.to(self.device)
                    with torch.no_grad():
                        output = self.model(clndata)
                    test_clnloss += F.cross_entropy(
                        output, target, reduction='sum').item()
                    pred = output.max(1, keepdim=True)[1]
                    clncorrect += pred.eq(target.view_as(pred)).sum().item()
                test_clnloss /= 10000
                print('\n epoch{} ADV{} Test set: avg cln loss: {:.4f},'
                    ' cln acc: {}/{} ({:.0f}%)'.format(
                        epoch,i,test_clnloss, clncorrect, 10000,
                        100. * clncorrect / 10000))
                adv=adv+100. * clncorrect / 10000
            adv=adv/(cur_iter+1)
            if adv>best_adv:
                best_adv=adv
                best_epoch=epoch
            self.scheduler.step()
        print("best_epoch:{},best_adv:{}".format(best_epoch,best_adv))
        
            
            # # writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            # # writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            # # writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            # # writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            # # writer.add_scalar(
            # #     f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            # # )

            # logger.info(
            #     f"Task {cur_iter} | Epoch {epoch+1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            #     f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
            #     f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            # )

            # best_acc = max(best_acc, eval_dict["avg_acc"])
        self.already_mem_update = False
        # return best_acc, eval_dict

    def update_model(self, x1,x2, y, criterion, optimizer,cur_iter,mode):
        optimizer.zero_grad()

        # do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        # if do_cutmix:
        #     x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
        #     logit = self.model(x)
        #     loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
        #         logit, labels_b
        #     )
        # else:
        model2 = ResNet18()
        model2.load_state_dict(torch.load('/home/workspace/wcl/atcode/model/basedatamintrain.pt3'))
        model2=model2.cuda()
        outputp1 = self.model(x1)
        outputp2 = model2(x2)
        outputp2=outputp2.detach()
        # loss = criterion(logit, y)
        # a=0.5
        # b=1
        if mode==2:
            loss = 0.5*(F.cross_entropy(outputp1, y, reduction='mean')+F.cross_entropy(
                        outputp2, y, reduction='mean'))+_jensen_shannon_div(outputp2, outputp1,1)
            # loss = 0.5*(F.cross_entropy(outputp1, y, reduction='mean')+F.cross_entropy(
            #             outputp2, y, reduction='mean'))+kdloss(outputp2, outputp1,1)
            # print("kd的比例：a={},b={}".format(a,b))
        else:
            loss = F.cross_entropy(outputp1, y, reduction='mean')

        _, preds = outputp1.topk(self.topk, 1, True, True)

        loss.backward()
        optimizer.step()
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)

    def _train(
        self, train_loader, memory_loader, optimizer, criterion,cur_iter
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        memory_loader=None
        self.model.train()
        for data in train_loader:
            x1 = data["image1"]
            x2 = data["image2"]
            y = data["label"]
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            y = y.to(self.device)
            
            l, c, d = self.update_model(x1,x2, y, criterion, optimizer,cur_iter,mode=1)
        # for data in memory_loader:
        #     x1 = data["image1"]
        #     x2 = data["image2"]
        #     y = data["label"]
        #     x1 = x1.to(self.device)
        #     x2 = x2.to(self.device)
        #     y = y.to(self.device)
            
        #     l, c, d = self.update_model(x1,x2, y, criterion, optimizer,cur_iter,mode=2)
        # # for batch_idx,(data, target) in enumerate(train_loader):
                # data, target = data.to(self.device), target.to(self.device)
                # l, c, d = self.update_model(data,target , criterion, optimizer)
                # total_loss += l
                # correct += c
                # num_data += d
        #         ori = data
        #         optimizer.zero_grad()
        #         output = model(data)
        #         # output2=model2(data)
                
        #         outputs_S = F.log_softmax(output,dim=1)
        #         outputs_T = F.softmax(output2,dim=1)

        #         kl=nn.KLDivLoss(reduction='batchmean')
        #         if t>5:
        #             loss = 0.75*F.cross_entropy(
        #                 output, target, reduction='mean')+0.25*kl(outputs_S,outputs_T)
        #         else:
        #             loss = F.cross_entropy(
        #                 output, target, reduction='mean')
        #         loss.backward()
        #         optimizer.step()
        #         if batch_idx % args.log_interval == 0 or batch_idx == 49999:
        #             print('TASK{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #                 t,epoch, batch_idx *
        #                 len(data), 50000,
        #                 100. * batch_idx / len(trdata_list), loss.item()))
        # if memory_loader is not None:

        #     for data in memory_loader:
        #         x = data["image"]
        #         y = data["label"]
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         # print(x.size())
        #         l, c, d = self.update_model(x, y, criterion, optimizer)
        #         total_loss += l
        #         correct += c
        #         num_data += d
        # memory_loader=None
        
        
        
        # if train_loader is not None:
        #     n_batches = len(train_loader)
        # else:
        #     n_batches = len(memory_loader)

        # return total_loss , correct / num_data

    def allocate_batch_size(self, n_old_class, n_new_class):
        new_batch_size = int(
            self.batch_size * n_new_class / (n_old_class + n_new_class)
        )
        old_batch_size = self.batch_size - new_batch_size
        return new_batch_size, old_batch_size
