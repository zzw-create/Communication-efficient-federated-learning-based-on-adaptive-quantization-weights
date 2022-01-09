#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from Ternary import  quantize as Quantize
import torch.nn.functional as F
import numpy as np
import SMD_opt
temp_q1=[0 for _ in range(10)]
temp_loss=[0 for _ in range(10)]
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger,client_name):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        #self.criterion = nn.NLLLoss().to(self.device)
        self.criterion=nn.CrossEntropyLoss().to(self.device)
        self.client_name=client_name


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader
    def update_weights(self, model, global_round,client_name,args):
        # Set mode to train model
        global  q1,temp_loss,temp_q1
        model.train()
        epoch_loss = []

        def initial_scales():
            """
            :return: initialized quantization factor w_p
            """
            return 1.0
        def quantize(Q2, Q1, b):
            list_q2 = []
            list_q1 = []
            for list in Q2:
                list_q2.append(list)
            for list in Q1:
                list_q1.append(list)
            for i in range(len(list_q1)):
                a = Q2[list_q2[i]]
                #print("aaa",a)
                c = Q1[list_q1[i]]
                #print("cccc",c)
                r = torch.max(abs(a - c))
                # r = np.floor(r * 1e4)//1e4
                print("rrrrrrrrrr", r)
                delta = r / np.floor(2 ** (b - 1))
                # delta = np.floor(delta * 1e4)//1e4
                print("delta", delta)
                quant = c - r + 2 * delta * np.floor((a - c + r + delta) / (2 * delta))
                # quant = np.floor(quant * 1e4)// 1e4
                # print("quant----------",quant)
                Q2[list_q2[i]] = np.floor(quant * 1e4)/1e4
            return Q2

        def distillation(y, labels, teacher_scores, temp, alpha):
            return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
                    temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
             optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.5)
            #optimizer = SMD_opt.LinearCoupl(model.parameters(), lr=args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        teacher_model=torch.load("teachernet.h5")
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                #teacher_output = teacher_model(images)
                #teacher_output = teacher_output.detach()  # 切断老师网络的反向传播，感谢B站“淡淡的落”的提醒
                #loss = distillation(log_probs, labels, teacher_output, temp=5.0, alpha=1)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} |client name: {}| Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round,client_name, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        ave_loss=sum(epoch_loss)/len(epoch_loss)
        # if temp_q1[client_name]==0:
        #     temp_q1[client_name]= model.state_dict()
        #     ww = model.state_dict()
        #     print("第一轮未量化")
        # else:
        #     ww = quantize(model.state_dict(), temp_q1[client_name], 6)
        #     temp_q1[client_name] = ww
        # if ave_loss<temp_loss[client_name]:
        #     temp_loss[client_name]=ave_lossww
        # else:
        #     print("ave_loss",ave_loss)
        #     ww=0
        # print("wwwwwwww",ww)
        # print("quant_weight",temp_q1[client_name])
        #-----------------------------------------------------------
        # copy almost all full precision kernels of the model
        # all_fp_kernels = [
        #     kernel.clone().detach().requires_grad_(True)
        #     for kernel in optimizer.param_groups[1]['params']]
        #
        # # init quantification
        # initial_scaling_factors = []
        #
        # all_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
        #
        # ii = 0
        # for k, k_fp in zip(all_kernels, all_fp_kernels):
        #     w_p_initial = initial_scales()
        #
        #     initial_scaling_factors += [w_p_initial]
        #     # quantization
        #     k.data = quantize(k_fp.data, w_p_initial, args)
        #     ii += 1

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)



    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    acc=[]

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        # if batch_idx%10==0:
        #     acc.append(correct/total)

    accuracy = correct/total
    return accuracy, loss#,acc
