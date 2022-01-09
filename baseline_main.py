#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import SMD_opt
from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,modelC,Net,TeacherNet,StudentNet


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'  # 加载参数，判断是否支持gpu计算

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model=='CNN':
        if args.dataset=="cifar":
            global_model=Net(args=args)
    elif args.model=='teacher':
        if args.dataset=="mnist":
            global_model=TeacherNet()
    elif args.model=='student':
        if args.dataset=="mnist":
            global_model=StudentNet()
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        #optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
         #                           momentum=0.5)#84.14
        #optimizer=SMD_opt.SMD_compress(global_model.parameters(),lr=args.lr,momentum=0.5)#76.43
        #optimizer = SMD_opt.SGD(global_model.parameters(), lr=args.lr, momentum=0.5)#83.96
        #optimizer = SMD_opt.SMD_qnorm(global_model.parameters(), lr=args.lr, momentum=0.5)#93.43
        optimizer = SMD_opt.LinearCoupl(global_model.parameters(), lr=args.lr, momentum=0.5)#94.52
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_loss = []
    test_accuracy = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)

    # Plot loss
    # plt.figure()
    # plt.plot(range(len(epoch_loss)), epoch_loss)
    # plt.xlabel('epochs')
    # plt.ylabel('Train loss')
    # plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
    #                                              args.epochs))

    # testing
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100 * test_acc))
    plt.figure()
    plt.plot(range(len(test_accuracy)), test_accuracy)
    plt.xlabel('epochs')
    plt.ylabel('baseline Test Accuracy')
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))
    #torch.save(global_model,'mlp.h5')
    with open("Fe_student.txt", 'a') as f:
        for temp_zero in test_accuracy:
            f.write(str(temp_zero) + "\n")
        f.close()
    #torch.save(global_model,'teachernet.h5')
    #torch.save(global_model,'teachernet.pt')
    #torch.save(global_model.state_dict(),'teachernet.pkl')

