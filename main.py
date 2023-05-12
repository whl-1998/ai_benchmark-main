# @Time    : 2023/5/11 11:58
# @Author  : emo
import os

import torch.utils.data as data
import tqdm
import torch
from torch import nn
from args import args

from data.transform import get_transforms
from model.build_net import make_model
from data.utils import AverageMeter, accuracy, get_optimizer

from data.dataset_init import SkinDisease
from data.data_gen import Dataset


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    losses = AverageMeter()
    train_acc = AverageMeter()

    for (inputs, targets) in train_loader:
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # 梯度参数设为0
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        acc = accuracy(outputs.data, targets.data)
        # inputs.size(0)=32
        losses.update(loss.item(), inputs.size(0))
        train_acc.update(acc.item(), inputs.size(0))

    return losses.avg, train_acc.avg


def val(val_loader, model, criterion, epoch, use_cuda):
    losses = AverageMeter()
    val_acc = AverageMeter()

    model.eval()  # 将模型设置为验证模式
    for _, (inputs, targets) in enumerate(val_loader):
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs.data, targets.data)

        # measure accuracy and record loss
        losses.update(loss.item(), inputs.size(0))
        val_acc.update(acc1.item(), inputs.size(0))
    return losses.avg, val_acc.avg


if __name__ == '__main__':
    # 初始化使用的数据集
    skin_disease = SkinDisease()

    transformations = get_transforms(input_size=args.image_size)

    train_set = Dataset(description_path=skin_disease.train_description,
                        dataset_path=skin_disease.dataset_path,
                        transform=transformations['train'])
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    val_set = Dataset(description_path=skin_disease.val_description,
                      dataset_path=skin_disease.dataset_path,
                      transform=transformations['val'])
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # model
    model = make_model()
    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)

    # train
    for epoch in range(args.epochs):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, False)
        val_loss, val_acc = val(val_loader, model, criterion, epoch, False)
        scheduler.step(val_loss)
        print(f'train_loss:{train_loss}\t val_loss:{val_loss}\t train_acc:{train_acc} \t val_acc:{val_acc}')
