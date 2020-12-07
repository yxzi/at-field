import argparse
import os
import torch
from torch.optim import optimizer
from torch.utils.data import dataloader
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR

import art
from art.estimators.classification import PyTorchClassifier

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from utils import get_num_classes, get_architecture, ARCHITECTURES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Training')


parser.add_argument(
        "tag", 
        type=str,
        help="Experiment Tag for storing logs,models"
    )
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")

parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--k_value', default=0.2, type=float,
                    help='weight of targeted noise (default: 0.2)')
parser.add_argument('--eps_step', default=0.1, type=float,
                    help='scale of targeted noise (default: 0.1)')
args = parser.parse_args()

min_pixel_value = 0.0   # min value
max_pixel_value = 1.0   # max value



def main():
    
    expdir = os.path.join("exp", "train_" + args.tag)
    model_dir = os.path.join(expdir,"models")
    args.model_dir = model_dir
    for x in ['exp', expdir, model_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)
      
    
    cifar_train = datasets.CIFAR10("./dataset_cache", train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    cifar_val = datasets.CIFAR10("./dataset_cache", train=False, download=True, transform=transforms.ToTensor())


    train_loader = DataLoader(cifar_train, shuffle=True, batch_size=args.batch, num_workers=args.workers)
    val_loader = DataLoader(cifar_val, shuffle=False, batch_size=args.batch,num_workers=args.workers)
    
    # model = get_architecture(args.arch)
    model = torchvision.models.resnet18(pretrained=False, progress=True, **{"num_classes": 10}).to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma, verbose=True)

    for i in range(args.epochs):
        train(train_loader, model, criterion, optimizer, scheduler, i)
        acc = validate(val_loader, model, criterion)

        torch.save(
            {
            'epoch': i,
            'arch': args.arch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
            }, os.path.join(args.model_dir,"ep{}_acc_{}.pth".format(i,acc)))


    
        
def train(dataloader, model,criterion, optimizer, scheduler, epoch):
    model.train()
    print('epoch ' + str(epoch))

    train_loss = 0
    train_acc = 0
    total = len(dataloader)
    start = time.time()
    toPilImage = transforms.ToPILImage()    # transform tensor into PIL image to save

    for batch_num, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)


        # gauss noise training
        gauss_noise = torch.randn_like(x, device=device) * args.noise_sd
        # x_noise = x + torch.randn_like(x, device=device) * args.noise_sd

        # targeted noise training
        tmp_criterion = nn.CrossEntropyLoss()
        tmp_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        classifier = PyTorchClassifier(
            model=model,
            clip_values=(min_pixel_value, max_pixel_value),
            loss=tmp_criterion,
            optimizer=tmp_optimizer,
            input_shape=(3, 32, 32),
            nb_classes=10,
        )
        # generate random targets
        targets = art.utils.random_targets(y.cpu().numpy(), get_num_classes())

        # calculate loss gradient
        grad = classifier.loss_gradient(x=x.cpu().numpy(), y=targets)
        scaled_grad = torch.Tensor(grad * args.eps_step).to(device)

        # print((scaled_grad.shape, gauss_noise.shape, targets.shape))

        # combine noise and targeted noise
        x_combine = x + (gauss_noise * (1.0 - args.k_value)) + (scaled_grad * args.k_value)
        

        output = model(x_combine)       
        loss = criterion(output, y)
        acc = accuracy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()       
        train_acc += acc

    end = time.time()
    print('trainning time:',end - start,'sec, loss: ', train_loss/total, 'acc: ', train_acc/total)
    
    

    

def validate(dataloader, model,criterion):
    model.eval()
    val_loss = 0
    val_acc = 0

    total = len(dataloader)
    for batch_num, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        x = x + torch.randn_like(x, device=device) * args.noise_sd      
        output = model(x)      
        loss = criterion(output, y)
        acc = accuracy(output, y)
        val_loss += loss.item()     
        val_acc += acc
    print('validate: loss: ', val_loss/total, 'acc: ', val_acc/total )
    return val_acc/total




def accuracy(output, target):
    
    with torch.no_grad():
        pred_labels = output.argmax(axis=1).float()
        batch_size = len(target)
        
         
        correct = (pred_labels == target).sum().item()

        
        return correct/batch_size


if __name__ == "__main__":
    main()