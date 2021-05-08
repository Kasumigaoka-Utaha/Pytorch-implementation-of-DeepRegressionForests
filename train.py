import sys, os, re, urllib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from DeepRegressionForestsNetwork import DeepRegressionForestsNetwork
from torch.utils.data import DataLoader
from defaults import _C as cfg
from Input import FaceDataset



class Average_data(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, net, criterion, optimizer, epoch, device):
    # training function
    print('\nTraining Epoch: %d' % epoch)
    net.train()
    loss_data = Average_data()
    accuracy_data = Average_data()
    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)
            # compute output
            outputs = net(x)
            # calc loss
            loss = criterion(outputs, y)
            cur_loss = loss.item()
            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()
            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_data.update(cur_loss, sample_num)
            accuracy_data.update(correct_num, sample_num)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_data.avg),
                              acc=accuracy_data.avg, correct=correct_num, sample_num=sample_num)    
    return loss_data.avg, accuracy_data.avg

def test(test_loader, net, criterion, optimizer, epoch, device):
    # test function for every epoch
    net.eval()
    loss_data = Average_data()
    accuracy_data = Average_data()
    preds = []
    gt = []
    with torch.no_grad():
        with tqdm(test_loader) as _tqdm:
            for x, y in _tqdm:
                x = x.to(device)
                y = y.to(device)
                # compute output
                outputs = net(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())
                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()
                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()
                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_data.update(cur_loss, sample_num)
                    accuracy_data.update(correct_num, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_data.avg),
                                      acc=accuracy_data.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()
    return loss_data.avg, accuracy_data.avg, mae

def main():
  tmp_dir = 'tmp'
  tmp_dir = join(dirname(__file__), tmp_dir)
  if not isdir(tmp_dir):
    os.makedirs(tmp_dir)

  parser = argparse.ArgumentParser(description='DRF')
  parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
  #parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
  #parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
  #parser.add_argument('--nout', type=int, required=False, default=128)
  #parser.add_argument('--save', type=str, required=False, default='model')
  args=parser.parse_args()



  # some useful options ##
  ntree = args.tree      #
  treeDepth = args.depth # 
  #test_batch_size = 15   #
  ########################

  #system initialize

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Current used device is ',device,'.')

  #prepare the data
  print('Preparing the data...')
  print('Preparing the training data...')
  train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
  train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)    
  print('Training data finished...')
  print('Preparing the testing data...')
  test_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
  test_loader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)
  print('Testing data finished...')
  #initialize the network
  if args.nout > 0:
    assert(args.nout >= int(pow(2, treeDepth - 1) - 1))
    nout = args.nout
  else:
    if ntree == 1:
      nout = int(pow(2, treeDepth - 1) - 1)
    else:
      nout = int((pow(2, treeDepth - 1) - 1) * ntree * 2 / 3)

  nin=3
  print('Build the DRF Network...')
  net = DeepRegressionForestsNetwork(nin,nout)
  net = net.to(device)
  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
  print('Successfully build the DRF Network...')
  start_epoch=0
  criterion = nn.CrossEntropyLoss().to(device)
  optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  best_acc = 0
  best_mae = 0
  for epoch in range(start_epoch,start_epoch+200):
    #train
    train_loss,train_acc = train(train_loader,net,criterion,optimizer,epoch,device)
    print('In current epoch, train_loss_avg is ',train_loss,' train_acc_avg is ',train_acc)
    #test
    test_loss,test_acc,mae = test(test_loader,net,criterion,optimizer,epoch,device)
    print('In current epoch, test_loss_avg is ',test_loss,' test_acc_avg is ',test_acc,' test_mae is ',mae)
    #checkpoint
    if best_acc < test_acc:
      best_acc = test_acc
    if best_mae < mae:
      best_mae = mae
    scheduler.step()

  print('Training finished...')
  print('In the total training process, the best accuracy is ',best_acc,' the best mae is ',best_mae)

if __name__ == '__main__':
    main()





