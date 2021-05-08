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
from solver import solver
from torch.utils.data import DataLoader
from defaults import _C as cfg
from Input import FaceDataset




def main():
  tmp_dir = 'tmp'
  tmp_dir = join(dirname(__file__), tmp_dir)
  if not isdir(tmp_dir):
    os.makedirs(tmp_dir)

  parser = argparse.ArgumentParser(description='DRF')
  parser.add_argument('--tree', type=int, required=False, default=5)
  parser.add_argument('--depth', type=int, required=False, default=6)
  parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
  parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
  parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
  parser.add_argument('--nout', type=int, required=False, default=128)
  parser.add_argument('--save', type=str, required=False, default='model')
  args=parser.parse_args()

  #testdir = './data/morph/'  #directory for test dataset
  #traindir = './data/morph/' #directory for train dataset


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

  print('Build the DRF Network...')
  start_epoch = 0
  Solver = solver(device,nout=nout,num_tree=ntree,depth=treeDepth)
  print('Successfully build the DRF Network...')
  
  for epoch in range(start_epoch,start_epoch+200):
    #train
    train_loss,train_acc = Solver.train(train_loader,epoch,device)
    print('In current epoch, train_loss_avg is ',train_loss,' train_acc_avg is ',train_acc)
    #test
    test_loss,test_acc,mae = Solver.test(test_loader,epoch,device)
    print('In current epoch, test_loss_avg is ',test_loss,' test_acc_avg is ',test_acc,' test_mae is ',mae)
    #checkpoint
    if best_acc < test_acc:
      best_acc = test_acc
    if best_mae < mae:
      best_mae = mae

  print('Training finished...')
  print('In the total training process, the best accuracy is ',best_acc,' the best mae is ',best_mae)

if __name__ == '__main__':
    main()





