import DeepRegressionForestsNetwork as NDF
from DeepRegressionForestsNetwork import getFeature
from DeepRegressionForestsNetwork import DeepRegressionForestNetwork
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
import sys, os, re, urllib
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from Input import FaceDataset
from defaults import _C as cfg

class Average_data(object):
    # class to calculate average loss
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # update the results
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class solver():
    # training device
    def __init__(self,device,nout=128,num_tree=5,depth=6):
        super(solver, self).__init__()
        # model construction
        feature_net = getFeature(3,nout)
        forest = NDF.Forest(num_tree=num_tree, depth=depth, input_feature=nout)
        model = DeepRegressionForestNetwork(feature_net,forest)
        self.model = model
        self.num_node = pow(2,depth-1)
        model = model.to(device)
        # set the optimizer and the scheduler
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        if cfg.TRAIN.OPT == "sgd":
            optimizer = torch.optim.SGD(feature_net.parameters(), lr=cfg.TRAIN.LR,momentum=cfg.TRAIN.MOMENTUM,weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.Adam(feature_net.parameters(), lr=cfg.TRAIN.LR)
        start_epoch = 0
        self.optim = optimizer
        self.scheduler = StepLR(self.optim, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,last_epoch=start_epoch - 1)
    
    def forward(self,x):
        pred,preds = self.model(x)
        return pred,preds

    def get_loss(self,x,y):
        pred,preds = self.forward(x)
        loss = torch.sum(0.5 * (y.view(-1, 1)-pred) ** 2)/x.shape[0]
        return loss,pred,preds
    
    def backward_theta(self,x,y):
        self.optim.zero_grad()
        loss,pred,pred4Pi = self.get_loss(x, y)
        loss.backward()
        self.optim.step()
        return loss.item(),pred,pred4Pi

    def backward_pifunc(self,x,y):
        self.model.forest.pi.update_leaf(x,y)

    def update_lr(self):
        self.scheduler.step()

    def train(self,train_loader,epoch,device):
        print('\nTraining Epoch: %d' % epoch)
        self.model.train()
        loss_data = Average_data()
        accuracy_data = Average_data()
        update_leaf_pred = []
        update_leaf_label = []
        with tqdm(train_loader) as _tqdm:
            for x, y in _tqdm:
                x = x.to(device)
                y = y.to(device)
                cur_loss,outputs,pred4Pi = self.backward_theta(x,y)
                update_leaf_pred.append(pred4Pi)
                update_leaf_label.append(y.view(-1, 1))
                update_pred = torch.cat(update_leaf_pred, dim=0).transpose(1, 2).detach().cpu().numpy()
                update_label = torch.cat(update_leaf_label, dim=0).detach().cpu().numpy()
                self.backward_pifunc(update_pred,update_label)
                _, predicted = outputs.max(1)
                correct_num = predicted.eq(y).sum().item()
                # measure accuracy and record loss
                sample_num = x.size()[0]
                loss_data.update(cur_loss, sample_num)
                accuracy_data.update(correct_num, sample_num)
                _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_data.avg),
                                acc=accuracy_data.avg, correct=correct_num, sample_num=sample_num) 
        self.scheduler.step()   
        return loss_data.avg, accuracy_data.avg

    def test(self,test_loader,epoch,device):
        mae = 0.0
        total_num = 0
        kl = 0
        with tqdm(test_loader) as _tqdm:
            for x, y in _tqdm: 
                total_num += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                self.model.eval()
                pred, _ = self.forward(x)
                pred = torch.mean(pred, dim=1)
                mae += torch.sum(torch.abs(y - pred)).item()
                kl += torch.sum(torch.abs(y-pred) < 5).item()
            res_mae = mae /total_num
            res_kl = kl/ total_num
        return res_kl, res_mae

  


    



