import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import pickle

class getFeature(nn.Module):
    # Network to get image features, 5 CNN layer and 3 fc layer 
    def __init__(self,input_channels,nout):
        super(getFeature, self).__init__()
        #first CNN layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        #Second CNN layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        #Third CNN layer
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu6 = nn.ReLU(inplace=False)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu7 = nn.ReLU(inplace=False)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        #Fourth CNN layer
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu8 = nn.ReLU(inplace=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu9 = nn.ReLU(inplace=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu10 = nn.ReLU(inplace=False)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        #Fifth CNN layer
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu11 = nn.ReLU(inplace=False)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu12 = nn.ReLU(inplace=False)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu13 = nn.ReLU(inplace=False)
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        #Sixth FC layer
        self.fc6 = nn.Linear(512*7*7,4096)
        self.relu14 = nn.ReLU(inplace=False)
        self.drop6 = nn.Dropout(p=0.5,inplace=True)
        #Seventh FC layer
        self.fc7 = nn.Linear(4096,4096)
        self.relu15 = nn.ReLU(inplace=False)
        self.drop7 = nn.Dropout(p=0.5,inplace=True)
        #Eightth FC layer
        self.fc8 = nn.Linear(4096,nout)
    
    def forward(self,x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.pool1(out)
        out = self.relu3(self.conv3(out))
        out = self.relu4(self.conv4(out))
        out = self.pool2(out)
        out = self.relu5(self.conv5(out))
        out = self.relu6(self.conv6(out))
        out = self.relu7(self.conv7(out))
        out = self.pool3(out)
        out = self.relu8(self.conv8(out))
        out = self.relu9(self.conv9(out))
        out = self.relu10(self.conv10(out))
        out = self.pool4(out)
        out = self.relu11(self.conv11(out))
        out = self.relu12(self.conv12(out))
        out = self.relu13(self.conv13(out))
        out = self.pool5(out)
        out = out.view(out.size(0), -1)
        out = self.fc6(out)
        out = self.relu14(out)
        out = self.drop6(out)
        out = self.fc7(out)
        out = self.relu15(out)
        out = self.drop7(out)
        out = self.fc8(out) #[4,128]
        return out

def gaussian_func(y, mu, sigma):
    # gaussian distribution implementation, to avoid 0 as divisor, we add 1e-9
    samples = y.shape[0]
    num_tree, leaf_num, _, _ = mu.shape
    y = np.reshape(y, [samples, 1, 1])
    y = np.repeat(y, num_tree, 1)
    y = np.repeat(y, leaf_num, 2)   
    mu = np.reshape(mu, [1, num_tree, leaf_num]) 
    mu = mu.repeat(samples, 0) # mean matrix
    sigma = np.reshape(sigma, [1, num_tree, leaf_num])
    sigma = sigma.repeat(samples, 0) # covariance matrix
    res = 1.0 / np.sqrt(2 * 3.14 * (sigma + 1e-9)) * \
         (np.exp(- (y - mu) ** 2 / (2 * (sigma + 1e-9))) + 1e-9)
    return res

class pi_func():
    # leaf node distribution
    def __init__(self,num_tree,depth,task_num=1,iter_num=20):
        super(pi_func, self).__init__()
        self.num_node = pow(2,depth-1)
        self.num_tree = num_tree
        self.mean = np.random.rand(num_tree, self.num_node, task_num, 1).astype(np.float32) # mean matrix
        self.sigma = np.random.rand(num_tree, self.num_node, task_num, task_num).astype(np.float32) # covariance matrix
        self.iter_num = iter_num # num iterations

    def init_kmeans(self,mean,sigma): 
    # functions for init mean and sigma
        for i in range(self.num_node):
            self.mean[:, i, :, :] = mean[i]
            self.sigma[:, i, :, :] = sigma[i]
    
    def get_mean(self):
        return torch.tensor(self.mean).squeeze().cuda()
    
    def getGaussionVal(self,y):
        return gaussian_func(y,self.mean,self.sigma)
    
    def update_leaf(self,x,y):        
        for i in range(self.iter_num):
            gauss = self.getGaussionVal(y)
            leaf_prob = x*(gauss+ 1e-9)
            leaf_prob_sum = np.sum(leaf_prob,axis=2, keepdims=True)
            zeta = leaf_prob/(leaf_prob_sum+1e-9)
            y_temp = np.expand_dims(y, 2)
            y_temp = np.repeat(y_temp, self.num_tree, axis=1)
            y_temp = np.repeat(y_temp, self.num_node, axis=2)
            zeta_y = np.sum(zeta*y_temp,axis=0)
            zeta_sum = np.sum(zeta,axis=0)
            mean = zeta_y/(zeta_sum+1e-9)
            new_mean = y_temp-np.expand_dims(mean,axis=0)
            self.mean[:,:,0,0] = mean
            zeta_for_sigma = zeta * new_mean * new_mean
            zeta_for_sigma = np.sum(zeta_for_sigma, 0)
            sigma = zeta_for_sigma / (zeta_sum + 1e-9)
            self.sigma[:,:,0,0] = sigma


class Tree(nn.Module):
    # implement the basic tree
    def __init__(self,depth,input_feature):
        super(Tree, self).__init__()
        self.depth = depth
        self.leaf_num = pow(2,depth-1)
        features = np.eye(input_feature)
        # randomly choose a feature at one dimension within the range of nout
        choose_node = np.random.choice(np.arange(input_feature), self.leaf_num-1, replace=False)
        self.feature = features[choose_node].T
        self.feature = Parameter(torch.from_numpy(self.feature).type(torch.FloatTensor),requires_grad=False) #[128,31]

    def forward(self,x):
        # to compute the probability of sample x falling into leaf node l, use formula(2)
        if x.is_cuda and not self.feature.is_cuda:
            self.feature = self.feature.cuda()
        temp = torch.mm(x,self.feature) #[4,31]
        deter = torch.sigmoid(temp) #[4,31]
        deter = torch.unsqueeze(deter,2) #[4,31,1]
        deter_inv = 1-deter
        deter = torch.cat((deter,deter_inv),dim=2) #[4,31,2]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size,1,1).fill_(1.)) #[4,1,1]
        begin_idx = 0
        end_idx = 1
        # compute the route probability for the tree
        for n_layer in range(0, self.depth - 1):
            _mu = _mu.view(batch_size,-1,1).repeat(1,1,2) #[4,1,2] --> [4,2,2] --> [4,4,2] --> ... -->[4,32,2]
            _deter = deter[:, begin_idx:end_idx, :]  #[4,1,2] --> [4,2,2] --> [4,4,2] --> ... -->[4,32,2]
            _mu = _mu*_deter #[4,1,2] --> [4,2,2] --> [4,4,2] --> ... -->[4,32,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
        mu = _mu.view(batch_size,self.leaf_num) #[4,32]
        return mu

       
class Forest(nn.Module):
    # implement the forest code
    def __init__(self,num_tree,depth,input_feature):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        #append all the tree to the trees list
        for i in range(num_tree):
            self.trees.append(Tree(depth,input_feature))
        self.num_tree = num_tree
        self.depth = depth
        self.pi = pi_func(num_tree,depth)

    def forward(self,x):
        # compute the conditional probability by formula(3),formula(4)
        probs = []
        for tree in self.trees:
            tree_out = tree(x)
            probs.append(tree_out.unsqueeze(2))
        pi = self.pi.get_mean() #[5,32]
        probs = torch.cat(probs,dim=2) #[4,32,5]
        prob = probs * pi.transpose(0, 1).unsqueeze(0)  # [4,32,5] calculate the conditional probability
        prob = torch.sum(prob, dim=1) #[4,5] sum the results and got a final pred result
        return prob, probs
        

class DeepRegressionForestNetwork(nn.Module):
    # the total network
    def __init__(self,feature_net,forest):
        super(DeepRegressionForestNetwork, self).__init__()
        self.feature_net = feature_net
        self.forest = forest
    
    def forward(self,x):
        out = self.feature_net(x)
        out = out.view(x.size()[0],-1)
        out = self.forest(out)
        return out
