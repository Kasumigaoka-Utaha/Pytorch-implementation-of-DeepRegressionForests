import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




class DeepRegressionForestsNetwork(nn.Module):
    def __init__(self,input_channels,nout):
        super(DeepRegressionForestsNetwork, self).__init__()
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
        self.fc6 = nn.Linear(7,4096)
        self.relu14 = nn.ReLU(inplace=False)
        self.drop6 = nn.Dropout(p=0.5,inplace=True)
        #Seventh FC layer
        self.fc7 = nn.Linear(4096,1)
        self.relu15 = nn.ReLU(inplace=False)
        self.drop7 = nn.Dropout(p=0.5,inplace=True)
        #Eightth FC layer
        self.fc8 = nn.Linear(512*7,nout)
    
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
        out = self.fc6(out)
        out = self.relu14(out)
        out = self.drop6(out)
        out = self.fc7(out)
        out = self.relu15(out)
        out = self.drop7(out)
        out = torch.flatten(out,1)
        out = self.fc8(out)
        return out
