# Pytorch-implementation-of-DeepRegressionForests
This is a pytorch version implementation of the caffe-DeepRegressionForests for age estimation task.

# Original Paper and Code
Original Paper: https://arxiv.org/abs/1712.07195

Original Code: https://github.com/shenwei1231/caffe-DeepRegressionForests

The original version utilize the caffe to implement the whole project, which needs a lot of effort to implement the conv functions. I decided to reimplement the algorithm with pytorch version code

    @inproceedings{shen2018DRFs,
        author = {Wei Shen and Yilu Guo and Yan Wang and Kai Zhao and Bo Wang and Alan Yuille},
        booktitle = {Proc. CVPR},
        title = {Deep Regression Forests for Age Estimation},
        year = {2018}
    }
# Initialize
Firstly, download the codes with following command:

    git clone --recursive https://github.com.cnpmjs.org/Kasumigaoka-Utaha/Pytorch-implementation-of-DeepRegressionForests
    cd Pytorch-implementation-of-DeepRegressionForests
# Create environment
You can follow the following code to create environment:

    conda create -n drf python==3.7.9
    conda activate drf
    pip install -r requirements.txt

# Dataset 
In this project, I decided to use FGNET for a light training process. You can download the dataset via

    wget http://yanweifu.github.io/FG_NET_data/FGNET.zip
    unzip FGNET.zip

You can preprocess the dataset with 

    python readData.py
    
The results were saved in the info.csv and the split results were saved in imgs_train.csv and imgs_test.csv.

As for splitting the datasat to the training set and test set, I choose to directly split the images directly. The training set contains 839 images and the test set contains 163 images.

# Training
You can train the model by 

    python train.py --data_dir ./FGNET/images

# Experiment Training Results

Test Dataset: FGNET

Train.opt = 'sgd'

Experiment 1: modify the initial learning rate

Basic hyperparameters:
    
    TRAIN.LR_DECAY_STEP = 20
    TRAIN.LR_DECAY_RATE = 0.2
    TRAIN.MOMENTUM = 0.9
    TRAIN.WEIGHT_DECAY = 0.0

|Initial learning rate|Best_cs|Best_mae|
|----|-----|-----|
|0.01|0.356|7.18|
|0.1|0.675|4.68|
|0.09|0.724|4.538|
|0.085|0.692|4.585|

Experiment 2: modify the momentum

Basic hyperparameters:
    
    TRAIN.LR_DECAY_STEP = 20
    TRAIN.LR_DECAY_RATE = 0.2
    TRAIN.LR = 0.09
    TRAIN.WEIGHT_DECAY = 0.0

|Momentum|Best_cs|Best_mae|
|----|-----|-----|
|0.9|0.724|4.538|
|0.85|0.711|4.551|
|0.8|0.724|4.538|


Current best hyperparameters:

    TRAIN.LR_DECAY_STEP = 20
    TRAIN.LR_DECAY_RATE = 0.2
    TRAIN.LR = 0.09
    TRAIN.WEIGHT_DECAY = 0.0
    TRAIN.MOMENTUM = 0.9

# Further Plans
These are some of my further plans:

    1. Modify the other hyperparameters, lr_decay_rate, lr_decay_step
    2. Try to use another dataset instead of FGNET (current dataset is too small)
    3. Try to modify the DeepRegressionForest structure 

# Details
Further information will be updated soon.
