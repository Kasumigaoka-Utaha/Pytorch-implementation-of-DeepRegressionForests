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

# Experiment Results
Test Dataset: FGNET

Initial random setting performance:

    Weight_decay = 5e-4
    Momentum = 0.9
    Learning rate = 0.1 with no step variance
    Best_acc = 0.0368
    Best_mae = 9.831

Adam training performance (Experiment will be implemented further):

    TRAIN.LR_DECAY_STEP = 20
    TRAIN.LR_DECAY_RATE = 0.2
    TRAIN.MOMENTUM = 0.9
    TRAIN.WEIGHT_DECAY = 0.0
    TRAIN.LR = 0.001
    Best_acc = 0.0675
    Best_mae = 19.162

SGD training performance:
Experiment 1: modify initial learning rate:

    TRAIN.LR_DECAY_STEP = 20
    TRAIN.LR_DECAY_RATE = 0.2
    TRAIN.MOMENTUM = 0.9
    TRAIN.WEIGHT_DECAY = 0.0
    
|Initial learning rate|Best_acc|Best_mae|
|0.01|0.0368|13.258|
|0.1|0.0675|9.49|
|0.08|0.0368|9.409|
|0.07|0.0675|9.445|
|0.085|0.0675|9.456|

# Details
Further information will be updated soon.
