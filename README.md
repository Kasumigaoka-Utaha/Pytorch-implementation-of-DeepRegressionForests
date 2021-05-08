# Pytorch-implementation-of-DeepRegressionForests
This is a pytorch version implementation of the caffe-DeepRegressionForests for age estimation task.

# Original Paper and Code
Original Paper: https://arxiv.org/abs/1712.07195

Original Code: https://github.com/shenwei1231/caffe-DeepRegressionForests

The original version utilize the caffe to implement the whole project, which needs a lot of effort to implement the conv functions. I decided to reimplement the algorithm with pytorch version code

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

# Details
Further information will be updated soon.
