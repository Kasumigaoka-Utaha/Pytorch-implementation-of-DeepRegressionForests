# Pytorch-implementation-of-DeepRegressionForests
This is a pytorch version implementation of the caffe-DeepRegressionForests for age estimation task.

# Original Paper and Code
Original Paper: https://arxiv.org/abs/1712.07195

Original Code: https://github.com/shenwei1231/caffe-DeepRegressionForests

The original version utilize the caffe to implement the whole project, which needs a lot of effort to implement the conv functions. I decided to reimplement the algorithm with pytorch version code

# Create environment
You can follow the followwing code to create environment:

    conda create -n drf python==3.7.9
    conda activate drf
    pip install -r requirements.txt

# Dataset 
In this project, I decided to use FGNET for a light training process. You can preprocess the dataset with 
  python readData.py
As for splitting the datasat to the training set and test set, I choose to directly split the images directly. The training set contains 839 images and the test set contains 163 images.

# Training
You can train the model by 

    python train.py

# Details
Further information will be updated soon.
