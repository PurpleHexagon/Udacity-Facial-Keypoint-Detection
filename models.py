## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        # W = 222, F = 5, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (224 - 5 + 2) / 1 + 1 = 222
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 1) # (32, 222, 222)
        self.conv1_batch_norm = nn.BatchNorm2d(32)

        # W = 111, F = 3, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (111 - 3 + 2) / 1 + 1 = 111
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)  # (64, 111, 111) after pool (64, 55, 55)
        self.conv2_batch_norm = nn.BatchNorm2d(64)

        # W = 55, F = 3, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (55 - 3 + 2) / 1 + 1 = 55
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)  # (128, 55, 55) after pool (128, 27, 27)

        # W = 55, F = 3, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (55 - 3 + 2) / 1 + 1 = 27
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)  # (256, 27, 27) after pool (256, 13, 13)

        # W = 55, F = 3, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (55 - 3 + 2) / 1 + 1 = 27
        self.conv5 = nn.Conv2d(256, 512, 1, 1, 1)  # (512, 13, 13) after pool (512, 6, 6)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.1)
        self.drop3 = nn.Dropout(p = 0.1)
        self.drop4 = nn.Dropout(p = 0.1)
        self.drop5 = nn.Dropout(p = 0.3)
        self.drop6 = nn.Dropout(p = 0.4)

        self.fc1 = nn.Linear(512 * 7 * 7, 1024)

        # Final output
        self.fc2 = nn.Linear(1024, 136)


    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers

        # W = 224, F = 5, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (224 - 5 + 2) / 1 + 1 = 222
        x = self.pool1(F.selu(self.conv1_batch_norm(self.conv1(x)))) # (32, 222, 222) after pool (32, 111, 111)
        x = self.drop1(x)

        # W = 111, F = 5, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (111 - 5 + 2) / 1 + 1 = 109
        x = self.pool2(F.selu(self.conv2_batch_norm(self.conv2(x)))) # (64, 109, 109) after pool (64, 54, 54)
        x = self.drop2(x)

        # W = 54, F = 3, S = 1, P = 1
        # output size = (W - F + 2P)/S +1 = (54 - 3 + 2) / 1 + 1 = 54
        x = self.pool3(F.selu(self.conv3(x))) # (128, 54, 54) after pool (128, 27, 27)
        x = self.drop3(x)


        x = self.pool4(F.selu(self.conv4(x)))
        x = self.drop4(x)

        x = self.pool5(F.selu(self.conv5(x)))
        x = self.drop5(x)

        # prep for linear layer by flattening
        x = x.view(x.size(0), -1)

        # linear layers and dropout
        x = F.selu(self.fc1(x))
        x = self.drop6(x)

        x = self.fc2(x)

        # final output
        return x