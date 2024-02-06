import ast
import torchvision.transforms as transforms

import pandas as pd

import torch
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Softmax
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path='tensors.csv'):
        images_df = pd.read_csv(path)
        self.features = images_df['features'].map(ast.literal_eval).map(torch.Tensor)
        self.labels = images_df['label']
        assert len(self.features) == len(self.labels), f"image list and labels have different sizes"
        print(f"Loaded {len(self.features)} images from ColBERT Dataset")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def calc_init_w_h(self):
        return self.features[0].size


class CNNModel(Module):
    def __init__(self, h, w):
        # call the parent constructor
        super(CNNModel, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        h = CNNModel.calculate_output_shape(h, 5)
        w = CNNModel.calculate_output_shape(w, 5)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        h = CNNModel.calculate_output_shape(h, 2, stride=2)
        w = CNNModel.calculate_output_shape(w, 2, stride=2)

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        h = CNNModel.calculate_output_shape(h, 5)
        w = CNNModel.calculate_output_shape(w, 5)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        h = CNNModel.calculate_output_shape(h, 2, stride=2)
        w = CNNModel.calculate_output_shape(w, 2, stride=2)

        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=(h * w * 50), out_features=1000)
        self.relu3 = ReLU()

        # initialize our softmax classifier
        self.fc2 = Linear(in_features=1000, out_features=2)
        self.logSoftmax = Softmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output from the previous layer and pass it through our only set of FC => RELU layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output to our softmax classifier to get our output predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output

    @staticmethod
    def calculate_output_shape(x, kernel_size, stride=1, padding=0, dilation=1):
        x += 2 * padding - dilation * (kernel_size - 1) - 1
        x /= stride
        return int(x + 1)
