from os import listdir, getcwd
from os.path import isdir, join
import torchvision.transforms as transforms

import pandas as pd

import torch
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Softmax
from torch.utils.data import Dataset, DataLoader
from torch import nn
from PIL import Image


def split_dataset(dataset, ratio):
    dataset_len = len(dataset)
    train_len = int(dataset_len * ratio)
    val_len = dataset_len - train_len
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_len, val_len],
                                                              generator=torch.Generator().manual_seed(1234))
    print(F"Train set size: {len(train_set)}\nTest set size: {len(validation_set)}")
    return train_set, validation_set, train_len, val_len


def read_set():
    # Get image files paths
    dirs = [name for name in listdir(".") if isdir(name)]
    cwd = getcwd()

    image_dir = join(cwd, dirs[1])

    images = [join(image_dir, f) for f in listdir(image_dir)]
    images = sorted(images, key=len)

    # Load images into lists

    img_vecs = []
    for path in images:
        img = Image.open(path)
        img_vecs.append(img)
    # print(len(img_vecs))
    # img_vecs[0].show()
    with open('labels.csv') as f:
        reader = f.readlines()
        data = [int(i) for i in reader]
    # print(len(data))
    # Create dataframe of paths images and labels
    dict_to_df = {"img_path": images, "label": data, "img_vec": img_vecs}
    df = pd.DataFrame(dict_to_df)
    # print(df.head())
    assert len(img_vecs) == len(data) == len(images), f"image list and labels have different sizes"
    return img_vecs, data, len(img_vecs)


class CustomDataset(Dataset):
    def __init__(self, transform=transforms.ToTensor()):
        self.transform = transform
        self.images, self.labels, self.length = read_set()
        print(f"Loaded {len(self.images)} images from ColBERT Dataset")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        return image, self.labels[index]

    def validate_images(self):
        assert len(self.images) == len(self.labels), f"images and labels have different sizes"
        for i, _ in enumerate(self.images):
            assert self.images[0].size == self.images[i].size, f"images have different sizes"
        w, h = self.images[0].size
        return w, h


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
