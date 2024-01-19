from torch import flatten
from torch import float as tfloat
from torch.nn import NLLLoss
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU
from torch.optim import Adam

from colbert.nn.nn_ranker import NNRanker


class CNNModel(Module):
    def __init__(self):
        # call the parent constructor
        super(CNNModel, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        # initialize our softmax classifier
        self.fc2 = Linear(in_features=500, out_features=2)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output


class CNNRanker(NNRanker):
    def __init__(self, args):
        super(CNNRanker, self).__init__(args)
        self.model = CNNModel()
        self.model.to('cpu')

    def fit(self, similarities, labels):
        # initialize our optimizer and loss function
        opt = Adam(self.model.parameters(), lr=self.args.lr)
        lossFn = NLLLoss()
        # initialize a dictionary to store training history
        H = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        # loop over our epochs
        for e in range(0, self.args.epochs):
            # set the model in training mode
            self.model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            # loop over the training set
            for i in range(labels):
                # send the input to the device
                (x, y) = (similarities[i].to('cpu'), labels[i].to('cpu'))
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = lossFn(pred, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(tfloat).sum().item()
