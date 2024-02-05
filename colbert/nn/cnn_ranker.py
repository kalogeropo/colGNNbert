import torch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import Softmax
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import NLLLoss
from torch.nn import ReLU
from torch.optim import Adam

from colbert.nn.nn_ranker import NNRanker, load_nn_ranker


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


class CNNRanker(NNRanker):
    def __init__(self, args):
        super(CNNRanker, self).__init__(args)
        self.model = None
        self.loss = None
        self.optimizer = None

    def fit(self, train_data_loader, val_data_loader, h, w):
        self.model = CNNModel(h, w)
        self.model.to('cpu')
        # initialize our optimizer and loss function
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
        self.loss = NLLLoss()

        # loop over our epochs
        for e in range(0, self.args.epochs):
            # set the model in training mode
            self.model.train()

            # initialize the total training and validation loss
            self.total_train_loss = 0
            self.total_val_loss = 0

            # initialize the number of correct predictions in the training and validation step
            self.train_correct = 0
            self.val_correct = 0

            self._fit(train_data_loader)
            self._validation(val_data_loader)
            self._training_info(e + 1)

    def _fit(self, train_data_loader):
        # loop over the training set
        for (x, y) in train_data_loader:
            # send the input to the device
            (x, y) = (x.to('cpu'), y.to('cpu'))

            # perform a forward pass and calculate the training loss
            pred = self.model(x)
            loss = self.loss(pred, y)

            # zero out the gradients, perform the backpropagation step and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # add the loss to the total training loss so far and calculate the number of correct predictions
            self.total_train_loss += loss
            self.train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    def _validation(self, val_data_loader):
        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            self.model.eval()

            # loop over the validation set
            for (x, y) in val_data_loader:
                # send the input to the device
                (x, y) = (x.to('cpu'), y.to('cpu'))

                # make the predictions and calculate the validation loss
                pred = self.model(x)
                self.total_val_loss += self.loss(pred, y)

                # calculate the number of correct predictions
                self.val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    def predict(self, similarities, path=None):
        if self.model is None:
            self.model = load_nn_ranker(path)
            self.model.to('cpu')
        # switch off autograd
        with torch.no_grad():
            x = similarities.to('cpu')
            pred = self.model(x)
            return pred[:, 1]
