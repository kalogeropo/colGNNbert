import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from image_cnn import CustomDataset, CNNModel

NUM_EPOCHS = 10
TRAIN_SPLIT = 0.9
BSIZE = 32
SHUFFLE = True
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def split_dataset(dataset, ratio):
    dataset_len = len(dataset)
    train_len = int(dataset_len * ratio)
    val_len = dataset_len - train_len
    train_set, validation_set = torch.utils.data.random_split(dataset, [train_len, val_len],
                                                              generator=torch.Generator().manual_seed(1234))
    print(F"Train set size: {len(train_set)}\nTest set size: {len(validation_set)}")
    return train_set, validation_set, train_len, val_len


if __name__ == '__main__':
    # Initialize Data set
    image_custom_set = CustomDataset()
    h, w = image_custom_set.calc_init_h_w()

    # set the model in training mode
    train_set, validation_set, train_len, val_len = split_dataset(image_custom_set, TRAIN_SPLIT)
    train_loader = DataLoader(dataset=train_set, shuffle=SHUFFLE, batch_size=BSIZE)
    validation_loader = DataLoader(dataset=validation_set, shuffle=SHUFFLE, batch_size=BSIZE)

    train_steps = train_len // BSIZE
    val_steps = val_len // BSIZE
    print(train_steps, val_steps)

    model = CNNModel(h, w)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    lossFn = NLLLoss()

    # loop over our epochs
    for e in range(0, NUM_EPOCHS):
        print(e)
        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        model.train()
        # initialize the number of correct predictions in the training and validation step
        train_correct = 0
        val_correct = 0
        for (x, y) in train_loader:
            # send the input to the device
            (x, y) = (x.to(device), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossFn(pred, y)

            # zero out the gradients, perform the backpropagation step and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add the loss to the total training loss so far and calculate the number of correct predictions
            total_train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # _fit(train_data_loader)
            # self._validation(val_data_loader)
            # self._training_info(e + 1)
