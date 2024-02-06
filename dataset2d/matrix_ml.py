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
metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}


def training_info(epoch=0):
    avg_train_loss = total_train_loss / train_steps
    # avg_val_loss = self.total_val_loss / self.val_steps
    # calculate the training and validation accuracy
    train_accuracy = train_correct / train_len
    # val_accuracy = self.val_correct / self.val_len

    # update our training history
    metrics['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    metrics['train_acc'].append(train_accuracy)
    # metrics['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    # metrics['val_acc'].append(val_accuracy)
    if epoch > 0:
        print('[INFO] EPOCH: {}/{}'.format(epoch, NUM_EPOCHS))
        print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_accuracy))
        # print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(avg_val_loss, val_accuracy))


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
    epoch_train_acc = []
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
            train_accuracy = train_correct / train_len
            # val_accuracy = val_correct / val_len
        training_info(e + 1)
