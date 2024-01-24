import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from colbert.modeling.inference import ModelInference
from colbert.utils.collection_parser import Collection


class TripletsDataset(Dataset):
    def __init__(self, collection, inference):
        self.collection = collection
        self.inference = inference
        self.similarities, self.labels = self._parse_triplets()

    def _parse_triplets(self):
        def create_similarities_matrix(q, d):
            q = self.inference.queryFromText(q)
            d = self.inference.docFromText(d)
            return self.inference.scores_matrix(q, d)

        triplets = self.collection.create_set()
        return triplets[['Query', 'Doc']].agg(create_similarities_matrix, axis='columns'), triplets['Pos/Neg']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.similarities.iloc[idx], self.labels[idx]


class NNRanker:
    def __init__(self, args):
        self.args = args
        self.collection = None
        self.inference = None
        self.model = None
        self.train_split = 0.75
        self.train_len = 0
        self.val_len = 0
        self.train_steps = 0
        self.val_steps = 0
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.train_correct = 0
        self.val_correct = 0
        self.metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def _training_info(self, epoch=0):
        avg_train_loss = self.total_train_loss / self.train_steps
        avg_val_loss = self.total_val_loss / self.val_steps

        # calculate the training and validation accuracy
        train_accuracy = self.train_correct / self.train_len
        val_accuracy = self.val_correct / self.val_len

        # update our training history
        self.metrics['train_loss'].append(avg_train_loss.cpu().detach().numpy())
        self.metrics['train_acc'].append(train_accuracy)
        self.metrics['val_loss'].append(avg_val_loss.cpu().detach().numpy())
        self.metrics['val_acc'].append(val_accuracy)

        if epoch > 0:
            print('[INFO] EPOCH: {}/{}'.format(epoch, self.args.epochs))
            print('Train loss: {:.6f}, Train accuracy: {:.4f}'.format(avg_train_loss, train_accuracy))
            print('Val loss: {:.6f}, Val accuracy: {:.4f}\n'.format(avg_val_loss, val_accuracy))

    def train(self, save=True, path=None):
        self.collection = Collection(triplets_tsv=self.args.triples)
        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        triplets_dataset = TripletsDataset(self.collection, self.inference)

        dataset_len = len(triplets_dataset)
        self.train_len = int(dataset_len * self.train_split)
        self.val_len = int(dataset_len * (1 - self.train_split))
        (train_x, val_x) = random_split(triplets_dataset, [self.train_len, self.val_len],
                                        generator=torch.Generator().manual_seed(42))

        train_data_loader = DataLoader(train_x, shuffle=True, batch_size=self.args.bsize)
        val_data_loader = DataLoader(val_x, batch_size=self.args.bsize)

        self.train_steps = len(train_data_loader.dataset) // self.args.bsize
        self.val_steps = len(val_data_loader.dataset) // self.args.bsize

        print('[INFO] training the network...')
        start_time = time.time()
        self.fit(train_data_loader, val_data_loader)
        print('[INFO] total time taken to train the model: {:.2f}s'.format(time.time() - start_time))
        if save:
            self.save(path)

    def fit(self, train_loader, val_loader):
        pass

    def predict(self, x, path=None):
        pass

    def save(self, path=None):
        if path is None:
            torch.save(self.model, './nn_ranker')
        else:
            torch.save(self.model, path)

    @staticmethod
    def load(path=None):
        if path is None:
            return torch.load('./nn_ranker')
        else:
            return torch.load(path)
