import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split


def load_nn_ranker(path=None):
    if path is None:
        return torch.load('./nn_ranker')
    else:
        return torch.load(path)


class TripletsDataset(Dataset):
    def __init__(self, collection, inference, bsize):
        self.collection = collection
        self.inference = inference
        self.bsize = bsize
        self.similarities, self.labels = self._parse_triplets()

    def _parse_triplets(self):
        print('[INFO] reading triplets tsv file...')
        triplets = self.collection.create_set()
        print('[INFO] creating queries tensors...')
        q = self.inference.queryFromText(triplets['Query'].tolist(), self.bsize, True).permute(0, 2, 1)
        print('[INFO] creating documents tensors...')
        d = self.inference.queryFromText(triplets['Doc'].tolist(), self.bsize, True)
        print('[INFO] calculating the normalized similarities matrices...')
        m = pd.DataFrame({'Q': list(q), 'D': list(d)}).agg(self._create_normalized_scores, axis='columns')
        return m.tolist(), triplets['Pos/Neg'].tolist()

    def _create_normalized_scores(self, r):
        x = self.inference.scores_matrix(r[0], r[1])
        x = torch.add(x, 1)
        x = torch.div(x, 2)
        return x.view(1, x.shape[0], x.shape[1])

    def hw(self):
        if len(self.similarities):
            return self.similarities[0].shape[1], self.similarities[0].shape[2]
        return 0, 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.similarities[idx], self.labels[idx]


class NNRanker:
    def __init__(self, args):
        self.args = args
        self.collection = None
        self.inference = None
        self.model = None
        self.train_split = 0.9
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
        from colbert.modeling.inference import ModelInference
        from colbert.utils.collection_parser import Collection

        self.collection = Collection(triplets_tsv=self.args.triples)
        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        print('[INFO] creating triplets dataset...')
        triplets_dataset = TripletsDataset(self.collection, self.inference, self.args.bsize)
        h, w = triplets_dataset.hw()

        dataset_len = len(triplets_dataset)
        self.train_len = int(dataset_len * self.train_split)
        self.val_len = dataset_len - self.train_len
        print('[INFO] splitting dataset to training/validation...')
        (train, val) = random_split(triplets_dataset, [self.train_len, self.val_len],
                                    generator=torch.Generator().manual_seed(42))

        print('[INFO] creating torch data loaders...')
        train_data_loader = DataLoader(train, shuffle=True, batch_size=self.args.bsize)
        val_data_loader = DataLoader(val, batch_size=self.args.bsize)

        self.train_steps = self.train_len // self.args.bsize
        self.val_steps = self.val_len // self.args.bsize

        print('[INFO] training the network...')
        start_time = time.time()
        self.fit(train_data_loader, val_data_loader, h, w)
        print('[INFO] total time taken to train the model: {:.2f}s'.format(time.time() - start_time))
        if save:
            self.save(path)

    def fit(self, train_loader, val_loader, h=None, w=None):
        pass

    def predict(self, x, path=None):
        pass

    def save(self, path=None):
        if path is None:
            torch.save(self.model, './nn_ranker')
        else:
            torch.save(self.model, path)
