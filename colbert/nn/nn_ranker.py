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
        self.total_train_loss = 0
        self.total_val_loss = 0
        self.train_correct = 0
        self.val_correct = 0

    def train(self, save=True, path=None):
        self.collection = Collection(triplets_tsv=self.args.triples)
        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        triplets_dataset = TripletsDataset(self.collection, self.inference)

        dataset_len = len(triplets_dataset)
        train_len = int(dataset_len * self.train_split)
        val_len = int(dataset_len * (1 - self.train_split))
        (train_x, val_x) = random_split(triplets_dataset, [train_len, val_len],
                                        generator=torch.Generator().manual_seed(42))

        train_data_loader = DataLoader(train_x, shuffle=True, batch_size=self.args.bsize)
        val_data_loader = DataLoader(val_x, batch_size=self.args.bsize)

        train_steps = len(train_data_loader.dataset) // self.args.bsize
        val_steps = len(val_data_loader.dataset) // self.args.bsize

        self.fit(train_data_loader, val_data_loader)
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
