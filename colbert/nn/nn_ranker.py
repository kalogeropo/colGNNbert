from torch import load as t_load
from torch import load as t_save
from colbert.utils.collection_parser import Collection
from colbert.modeling.colbert import ColBERT


class NNRanker:
    def __init__(self, args):
        self.args = args
        self.model = None

    def train(self, save=True, path=None):
        col = Collection(triplets_tsv=self.args.triples)
        colbert = ColBERT()
        dataset = col.create_set()
        dataset['Q'] = dataset['Query'].apply(colbert.query())

        self.fit(M, y)
        if save:
            self.save(path)

    def fit(self, M, y):
        pass

    def predict(self, M):
        pass

    def save(self, path=None):
        if path is None:
            t_save(self.model, './nn_ranker')
        else:
            t_save(self.model, path)

    @staticmethod
    def load(path=None):
        if path is None:
            return t_load('./nn_ranker')
        else:
            return t_load(path)
