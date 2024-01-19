from torch import load as torch_load
from torch import save as torch_save

from colbert.modeling.inference import ModelInference
from colbert.utils.collection_parser import Collection


class NNRanker:
    def __init__(self, args):
        self.args = args
        self.inference = None
        self.model = None

    def train(self, save=True, path=None):
        self.inference = ModelInference(self.args.colbert, amp=self.args.amp)
        collection = Collection(triplets_tsv=self.args.triples)
        triplets = collection.create_set()

        def create_similarities_matrix(q, d):
            q = self.inference.queryFromText(q)
            d = self.inference.docFromText(d)
            return self.inference.scores_matrix(q, d)

        similarities = triplets[['Query', 'Doc']].agg(create_similarities_matrix, axis='columns')

        self.fit(similarities, triplets['Pos/Neg'])
        if save:
            self.save(path)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass

    def save(self, path=None):
        if path is None:
            torch_save(self.model, './nn_ranker')
        else:
            torch_save(self.model, path)

    @staticmethod
    def load(path=None):
        if path is None:
            return torch_load('./nn_ranker')
        else:
            return torch_load(path)
