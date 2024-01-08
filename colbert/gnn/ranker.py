class GNNRanker:
    def __init__(self):
        pass

    def train(self, C, Q, R, save=False, path=None):
        m = self.fit(C, Q, R)
        if save:
            if path is None:
                self.save()
            else:
                self.save(path)

    def fit(self, C, Q, R):
        pass

    def predict(self, M):
        pass

    def save(self, path='./gnn_ranker'):
        pass

    def load(self, path='./gnn_ranker'):
        pass
