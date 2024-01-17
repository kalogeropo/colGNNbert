
from os.path import join, exists
from os import listdir, getcwd, path

from pandas import read_csv


class Collection:
    """ A collection of documents consisted of:

            - path - collection path in disk/project, a .tsv file with ids and docs
            - num_docs - the number of documents in the collection
            - docs [] - a list of document from .tsv
            - queries - a list of queries for the collection
            - relevant - a list of relevant documents for each query
            - triplets - a list of triplets (Query, relevant, non-relevant)
                         for training purposes
    """

    def __init__(self, col_path='',  name='',doc_tsv='docs.tsv',
                 queries_tsv='Queries.tsv',relevant_tsv='Qrels.tsv',triplets_tsv='triplets.tsv'):
        if exists(relevant_tsv):
            self.relevant = self.get_rel_from_file(relevant_tsv)
        else:
            print(print(f'file {join(getcwd(), relevant_tsv)} does not exist'))
            self.relevant = []

        if exists( queries_tsv):
            self.queries = self.get_queries_from_file(queries_tsv)
        else:
            print(f'file {join(getcwd(), queries_tsv)} does not exist')
            self.queries = []

        if exists(triplets_tsv):
            self.triplets = self.get_triplets_from_file(triplets_tsv)
        else:
            print(f'file {join(getcwd(), triplets_tsv)} does not exist')
            self.triplets = []

        if exists(doc_tsv):
            self.docs = self.get_docs_from_file(doc_tsv)
        else:
            print(f'file {join(getcwd(), doc_tsv)} does not exist')
            self.docs = []
        self.name = name
        self.path = join(getcwd(), col_path)

        if exists(self.path):
            self.num_docs = len(self.docs)
        else:
            print(self.path)

    def get_rel_from_file(self):
        pass

    def get_docs_from_file(self,docs_tsv):
        docs = read_csv(docs_tsv, sep='\t', header=None)
        return docs[1].values.tolist()
    def get_triplets_from_file(self,triplets_tsv):
        triplets = read_csv(triplets_tsv, sep='\t', header=None)
        triplet_list = [triplets[i].tolist() for i in triplets.columns]
        triplet_list = zip(triplet_list[0], triplet_list[1], triplet_list[2])
        #print(len(list(triplet_list)))
        return triplet_list
    def get_queries_from_file(self,queries_tsv):
        queries = read_csv(queries_tsv, sep='\t', header=None)
        #print(queries)
        return queries[1].values.tolist()

