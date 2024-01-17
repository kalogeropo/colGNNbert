from os.path import join, exists
from os import getcwd

from pandas import read_csv


def get_queries_from_file(queries_tsv):
    queries = read_csv(queries_tsv, sep='\t', header=None)
    # print(queries)
    return queries[1].values.tolist()


def get_docs_from_file(docs_tsv):
    docs = read_csv(docs_tsv, sep='\t', header=None)
    return docs[1].values.tolist()


def get_triplets_from_file(triplets_tsv):
    triplets = read_csv(triplets_tsv, sep='\t', header=None)
    triplet_list = [triplets[i].tolist() for i in triplets.columns]
    triplet_list = zip(triplet_list[0], triplet_list[1], triplet_list[2])
    # print(len(list(triplet_list)))
    return triplet_list


def get_rel_from_file(relevant_tsv):
    list_to_group_by = []
    with open(relevant_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            list_to_group_by.append([int(line[1]), int(line[3].replace(">", ""))])
    # print(list_to_group_by)
    return list_to_group_by


def get_retrived_from_file(retrieved_tsv='experiments/cystic_2385/retrieve.py/2024-01-16_18.19.42/ranking.tsv'):
    list_to_group_by = []
    with open(retrieved_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            # print(line[0],line[1])
            list_to_group_by.append([int(line[0]), int(line[1])])


def retrieval_and_rel_parser(tsv_list=None):
    if tsv_list is None:
        tsv_list = []


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

    def __init__(self, col_path='', name='', doc_tsv='docs.tsv',
                 queries_tsv='Queries.tsv', relevant_tsv='Qrels.tsv', triplets_tsv='triplets.tsv'):
        if exists(relevant_tsv):
            self.relevant = get_rel_from_file(relevant_tsv)
        else:
            print(print(f'file {join(getcwd(), relevant_tsv)} does not exist'))
            self.relevant = []

        if exists(queries_tsv):
            self.queries = get_queries_from_file(queries_tsv)
        else:
            print(f'file {join(getcwd(), queries_tsv)} does not exist')
            self.queries = []

        if exists(triplets_tsv):
            self.triplets = get_triplets_from_file(triplets_tsv)
        else:
            print(f'file {join(getcwd(), triplets_tsv)} does not exist')
            self.triplets = []

        if exists(doc_tsv):
            self.docs = get_docs_from_file(doc_tsv)
        else:
            print(f'file {join(getcwd(), doc_tsv)} does not exist')
            self.docs = []
        self.name = name
        self.path = join(getcwd(), col_path)

        if exists(self.path):
            self.num_docs = len(self.docs)
        else:
            print(self.path)
