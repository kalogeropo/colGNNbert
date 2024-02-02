from os import getcwd
from os.path import join, exists

from pandas import read_csv, DataFrame


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
        prev_index = -1
        temp = []
        for line in lines:
            line = line.split(',')
            cur_index = int(line[1])
            if prev_index != cur_index:
                temp =[]
                list_to_group_by.append(temp)
                prev_index = cur_index
            #print(line)
            temp.append(int(line[3].replace(">", "")))
    print(len(list_to_group_by))
    return list_to_group_by


def get_retrieved_from_file(retrieved_tsv='experiments/cystic_2385/retrieve.py/2024-01-16_18.19.42/ranking.tsv'):
    list_to_group_by = []
    with open(retrieved_tsv, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split('\t')
            # print(line[0],line[1])
            list_to_group_by.append([int(line[0]), int(line[1])])
    return list_to_group_by


def retrieved_and_rel_parser(tsv_list=None):
    dict = {}
    if tsv_list is None:
        tsv_list = []
        return 0
    # print(tsv_list)
    prev_id = tsv_list[0][0]
    lst = []
    for item in tsv_list:
        lst.append(item[1])
        if item[0] != prev_id:
            dict[item[0]] = lst
            lst = []
        prev_id = item[0]
    # print(dict.keys())
    return dict


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

    def create_set(self):
        test_set = []
        for item in self.triplets:
            query = item[0]
            pos_doc = item[1]
            neg_doc = item[2]
            test_set.append([query, pos_doc, 1])
            test_set.append([query, neg_doc, 0])
        df = DataFrame(data=test_set, columns=['Query', 'Doc', 'Pos/Neg'])
        return df
