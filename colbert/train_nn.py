import os
from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_qrels, load_queries, load_topK_pids

from colbert.nn.cnn_ranker import CNNRanker
from colbert.nn.gnn_ranker import GNNRanker


def main():
    parser = Arguments(description='Training late interaction NN on a binary problem.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_nn_training_parameters()
    parser.add_nn_training_input()

    args = parser.parse()

    args.lazy = args.collection is not None

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)

        nn = None
        if args.cnn:
            nn = CNNRanker(args)
        if args.gnn:
            nn = GNNRanker(args)
        nn.train()


if __name__ == "__main__":
    main()
