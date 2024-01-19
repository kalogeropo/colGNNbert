from colbert.utils.parser import Arguments
from colbert.utils.runs import Run
from colbert.nn.cnn_ranker import CNNRanker
from colbert.nn.gnn_ranker import GNNRanker


def main():
    parser = Arguments(description='Training late interaction NN on a binary problem.')

    parser.add_model_parameters()
    parser.add_model_training_parameters()
    parser.add_nn_training_parameters()
    parser.add_nn_training_input()

    args = parser.parse()

    args.lazy = args.collection is not None

    nn = None
    if args.cnn:
        nn = CNNRanker()
    if args.gnn:
        nn = GNNRanker()

    with Run.context(consider_failed_if_interrupted=False):
        nn.train(args)


if __name__ == "__main__":
    main()
