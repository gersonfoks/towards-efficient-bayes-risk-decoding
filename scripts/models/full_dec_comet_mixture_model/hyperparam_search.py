'''
File to train the basic model
'''
import argparse

from models.MixtureModels.FullDecCometMixtureModel.FullDecCometMixtureModelHyperparamSearch import \
    FullDecCometMixtureModelHyperparamSearch

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Performs the hyperparam search for the full decoder model')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    hyperparamsearch = FullDecCometMixtureModelHyperparamSearch(args.smoke_test, args.utility, args.seed)
    hyperparamsearch()


if __name__ == '__main__':
    main()
