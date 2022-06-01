### A simple script which we can use to train a model
import argparse

from models.hypothesis_only_models.HiddenStateModel.hyperparamsearch import HiddenStateModelHyperParamSearch
from models.hypothesis_only_models.HypothesisLstmModel.hyperparamsearch import HypothesisLstmHyperParamSearch
from models.hypothesis_only_models.LastHiddenLstmModel.hyperparamsearch import LastHiddenStateLstmModelHyperParamSearch

models = {
    "hypothesis_lstm": HypothesisLstmHyperParamSearch,
    "hidden_state_model": HiddenStateModelHyperParamSearch,
    "last_hidden_state_model": LastHiddenStateLstmModelHyperParamSearch,
}


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--model_name', type=str, default='last_hidden_state_model',
                        help='the model to perform hyperparam search for')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    hyper_param_search = models[args.model_name](None, args.smoke_test)

    hyper_param_search()


if __name__ == '__main__':
    main()
