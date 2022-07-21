### A simple script which we can use to train a model
import argparse

from hyperparamsearch.BasicLstmModelHyperparamSearch import BasicLstmModelHyperparamSearch
from hyperparamsearch.LastHiddenStateAttentionHyperparamSearch import LastHiddenStateAttentionHyperparamSearch
from hyperparamsearch.LastHiddenStateLstmHyperparamSearch import LastHiddenStateLstmHyperparamSearch
from hyperparamsearch.TokenStatisticAttentionModelHyperparamSearch import TokenStatisticsAttentionModelHyperparamSearch
from hyperparamsearch.TokenStatisticLstmModelHyperparamSearch import TokenStatisticsLstmModelHyperparamSearch

models = {

    "basic_lstm_model": BasicLstmModelHyperparamSearch,
    "last_hidden_state_lstm_model": LastHiddenStateLstmHyperparamSearch,
    "last_hidden_state_attention_model": LastHiddenStateAttentionHyperparamSearch,
    "token_statistics_lstm": TokenStatisticsLstmModelHyperparamSearch,
    "token_statistics_attention": TokenStatisticsAttentionModelHyperparamSearch,

}


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--model-name', type=str, default='basic_lstm_model',
                        help='the model to perform hyperparam search for')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.add_argument('--utility', type=str, default='comet',
                        help='which utility to predict')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    hyper_param_search = models[args.model_name](None, args.smoke_test, args.utility)

    hyper_param_search()


if __name__ == '__main__':
    main()
