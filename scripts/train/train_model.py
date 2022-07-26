### A simple script which we can use to train a model
import argparse
import yaml

from models.QualityEstimationStyle.BasicLstmModel.BasicLstmTrainer import BasicLstmModelTrainer
from models.QualityEstimationStyle.FullDecModel.FullDecModelTrainer import FullDecModelTrainer

from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelTrainer import LastHiddenStateModelTrainer
from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelTrainer import TokenStatisticsModelTrainer
from models.ReferenceStyle.ReferenceFullDecModel.ReferenceFullDecModelTrainer import ReferenceFullDecModelTrainer
from utilities.config.ConfigParser import ConfigParser

model_trainers = {
    "basic_lstm_model": BasicLstmModelTrainer,
    "last_hidden_state_model": LastHiddenStateModelTrainer,
    "token_statistics_model": TokenStatisticsModelTrainer,
    "full_dec_model": FullDecModelTrainer,
    "ref_full_dec_model": ReferenceFullDecModelTrainer,


}

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,

                        default='./configs/full_dec_lstm_model.yml',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.set_defaults(smoke_test=False)

    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_parser = ConfigParser(args.utility)
    config = config_parser.parse(config)

    model_type = config["model"]["type"]

    smoke_test = args.smoke_test
    train_model = model_trainers[model_type](config, smoke_test)
    train_model()


if __name__ == '__main__':
    main()
