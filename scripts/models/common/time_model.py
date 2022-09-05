'''
File to train the basic model
'''
import argparse

import numpy as np
import pytorch_lightning


from models.QualityEstimationStyle.FullDecModel.FullDecModelManager import FullDecModelManager
from models.QualityEstimationStyle.FullDecNoStatModel.FullDecNoStatModelManager import FullDecNoStatModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelManager import LastHiddenStateModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.helpers import time_model
from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelManager import TokenStatisticsModelManager

managers_map = {
    'full_dec_model': FullDecModelManager,
    'token_statistics_model': TokenStatisticsModelManager,
    'last_hidden_state_model': LastHiddenStateModelManager,
    'full_dec_no_stat_model': FullDecNoStatModelManager,

}


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gets the mean running time of the model IMPORTANT: (this script works for quality estimation style models minus basic model)')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/comet/full_dec_model/best/',
                        help='config to load model from')
    parser.add_argument('--model-name', type=str,
                        default='full_dec_model',
                        help='config to load model from')


    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use

    model, model_manager  = managers_map[args.model_name].load_model(args.model_path)


    ### First load the dataset
    time_model(model, model_manager)


if __name__ == '__main__':
    main()
