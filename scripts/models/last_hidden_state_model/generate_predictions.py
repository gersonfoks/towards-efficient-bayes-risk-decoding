### A simple script which we can use to train a model
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning

from tqdm import tqdm

from models.QualityEstimationStyle.FullDecModel.FullDecModelManager import FullDecModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelManager import LastHiddenStateModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.helpers import load_test_data, generate_predictions



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Get the predictions of a basic lstm model ')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/comet/last_hidden_state_model/best/',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')



    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    # We first load the model as the model also has the tokenizer that we want to use

    model, model_manager = LastHiddenStateModelManager.load_model(args.model_path)

    generate_predictions(model, model_manager, args.utility, smoke_test=args.smoke_test, seed=args.seed)







if __name__ == '__main__':
    main()
