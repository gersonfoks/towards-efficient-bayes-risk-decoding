### A simple script which we can use to train a model
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning

from tqdm import tqdm

from models.QualityEstimationStyle.BasicModel.BasicLstmModelManager import BasicLstmModelManager
from models.QualityEstimationStyle.BasicModel.helpers import load_test_data
from utilities.constants import PRETTY_NAMES
from utilities.evaluation import  evaluate_predictions
from utilities.misc import load_bayes_risk_dataframe, map_to_utility

pretty_names = PRETTY_NAMES

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Analyses the predictions ')

    parser.add_argument('--predictions-ref', type=str,
                        default='./model_predictions/comet/basic_model_predictions.parquet',
                        help='Where to get the predictions from')

    parser.add_argument('--model-name', type=str, default='basic_model', help='name of the model')

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

    utility = args.utility





    # Load the test data
    test_df = load_bayes_risk_dataframe("ancestral",
                                        100,
                                        1000,
                                        'test',
                                        seed=args.seed,
                                        utility=utility
                                        )

    # Load the predictions df
    predictions_df = pd.read_parquet(args.predictions_ref)


    # Map utilities to mean utilities:
    test_df["utility"] = test_df[["utilities", 'references_count']].apply(map_to_utility, axis=1)

    test_df["predictions"] = predictions_df["predictions"]


    evaluate_predictions(test_df, args.model_name, pretty_names[args.model_name], args.utility)







if __name__ == '__main__':
    main()
