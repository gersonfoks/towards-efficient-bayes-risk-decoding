### A simple script which we can use to train a model
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.QualityEstimationStyle.BasicModel.BasicLstmModelManager import BasicLstmModelManager
from models.QualityEstimationStyle.BasicModel.helpers import load_test_data
from utilities.misc import load_nmt_model
from utilities.wrappers.NmtWrapper import NMTWrapper




def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Get the predictions of a basic lstm model ')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/comet/basic_model/best/',
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

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use

    model, model_manager = BasicLstmModelManager.load_model(args.model_path)



    utility = args.utility

    # Load the dataset
    test_df, test_dataloader = load_test_data(model_manager.tokenizer, utility, smoke_test=args.smoke_test, seed=args.seed )






    predictions = [

    ]
    all_sources = [

    ]
    all_hypotheses = [

    ]
    indices = [

    ]

    model = model.to("cuda").eval()

    print("start gathering predictions")
    for x in tqdm(test_dataloader):

        sources, hypotheses, features, scores = x
        indices += features["source_index"]
        batch_out = model.predict(x)
        all_sources += sources
        all_hypotheses += hypotheses
        predictions += batch_out["predictions"].cpu().numpy().tolist()
    #
    results = pd.DataFrame({
        "source": all_sources,
        "hypothesis": all_hypotheses,
        "prediction": predictions,
        "source_index": indices

    })

    grouped_result = {
        "source": [],
        "hypotheses": [],
        "predictions": [],
    }

    indices = []

    for i, x in test_df.iterrows():
        index = x["source_index"]


        indices.append(index)
        source = x["source"]
        temp = results[results["source_index"] == index]
        grouped_result["source"] += [source]
        grouped_result["hypotheses"].append(temp["hypothesis"].to_list())
        grouped_result["predictions"].append(temp["prediction"].to_list())

    grouped_results = pd.DataFrame(grouped_result)



    base_dir = './model_predictions/{}/'.format(args.utility)
    #
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    grouped_results.to_parquet(base_dir + '{}_predictions.parquet'.format(model.name))


if __name__ == '__main__':
    main()
