#

import argparse

import numpy as np
import pandas as pd
import pytorch_lightning
import yaml

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from tqdm import tqdm

from models.QualityEstimationStyle.FullDecModel.FullDecModelManager import FullDecModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelManager import LastHiddenStateModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.helpers import load_data
from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelManager import TokenStatisticsModelManager
from utilities.callbacks import CustomSaveCallback

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utilities.misc import load_nmt_model
from utilities.wrappers.NmtWrapper import NMTWrapper


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Check the average computation speed of the NMT model')
    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()



    nmt_model, tokenizer = load_nmt_model({
        "name": 'Helsinki-NLP/opus-mt-de-en',
        "checkpoint": './saved_models/NMT/de-en-model/',
        "type": 'MarianMT'
    }, pretrained=True)



    wrapped_model = NMTWrapper(nmt_model, tokenizer)

    df = pd.read_parquet('./data/comet/ancestral_10_100_validation_predictive_0_smoke_test.parquet')

    df = df.explode(["hypotheses", "utilities"])

    timings = []
    print("start timing")
    for i in tqdm(range(len(df.index))):

        source = df["source"].iloc[i]
        hypothesis = df["hypotheses"].iloc[i]
        timings.append(wrapped_model.timed_forward(source, hypothesis))


    print("mean time:", np.mean(timings))
    print("std time:", np.std(timings))



if __name__ == '__main__':
    main()