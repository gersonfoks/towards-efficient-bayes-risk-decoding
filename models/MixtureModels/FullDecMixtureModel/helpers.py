import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from collators.FullDecMixtureCollator import FullDecMixtureCollator
from collators.NMTCollator import NMTCollator
from custom_datasets.BayesRiskDataset import BayesRiskDataset
from utilities.misc import load_bayes_risk_dataframe

import numpy as np


def repeat_utilities(x):
    repeated_utilities = []

    for util, n_utils in zip(x["utilities"], x["references_count"]):
        repeated_utilities += [util] * n_utils
    return repeated_utilities

def prepare_dataframe(df):
    '''
    Prepares a dataframe such that it can be used to train the model
    :param df: dataframe to prepare
    :param tokenizer: tokenizer used to tokenize the source and hypothesis
    :return:
    '''


    df = df.explode(column=['hypotheses', 'utilities'])
    df.reset_index(drop=True, inplace=True)
    df.rename({"hypotheses": "hypothesis"}, inplace=True, axis=1)

    df["utilities"] = df[["utilities", 'references_count']].apply(repeat_utilities, axis=1)


    return df


def load_data(config, nmt_model, tokenizer, seed=0, smoke_test=False, utility="comet"):
    print("Preparing the data")
    train_df = load_bayes_risk_dataframe(config["dataset"]["sampling_method"],
                                         config["dataset"]["n_hypotheses"],
                                         config["dataset"]["n_references"],
                                         'train_predictive',
                                         seed=seed,
                                         smoke_test=smoke_test,
                                        utility=utility
                                         )

    val_df = load_bayes_risk_dataframe(config["dataset"]["sampling_method"],
                                       config["dataset"]["n_hypotheses"],
                                       config["dataset"]["n_references"],
                                       'validation_predictive',
                                       seed=seed,
                                       smoke_test=smoke_test,
                                       utility=utility
                                       )

    train_df = prepare_dataframe(train_df)

    val_df = prepare_dataframe(val_df)

    train_dataset = BayesRiskDataset(train_df)
    val_dataset = BayesRiskDataset(val_df)

    collator = FullDecMixtureCollator(nmt_model, tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=collator,
                                  batch_size=config["batch_size"], shuffle=True, )
    val_dataloader = DataLoader(val_dataset,
                                collate_fn=collator,
                                batch_size=config["batch_size"], shuffle=False, )

    return train_dataloader, val_dataloader


def load_test_data(nmt_model, tokenizer, utility="comet", seed=0, smoke_test=False):
    print("Preparing the data")
    test_df = load_bayes_risk_dataframe("ancestral",
                                         100,
                                         1000,
                                         'test',
                                         seed=seed,
                                         smoke_test=smoke_test,
                                            utility=utility
                                         )



    # Add the index
    test_df = test_df.reset_index()
    test_df["source_index"] = test_df["index"]

    temp = prepare_dataframe(test_df)

    test_dataset = BayesRiskDataset(temp)


    collator = FullDecMixtureCollator(nmt_model, tokenizer, include_source_id=True)


    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=collator,
                                  batch_size=32, shuffle=False, )


    return test_df, test_dataloader






