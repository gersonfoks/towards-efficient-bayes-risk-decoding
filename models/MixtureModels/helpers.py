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


    collator = NMTCollator(nmt_model, tokenizer, include_source_id=True)


    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=collator,
                                  batch_size=32, shuffle=False, )


    return test_df, test_dataloader


def load_data_for_timing(nmt_model, tokenizer, utility="comet", seed=0, smoke_test=False):
    print("Preparing the data")
    test_df = load_bayes_risk_dataframe("ancestral",
                                         100,
                                         1000,
                                         'test',
                                         seed=seed,
                                         smoke_test=smoke_test,
                                            utility=utility
                                         )[:100]



    # Add the index
    test_df = test_df.reset_index()
    test_df["source_index"] = test_df["index"]

    temp = prepare_dataframe(test_df)

    test_dataset = BayesRiskDataset(temp)


    collator = NMTCollator(nmt_model, tokenizer, include_source_id=False)


    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=collator,
                                  batch_size=1, shuffle=False, )


    return test_dataloader

def generate_predictions(model, model_manager, utility, smoke_test=False, seed=0):


    # Load the dataset
    test_df, test_dataloader = load_test_data(model_manager.nmt_model, model_manager.tokenizer, utility,
                                              smoke_test=smoke_test, seed=seed)

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

    base_dir = './model_predictions/{}/'.format(utility)
    #
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    grouped_results.to_parquet(base_dir + '{}_predictions.parquet'.format(model.name))



def time_model(model, model_manager,):
    test_dataloader = load_data_for_timing(model_manager.nmt_model, model_manager.tokenizer,)

    model.eval()
    model = model.to("cuda")
    ### Train
    timings = []
    print("start timing")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            timings.append(model.timed_forward(batch))
    print("mean time: {}".format(np.mean(timings)))

    result = np.mean(timings)

    base_dir = "./results/{}/".format(model.name)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    f_ref = base_dir + 'timing_result.json'.format(model.name)

    with open(f_ref, 'w') as f:
        json.dump({"mean_time": result}, f)



