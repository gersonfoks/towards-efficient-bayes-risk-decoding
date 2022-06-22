### Functions that help with the preprocessing
import torch
import numpy as np
from datasets import Dataset

from models.reference_models.BasicReferenceLstmModelV2.Preprocess import create_hypothesis_ids
from utilities.LookUpTable import LookUpTable


def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score


def explode_dataset(data, columns=None):
    if columns is None:
        columns = ["hypotheses", "utilities", "count", "hypotheses_ids"]

    df_exploded = data.explode(columns, ignore_index=True).rename(
            columns={
                "hypotheses": "hypothesis",
                "utilities": "utility",
                "hypotheses_ids": "hypothesis_id"
            })

    return df_exploded


def add_hypothesis_ids(data):
    dataset = Dataset.from_pandas(data)
    dataset = dataset.map(create_hypothesis_ids, batched=True, batch_size=32).to_pandas()
    return dataset


def get_prob_entropy_lookup_table(data, nmt_wrapper, index='hypothesis_id', location='', batch_size=32):
    location = location + 'prob_and_entropy/'

    features = [
        "log_prob",
        "entropy",
        "mean_log_prob",
        "std_log_prob",
        "mean_entropy",
        "std_entropy",

    ]

    if LookUpTable.exists(location):
        look_up_table = LookUpTable.load(location)
    else:
        data = data.map(nmt_wrapper.map_to_log_probs_and_entropy, batch_size=batch_size,
                        batched=True).to_pandas()
        look_up_table = LookUpTable(data, index=index,
                                    features=["hypothesis", "utility", ] + features)

        look_up_table.save(location)
    return look_up_table
