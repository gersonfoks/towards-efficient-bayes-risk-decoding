### Functions that help with the preprocessing
import numpy as np



def calc_mean_utility(x):
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

