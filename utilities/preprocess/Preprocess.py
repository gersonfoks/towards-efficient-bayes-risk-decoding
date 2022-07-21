import pandas as pd
from datasets import Dataset
import numpy as np
from utilities.LookUpTable import LookUpTable


class Preprocessor:

    def __init__(self, preprocess_functions):

        self.preprocess_functions = preprocess_functions


    def __call__(self, data: pd.DataFrame):

        for f in self.preprocess_functions:
            data = f(data)


        return data


class ResetIndex:

    def __call__(self, data):
        return data.reset_index(drop=True)


class HypToRefs:

    def __call__(self, data):
        data["references"] = data["hypotheses"]
        data["reference_counts"] = data["count"]
        return data


class AddHypIds:

    def __init__(self, add_ref_ids=True):
        self.add_ref_ids = add_ref_ids

    def __call__(self, data):


        data["hypotheses_ids"] = data.apply(self.create_hypothesis_ids, axis=1)
        # Also add the reference ids
        if self.add_ref_ids:
            data["reference_ids"] = data["hypotheses_ids"]
        return data

    def create_hypothesis_ids(self, x):


        hypotheses_ids = [
                "{}_{}".format(x.name, i) for i in range(len(x["hypotheses"]))
            ]
        return hypotheses_ids


class Explode:

    def __init__(self, cols=None, rename_columns=None):
        if cols is None:
            cols = ["hypotheses", "utilities", "count", "hypotheses_ids"]
        if rename_columns is None:
            rename_columns = {
                "hypotheses": "hypothesis",
                "utilities": "utility",
                "hypotheses_ids": "hypothesis_id"
            }
        self.cols = cols
        self.rename_columns = rename_columns

    def __call__(self, data):
        data = data.explode(self.cols, ignore_index=True)
        if self.rename_columns != None:

            data = data.rename(
                columns={
                    "hypotheses": "hypothesis",
                    "utilities": "utility",
                    "hypotheses_ids": "hypothesis_id"
                })
        return data


class UtilitiesToAverage:

    def __init__(self, new_col_name='utilities'):
        self.new_col_name = new_col_name


    def __call__(self, data):
        data[self.new_col_name] = data[["utilities", 'utilities_count']].apply(self.avg_utility, axis=1)

        return data

    def avg_utility(self, x):
        count = np.array(x['utilities_count'])
        utilities = x['utilities']

        result = []
        for util in utilities:
            result.append(
                np.sum(np.array(util) * count) / np.sum(count)
            )
        return pd.array(result, "float32")





