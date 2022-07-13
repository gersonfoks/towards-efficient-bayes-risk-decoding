import pandas as pd
from datasets import Dataset
import numpy as np
from utilities.LookUpTable import LookUpTable


class Preprocessor:

    def __init__(self, preprocess_functions, table_functions):

        self.preprocess_functions = preprocess_functions
        self.table_functions = table_functions

    def __call__(self, data: pd.DataFrame):

        for f in self.preprocess_functions:
            data = f(data)

        tables = [

        ]
        for f in self.table_functions:
            table = f(data)
            tables.append(table)

        return data, tables


class ResetIndex:

    def __call__(self, data):
        return data.reset_index()


class HypToRefs:

    def __call__(self, data):
        data["references"] = data["hypotheses"]
        data["reference_counts"] = data["count"]
        return data


class AddHypIds:

    def __init__(self, add_ref_ids=True):
        self.add_ref_ids = add_ref_ids

    def __call__(self, data):
        dataset = Dataset.from_pandas(data)
        data = dataset.map(self.create_hypothesis_ids, batched=True, batch_size=32).to_pandas()
        # Also add the reference ids
        if self.add_ref_ids:
            data["reference_ids"] = data["hypotheses_ids"]
        return data

    def create_hypothesis_ids(self, x):
        hypotheses_ids = []
        for ids, hypotheses in zip(x["index"], x["hypotheses"]):
            hypotheses_ids.append([
                "{}_{}".format(ids, i) for i in range(len(hypotheses))
            ])
        return {
            "hypotheses_ids": hypotheses_ids,
            **x
        }


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
        return result





### Lookup table preprocessing

class GetProbEntropyLookupTable:

    def __init__(self, wrapped_nmt_model, table_location=None, batch_size=32):
        self.wrapped_nmt_model = wrapped_nmt_model
        self.table_location = table_location
        self.table_ref = None

        if self.table_location != None:
            self.table_ref = table_location + 'prob_and_entropy/'

        self.features = [
            "log_prob",
            "entropy",
            "mean_log_prob",
            "std_log_prob",
            "mean_entropy",
            "std_entropy",

        ]

        self.batch_size = batch_size

    def __call__(self, data):

        if self.table_ref != None and LookUpTable.exists(self.table_ref):
            look_up_table = LookUpTable.load(self.table_ref)
        else:
            data = data.map(self.wrapped_nmt_model.map_to_log_probs_and_entropy, batch_size=self.batch_size,
                            batched=True).to_pandas()
            look_up_table = LookUpTable(data, index="hypothesis_id",
                                        features=["hypothesis", "utility", ] + self.features)

            if self.table_ref != None:
                look_up_table.save(self.table_ref)
        return look_up_table
