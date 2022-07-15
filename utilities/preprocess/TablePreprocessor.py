import pandas as pd

from utilities.LookUpTable import LookUpTable


class TablePreprocessor:

    def __init__(self, table_functions):
        self.table_functions = table_functions

    def __call__(self, data: pd.DataFrame):
        tables = {

        }
        for f in self.table_functions:
            tables[f.name] = f(data)

        return tables


class TokenStatisticsLookupTableCreator:

    def __init__(self, wrapped_nmt_model, table_location=None, batch_size=32):
        self.wrapped_nmt_model = wrapped_nmt_model
        self.table_location = table_location
        self.table_ref = None

        if self.table_location != None:
            self.table_ref = table_location + 'prob_and_entropy/'

        self.features = [
            "prob",
            "entropy",
            "top_5"

        ]

        self.batch_size = batch_size

    def __call__(self, data):

        if self.table_ref != None and LookUpTable.exists(self.table_ref):
            look_up_table = LookUpTable.load(self.table_ref)
        else:
            data = data.map(self.wrapped_nmt_model.map_to_token_statistics, batch_size=self.batch_size,
                            batched=True).to_pandas()
            look_up_table = LookUpTable(data, index="hypothesis_id",
                                        features=["hypothesis", "utility", ] + self.features)

            if self.table_ref != None:
                look_up_table.save(self.table_ref)
        return look_up_table
