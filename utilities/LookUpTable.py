import os.path
from pathlib import Path

import joblib
import pandas as pd
from tqdm import tqdm


class LookUpTable:

    def __init__(self, dataframe, index, features, ):
        # create
        self.table = {}
        self.index_map = []
        self.index = index
        self.features = features
        self.dataframe = dataframe
        for i, x in tqdm(dataframe.iterrows(), total=len(dataframe.index)):
            self.table[x[index]] = {
                feature: x[feature] for feature in features
            }
            self.index_map.append(x[index])


    def __getitem__(self, item):
         return self.table[item]

    def __len__(self):
        return len(self.index_map)


    def save(self, location):
        Path(location).mkdir(parents=True, exist_ok=True)

        dataframe_ref, info_ref = self.get_file_names(location)


        self.dataframe.to_parquet(dataframe_ref)
        info = [
            self.index,
            self.features
        ]

        joblib.dump(info, info_ref)

    @classmethod
    def get_file_names(self, location):

        dataframe_ref = location + 'dataframe.parquet'
        info_ref = location + "info.pkl"
        return dataframe_ref, info_ref

    @classmethod
    def load(cls, location):

        dataframe_ref, info_ref = cls.get_file_names(location)
        dataframe = pd.read_parquet(dataframe_ref)
        info = joblib.load(info_ref)
        return cls(dataframe, info[0], info[1])

    @classmethod
    def exists(cls, location):
        dataframe_ref, info_ref = cls.get_file_names(location)
        return os.path.exists(dataframe_ref) and os.path.exists(info_ref)
