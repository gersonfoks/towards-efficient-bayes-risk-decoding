'''
This file contains the code for the BayesRisk dataset
'''


class BayesRiskDataset:


    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe.index)


    def __getitem__(self, item):
        x = self.dataframe.iloc[item]


        # Do some processing

        return x


