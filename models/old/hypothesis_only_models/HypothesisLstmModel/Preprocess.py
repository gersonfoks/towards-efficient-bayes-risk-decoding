from datasets import Dataset
import numpy as np

def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score

class HypothesisLstmPreprocess:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __call__(self, data):
        df_exploded = self.explode_dataset(data)

        df_exploded["score"] = df_exploded["utilities"]

        # Create a unigram count vector for each hypotheses and put it into a lookup table

        dataset = Dataset.from_pandas(df_exploded)

        return dataset

    def explode_dataset(self, data):


        df_exploded = data.explode(["hypotheses", "utilities", "count",], ignore_index=True)

        return df_exploded
