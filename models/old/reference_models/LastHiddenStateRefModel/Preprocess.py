from datasets import Dataset


class LastHiddenStateRefModelPreprocessor:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __call__(self, data):
        # Add list of references
        data["references"] = data["hypotheses"]
        data["reference_counts"] = data["count"]

        df_exploded = self.explode_dataset(data)

        df_exploded["score"] = df_exploded["utilities"]


        dataset = Dataset.from_pandas(df_exploded)

        return dataset

    def explode_dataset(self, data):
        df_exploded = data.explode(["hypotheses", "utilities", "count", ], ignore_index=True)

        return df_exploded
