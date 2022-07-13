import torch
from datasets import Dataset
import numpy as np
from transformers import DataCollatorForSeq2Seq

from utilities.wrappers.NmtWrapper import NMTWrapper


def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score


class ProbEntropyModelPreprocess:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, ):
        self.nmt_model = nmt_model
        self.nmt_model.eval()
        self.nmt_model =self.nmt_model.to("cuda")

        self.nmt_model_wrapped = NMTWrapper(nmt_model, tokenizer)

        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.data_collator = DataCollatorForSeq2Seq(model=nmt_model, tokenizer=tokenizer,
                                                    padding=True, return_tensors="pt", max_length=max_seq_length)
        self.batch_size = 32

    def __call__(self, data):
        df_exploded = self.explode_dataset(data)

        df_exploded["score"] = df_exploded["utilities"]

        dataset = Dataset.from_pandas(df_exploded)

        # Lastly we want to calculate the probs and entropy of all the tokens

        dataset = dataset.map(self.nmt_model_wrapped.map_to_log_probs_and_entropy, batch_size=self.batch_size, batched=True)


        return dataset

    def explode_dataset(self, data):
        df_exploded = data.explode(["hypotheses", "utilities", "count", ], ignore_index=True)

        return df_exploded


