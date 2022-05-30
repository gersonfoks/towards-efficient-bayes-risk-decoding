'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from custom_datasets.SampleDataset.SampleDatasetLoader import SampleDatasetLoader
from utilities.misc import load_nmt_model

from utilities.utilities import NGramF


class DecoderTokenizer:

    def __init__(self, nmt_tokenizer, max_seq_length=75):
        self.nmt_tokenizer = nmt_tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, sentences):
        with self.nmt_tokenizer.as_target_tokenizer():
            tokenized_sentences = self.nmt_tokenizer(sentences, truncation=True,
                                                     max_length=self.max_seq_length)["input_ids"]
        return tokenized_sentences


def load_utility(utility):
    if utility == "unigram-f1":
        # Get the nmt model tokenizer
        config = {
            "model": {
                "name": 'Helsinki-NLP/opus-mt-de-en',
                "checkpoint": 'NMT/tatoeba-de-en/model',
                "type": 'MarianMT'
            }
        }
        nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

        tokenizer = DecoderTokenizer(tokenizer)

        return NGramF(1, tokenize=True, tokenizer=tokenizer)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')


    parser.add_argument('--utility', type=str, default="unigram-f1")

    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')
    parser.add_argument('--n-references-estimate', type=int, default=1, help='Number of references for each hypothesis in the estimate')

    parser.add_argument('--split', type=str, default="validation_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    args = parser.parse_args()


    n_hypotheses = args.n_hypotheses
    n_references = args.n_references
    utility = args.utility
    sampling_method = args.sampling_method

    dataset_dir = 'predictive/tatoeba-de-en/data/raw/'
    dataset_loader = BayesRiskDatasetLoader(args.split, n_hypotheses, n_references,
                                                  sampling_method, utility, develop=False,
                                                  base=dataset_dir)

    dataset = dataset_loader.load(type="pandas").data

    # Reference dataset
    base_dir = 'NMT/tatoeba-de-en/data/'
    ref_dataset_loader = SampleDatasetLoader(args.split, args.n_references, args.sampling_method,
                                             base=base_dir, )

    ref_dataset = ref_dataset_loader.load()
    ref_dataset = pd.DataFrame.from_dict(ref_dataset.data)


    dataset["refs"] = ref_dataset["samples"]
    dataset["refs_count"] = ref_dataset["count"]



    # Next we calculate the utility
    utility = load_utility(args.utility)

    class GetMcEstimate:

        def __init__(self, n_references):
            self.n_references = n_references

        def __call__(self, x):
            source = x["source"]

            hyp_list = x["hypotheses"].tolist()
            references = x["refs"]

            count = np.array(x["refs_count"])

            probs = count / np.sum(count)
            # From the list of references pick n random
            picked_references = np.random.choice(references, size=self.n_references, p=probs, replace=True).tolist()



            scores = utility.call_batched(source, hyp_list, picked_references)

            # Next we calculate the average




            average = [np.mean(score) for score in scores]

            return average

    mc_estimate_map = GetMcEstimate(args.n_references_estimate)
    tqdm.pandas()
    dataset["mc_estimate_n"] = dataset.progress_apply(mc_estimate_map, axis=1)


    # Next we calculate the differences

    def calc_differences(x):
        difference = np.array(x["utilities"]) -np.array(x["mc_estimate_n"])
        return difference**2
    dataset["differences"] = dataset.progress_apply(calc_differences, axis=1)


    mse = dataset["differences"].explode("differences").mean()
    print(mse)
if __name__ == '__main__':
    main()
