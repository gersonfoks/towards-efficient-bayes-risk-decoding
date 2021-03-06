'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from custom_datasets.BayesRiskDataset.BayesRiskDatasetCreator import BayesRiskDatasetCreator
from custom_datasets.SampleDataset.SampleDatasetLoader import SampleDatasetLoader
from utilities.PathManager import get_path_manager
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


def load_utility(utility, nmt_model=None, tokenizer=None):
    if utility == "unigram-f1":
        # Get the nmt model tokenizer
        config = {
            "model": {
                "name": 'Helsinki-NLP/opus-mt-de-en',
                "checkpoint": 'NMT/tatoeba-de-en/model',
                "type": 'MarianMT'
            }
        }
        if tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

        tokenizer = DecoderTokenizer(tokenizer)

        return NGramF(1, tokenize=True, tokenizer=tokenizer)
    else:
        raise ValueError("utility: {} not found!".format(utility))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')
    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='NMT/tatoeba-de-en/data/')

    parser.add_argument('--save-dir', type=str, default='predictive/tatoeba-de-en/data/raw/')

    parser.add_argument('--utility', type=str, default="unigram-f1")

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    args = parser.parse_args()

    ref_dataset_loader = SampleDatasetLoader(args.split, args.n_references, args.sampling_method, develop=args.develop,
                                             base=args.base_dir, )
    hyp_dataset_loader = SampleDatasetLoader(args.split, args.n_hypotheses, args.sampling_method, develop=args.develop,
                                             base=args.base_dir, )

    ref_dataset = ref_dataset_loader.load()
    hyp_dataset = hyp_dataset_loader.load()

    #Create a dataset loader (used to get the correct path)
    dataset_loader = BayesRiskDatasetCreator(args.split, args.n_hypotheses, args.n_references,
                                              args.sampling_method, args.utility, develop=args.develop,
                                              base=args.save_dir, )

    utility = load_utility(args.utility)


    hyp_dataset = pd.DataFrame.from_dict(hyp_dataset.data)
    ref_dataset = pd.DataFrame.from_dict(ref_dataset.data)


    hyp_dataset["refs"] = ref_dataset["samples"]
    hyp_dataset["hypotheses"] = hyp_dataset["samples"]
    hyp_dataset["utilities_count"] = ref_dataset["count"]

    hyp_dataset = hyp_dataset.drop("samples", axis=1)


    #
    def map_utility(x):
        source = x["source"]

        hyp_list = x["hypotheses"]
        references = x["refs"]



        scores = utility.call_batched_fast(source, hyp_list, references)

        # Next we calculate the average


        utilities_count = np.array(x["utilities_count"])

        average = [np.sum(utilities_count * np.array(score)) / np.sum(utilities_count) for score in scores]

        return average


    tqdm.pandas()
    hyp_dataset["utilities"] = hyp_dataset.progress_apply(map_utility, axis=1)

    print(hyp_dataset)


    hyp_dataset = hyp_dataset.drop("refs", axis=1)


    save_path = dataset_loader.get_dataset_path()
    hyp_dataset.to_parquet(save_path)






if __name__ == '__main__':
    main()
