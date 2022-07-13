'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from custom_datasets.BayesRiskDataset.BayesRiskDatasetCreator import BayesRiskDatasetCreator
from custom_datasets.SampleDataset.SampleDatasetLoader import SampleDatasetLoader
from utilities.PathManager import get_path_manager
from utilities.utilities import load_utility


def get_dataset_path(save_dir, utility, split, sampling_method, n_hypotheses, n_references, develop):
    path_manager = get_path_manager()
    relative_path = "{}/{}/".format(save_dir, utility, )
    abs_path = path_manager.get_abs_path(relative_path)

    Path(abs_path).mkdir(parents=True, exist_ok=True)

    abs_path += "{}_{}_scores_{}_{}".format(split, sampling_method,
                                            n_hypotheses, n_references, )
    if develop:
        abs_path += '_develop'
    abs_path += '.parquet'

    return abs_path


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')
    parser.add_argument('--develop', dest='develop', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(develop=False)

    parser.add_argument('--base-dir', type=str, default='NMT/tatoeba-de-en/data/')

    parser.add_argument('--save-dir', type=str, default='predictive/tatoeba-de-en/data/raw')

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

        utilities = utility.call_batched_fast(source, hyp_list, references)

        return utilities

    tqdm.pandas()
    hyp_dataset["utilities"] = hyp_dataset.progress_apply(map_utility, axis=1)

    hyp_dataset.drop("refs", axis=1, inplace=True)

    save_path = get_dataset_path(args.save_dir, args.utility, args.split, args.sampling_method, args.n_hypotheses, args.n_references, args.develop)
    hyp_dataset.to_parquet(save_path)


if __name__ == '__main__':
    main()
