

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from custom_datasets.SampleDataset.SampleDatasetLoader import SampleDatasetLoader
from scripts.preprocessing.create_bayes_risk_dataset import load_utility
from utilities.misc import load_nmt_model

from utilities.utilities import NGramF





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')
    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')
    parser.add_argument('--n-references-estimate', type=int, default=1, help='Number of references for each hypothesis in the estimate')

    parser.add_argument('--split', type=str, default="validation_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    args = parser.parse_args()


    n_hypotheses = args.n_hypotheses
    n_references = args.n_references

    sampling_method = args.sampling_method

    dataset_dir = 'predictive/tatoeba-de-en/data/raw/'
    dataset_loader = BayesRiskDatasetLoader(args.split, n_hypotheses, n_references,
                                                  sampling_method, 'comet', develop=False,
                                                  base=dataset_dir)

    df = dataset_loader.load(type="pandas").data

    df = df.explode(['hypotheses', 'utilities'])
    print(df)
    print(list(df.columns))


    class GetMcEstimate:

        def __init__(self, n_references):
            self.n_references = n_references

        def __call__(self, x):



            count = np.array(x["utilities_count"])
            utilities = np.array(x["utilities"])

            probs = count / np.sum(count)
            # From the list of references pick n random
            picked_utilities = np.random.choice(utilities, size=self.n_references, p=probs, replace=True).tolist()


            return np.mean(picked_utilities)

    mc_estimate_map = GetMcEstimate(args.n_references_estimate)
    tqdm.pandas()
    df["mc_estimate_n"] = df.progress_apply(mc_estimate_map, axis=1)

    def get_mc_estimate(x):
        return np.array(np.sum(x["utilities"] * x["utilities_count"] / np.sum(x["utilities_count"])))

    df["mc_estimate"] = df.progress_apply(get_mc_estimate, axis=1)
    # Next we calculate the differences
    print(df["mc_estimate"])
    print(df["mc_estimate_n"])

    def calc_differences(x):
        difference =  (x["mc_estimate"]  - np.array(x["mc_estimate_n"]))
        return difference**2
    df["differences"] = df.progress_apply(calc_differences, axis=1)


    mse = df["differences"].mean()
    print(mse)
    #mse = dataset["differences"].explode("differences").mean()

if __name__ == '__main__':
    main()
