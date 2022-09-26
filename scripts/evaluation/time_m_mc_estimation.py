'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse
import json
from time import time

import numpy as np
import pandas as pd
import pytorch_lightning
from tqdm import tqdm

from utilities.Utility import load_utility



def time_m_mc_estimate(m, utility, df):
    # Next get the utility function
    utility = load_utility(utility, chrf_parallel=False)

    temp_df = df[["source", "hypotheses", "references", "references_count"]]

    class GetNReferences:
        def __init__(self, n):
            self.n = n

        def __call__(self, x):
            p = x["references_count"] / np.sum(x["references_count"])

            refs = np.random.choice(x["references"], p=p, replace=True, size=self.n)

            # Make sure we have don't have double references
            refs = list(set(refs))

            return refs

    # we sample from the 1000 samples. If m=1000 we keep the original samples
    if m != 1000:
        get_n_references = GetNReferences(m)
        temp_df["references"] = temp_df[["references", "references_count"]].apply(get_n_references, axis=1)
    else:
        temp_df["references"] = temp_df[["references"]].apply(lambda x: x["references"].tolist(), axis=1)
    def time_utility(x):

        result = []
        source = x["source"]

        hyp_list = x["hypotheses"].tolist()
        references = x["references"]

        for hyp in hyp_list:
            start_time = time()
            utility.call_batched_fast(source, [hyp], references)
            end_time = time()

            result.append(end_time - start_time)

        return result

    #
    tqdm.pandas()
    temp_df["times"] = temp_df.progress_apply(time_utility, axis=1)

    all_times = []
    for _, r in temp_df.iterrows():
        all_times += r["times"]
    print(len(all_times))
    return np.mean(all_times)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Times for a given utility function')


    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--utility', type=str, default="comet", help='which utility function to use')

    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')

    parser.add_argument('--split', type=str, default="test",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    parser.add_argument('--seed', type=int, default=0,
                       help="seed number (when we need different samples, also used for identification)")
    parser.add_argument('--n-sources', type=int,
                        default=100,
                        help='how many sources we use for calculating the speed')

    args = parser.parse_args()

    base_dir = './data/samples/'

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)


    # Load the references
    ref_data_loc = base_dir + '{}_{}_{}_{}'.format(args.sampling_method, 1000, args.split,args.seed)
    hyp_data_loc = base_dir + '{}_{}_{}_{}'.format(args.sampling_method, args.n_hypotheses, args.split, 0) # We will always use seed 0 for the hypothesis


    ref_data_loc += '.parquet'
    hyp_data_loc += '.parquet'

    references_df = pd.read_parquet(ref_data_loc)[:args.n_sources]
    hypothesis_df = pd.read_parquet(hyp_data_loc)[:args.n_sources]


    # Then we merge the two dataframes

    result_df = hypothesis_df

    result_df.rename({
        "samples": 'hypotheses',
        "sample_count": 'hypotheses_count',
    }, inplace=True, axis=1)

    result_df["references"] = references_df["samples"]
    result_df["references_count"] = references_df["sample_count"]

    results = {}
    ms = [1,2,3,4,5,10,25,100, 1000]
    for m in ms:
        mean_time = time_m_mc_estimate(m, args.utility, result_df)
        results[m] = mean_time
        print(results)
    result_ref = "./results/{}/m_estimation_timing_.json".format(args.utility)

    with open(result_ref, 'w') as f:
        json.dump(results, f)




if __name__ == '__main__':
    main()
