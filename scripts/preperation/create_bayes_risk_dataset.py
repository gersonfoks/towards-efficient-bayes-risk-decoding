'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
from tqdm import tqdm

from utilities.Utility import load_utility


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')
    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(smoke_test=False)

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    parser.add_argument('--utility', type=str, default="comet", help='which utility function to use')

    parser.add_argument('--n-hypotheses', type=int, default=10, help='Number of hypothesis to use')



    parser.add_argument('--n-references', type=int, default=100, help='Number of references for each hypothesis')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")

    parser.add_argument('--seed', type=int, default=0,
                       help="seed number (when we need different samples, also used for identification)")

    args = parser.parse_args()

    base_dir = './data/samples/'

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)


    # Load the references
    ref_data_loc = base_dir + '{}_{}_{}'.format(args.sampling_method, args.n_references, args.seed)
    hyp_data_loc = base_dir + '{}_{}_{}'.format(args.sampling_method, args.n_hypotheses, args.seed)

    if args.smoke_test:
        ref_data_loc += '_smoke_test'
        hyp_data_loc += '_smoke_test'
    ref_data_loc += '.parquet'
    hyp_data_loc += '.parquet'

    references_df = pd.read_parquet(ref_data_loc)
    hypothesis_df = pd.read_parquet(hyp_data_loc)


    # Then we merge the two dataframes

    result_df = hypothesis_df

    result_df.rename({
        "samples": 'hypotheses',
        "sample_count": 'hypotheses_count',
    }, inplace=True, axis=1)

    result_df["references"] = references_df["samples"]
    result_df["references_count"] = references_df["sample_count"]


    # Next get the utility function
    utility = load_utility(args.utility)



    def map_utility(x):
        source = x["source"]

        hyp_list = x["hypotheses"].tolist()
        references = x["references"].tolist()


        utilities = utility.call_batched_fast(source, hyp_list, references)

        return utilities
    #
    tqdm.pandas()
    result_df["utilities"] = result_df.progress_apply(map_utility, axis=1)


    # Drop the references to make space (can easily be added back later when needed)
    result_df.drop("references", axis=1, inplace=True)
    # Also drop the hypothesis count, as it can also be added back later easily (save more space)
    result_df.drop("hypotheses_count", axis=1, inplace=True)
    #

    save_loc = './data/{}/'.format(args.utility)

    Path(save_loc).mkdir(parents=True, exist_ok=True)

    # Create save location
    save_loc += '{}_{}_{}_{}'.format(args.sampling_method, args.n_hypotheses, args.n_references, args.seed)
    if args.smoke_test:
        save_loc += '_smoke_test'
    save_loc += '.parquet'


    #save_path = get_dataset_path(args.save_dir, args.utility, args.split, args.sampling_method, args.n_hypotheses, args.n_references, args.develop)
    result_df.to_parquet(save_loc)


if __name__ == '__main__':
    main()
