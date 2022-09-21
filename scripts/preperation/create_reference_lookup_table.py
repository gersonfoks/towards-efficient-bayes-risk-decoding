'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from utilities.wrappers.CometWrapper import CometWrapper
from comet import download_model, load_from_checkpoint
import numpy as np
import pytorch_lightning

from utilities.misc import load_bayes_risk_dataframe, batch


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Give the COMET scores for hypothesis given a reference set')
    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true uses the develop set (with 100 sources) for fast development')

    parser.set_defaults(smoke_test=False)

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')



    parser.add_argument('--n-hypotheses', type=int, default=100, help='Number of hypothesis to use')



    parser.add_argument('--n-references', type=int, default=1000, help='Number of references for each hypothesis')

    parser.add_argument('--split', type=str, default="train_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")
    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.add_argument('--seed', type=int, default=0,
                       help="seed number (when we need different samples, also used for identification)")


    args = parser.parse_args()

    base_dir = './data/samples/'

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    reference_df_ref = './data/samples/{}_{}_{}_{}'.format(args.sampling_method, args.n_references, args.split, args.seed,)

    if args.smoke_test:
        reference_df_ref += "_smoke_test"
    reference_df_ref += '.parquet'

    reference_df = pd.read_parquet(reference_df_ref)


    ### Load the dataset
    df = load_bayes_risk_dataframe(args.sampling_method,
                                         args.n_hypotheses,
                                        args.n_references,
                                         args.split,
                                         seed=args.seed,
                                         smoke_test=args.smoke_test,
                                         utility=args.utility,
                                         )[['source', 'references_count', 'utilities' ]]

    print(df)
    print(reference_df)
    df["references"] = reference_df["samples"]


    df.reset_index(inplace=True)
    print(df["utilities"])
    save_dir = './data/{}/'.format(args.utility)
    ref_save_location = save_dir + '{}_{}_{}_{}_references'.format(args.sampling_method, args.n_references,
                                                                          args.split, args.seed)

    if args.smoke_test:
        ref_save_location += '_smoke_test'
    ref_save_location += '.parquet'



    df.to_parquet(ref_save_location)



if __name__ == '__main__':
    main()
