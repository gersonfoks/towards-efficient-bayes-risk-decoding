'''
This file is used to create a bayes risk dataset.
We need to define how many hypotheses and references we want. Furthermore we need to define which utility function we use.
'''

import argparse
from pathlib import Path

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

    ### Load the dataset
    df = load_bayes_risk_dataframe(args.sampling_method,
                                         args.n_hypotheses,
                                        args.n_references,
                                         args.split,
                                         seed=args.seed,
                                         smoke_test=args.smoke_test,
                                         utility='comet',
                                         )[["hypotheses", "source"]]


    ### Load comet model
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)
    model.to("cuda")
    model.eval()
    wrapped_model = CometWrapper(model)


    ### Create hypotheses ids
    def create_hyp_ids(x):
        idx = x.name

        hyp_idx = []
        for i in range(len(x["hypotheses"])):
            hyp_idx.append('{}_{}'.format(idx, i))
        return hyp_idx

    df["hypotheses_idx"] = df.apply(create_hyp_ids, axis=1)


    ### First we map the hypotheses to embeddings
    df_exploded = df.explode(["hypotheses", "hypotheses_idx"])

    def hypotheses_to_embeddings(x):
        with torch.no_grad():
            embeddings = wrapped_model.to_embedding(x["hypotheses"]).cpu().numpy()

        return {"embedding": embeddings, **x}

    dataset = Dataset.from_pandas(df_exploded)
    new_dataset = dataset.map(hypotheses_to_embeddings, batched=True, batch_size=8)

    new_df = new_dataset.to_pandas()

    # Save this new dataframe

    save_dir = './data/comet/tables/'

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    #
    save_location = save_dir + '{}_{}_{}_{}_hypotheses_embeddings'.format(args.sampling_method, args.n_hypotheses, args.split, args.seed)

    if args.smoke_test:
        save_location += '_smoke_test'
    save_location += '.parquet'
    new_df = new_df[["embedding", "hypotheses_idx"]]
    new_df.to_parquet(save_location)





    def to_source_embeddings(x):
        with torch.no_grad():
            embeddings = wrapped_model.to_embedding(x["source"]).cpu().numpy()

        return {"source_embedding": embeddings, **x}

    dataset = Dataset.from_pandas(df)
    new_dataset = dataset.map(to_source_embeddings,  batched=True, batch_size=64)


    new_df = new_dataset.to_pandas()


    ref_save_location = save_dir + '{}_{}_{}_{}_source_embeddings'.format(args.sampling_method, args.n_references,
                                                                          args.split, args.seed)

    if args.smoke_test:
        ref_save_location += '_smoke_test'
    ref_save_location += '.parquet'

    new_ref_df = new_df[["source_embedding",]]

    new_ref_df.to_parquet(ref_save_location)



if __name__ == '__main__':
    main()
