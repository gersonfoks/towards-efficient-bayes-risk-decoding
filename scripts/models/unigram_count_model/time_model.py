'''
File to train the basic model
'''
import argparse
import json
from pathlib import Path

import numpy as np
import pytorch_lightning
import torch

from tqdm import tqdm

from models.ReferenceStyle.UnigramCountModel.UnigramCountModelManager import UnigramCountModelManager
from models.ReferenceStyle.UnigramCountModel.helpers import load_data_for_timing


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gets the mean running time of the unigram count model')


    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--n-model-references', type=int, default=1,
                        help="The number of model references used")

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use
    path = './saved_models/comet/unigram_count_model_{}/best/'.format(args.n_model_references)
    model, model_manager  = UnigramCountModelManager.load_model(path)


    ### First load the dataset

    test_dataloader = load_data_for_timing(model_manager.nmt_model, model_manager.tokenizer, n_model_references=args.n_model_references)



    model.eval()
    model = model.to("cuda")

    timings = []
    print("start timing")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            timings.append(model.timed_forward(batch))

    base_dir = "./results/unigram_count_model_{}/".format(args.n_model_references)
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    f_ref = './results/unigram_count_model_{}/timing_result.json'.format(args.n_model_references)

    with open(f_ref, 'w') as f:
        json.dump({"mean_time": np.mean(timings)}, f)

if __name__ == '__main__':
    main()
