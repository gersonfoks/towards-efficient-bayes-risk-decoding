'''
File to train the basic model
'''
import argparse

import numpy as np
import pytorch_lightning
import torch

from tqdm import tqdm


from models.ReferenceStyle.BasicReferenceModel.BasicReferenceModelManager import BasicReferenceModelManager
from models.ReferenceStyle.BasicReferenceModel.helpers import load_data_for_timing


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Gets the mean and std running time of the comet model')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/comet/basic_reference_model_1/best/',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use

    model, model_manager  = BasicReferenceModelManager.load_model(args.model_path)


    ### First load the dataset

    test_dataloader = load_data_for_timing(model_manager.nmt_model, model_manager.tokenizer)



    model.eval()
    model = model.to("cuda")

    timings = []
    print("start timing")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):

            timings.append(model.timed_forward(batch))

    print("mean time:", np.mean(timings))
    print("std time:", np.std(timings))

if __name__ == '__main__':
    main()
