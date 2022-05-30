### A simple script which we can use to train a model
import argparse
import pytorch_lightning as pl
import yaml
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader

from utilities.train.train_model import train_model_from_config


def main():

    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,

                        default='./configs/unigram-f1/source-hypotheses-models/enc_dec_last_hidden_model.yml',
                        help='config to load model from')



    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')



    parser.set_defaults(smoke_test=False)




    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()


    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    train_model_from_config(config, args.smoke_test)


if __name__ == '__main__':
    main()
