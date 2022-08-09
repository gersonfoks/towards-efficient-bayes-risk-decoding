'''
File to train the basic model
'''
import argparse

import yaml


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,
                        default='./configs/full_dec_lstm_model.yml',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)


    model_type = config["model"]["type"]

    smoke_test = args.smoke_test


    ### First load the dataset

    base_dir = './data/comet/'






    ### Next load the model and common things needed



    ### Prepare data for training
    # Tokenize
    # Create dataset
    # Create collator
    # Create dataloader


    ### Train


    ### Evaluate

if __name__ == '__main__':
    main()
