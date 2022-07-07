### A simple script which we can use to train a model
import argparse
import yaml

from models.reference_models.FullDecUtilityModel.Trainer import FullDecUtilityModelTrainer

def alter_config(config, i):
    save_model_path = 'saved_models/full_dec_utility_model_{}_references/'.format(i)
    log_dir = './logs/full_dec_utility_model_{}_references/'.format(i)

    config["save_model_path"] = save_model_path
    config["log_dir"] = log_dir

    config["model"]["n_references"] = i

    config["model"]["feed_forward_layers"]["dims"][0] = 2048 + i

    return config



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')

    base_config = './configs/unigram-f1/reference-models/full_dec_utility_model_base.yml'

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.set_defaults(smoke_test=False)

    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()

    with open(base_config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)



    smoke_test = args.smoke_test


    n_references = [
        3,
        5,
        100,  # Hardest one first to check if everything goes according to plan
        1,
        2,
        4,
        10,
        25,
        50,


    ]

    for i in n_references:
        new_config = alter_config(config.copy(), i)

        train_model = FullDecUtilityModelTrainer(new_config, smoke_test)
        train_model()




if __name__ == '__main__':
    main()
