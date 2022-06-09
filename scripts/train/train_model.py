### A simple script which we can use to train a model
import argparse
import yaml

from models.hypothesis_only_models.FullDecHiddenLstmModel.Trainer import FullDecModelTrainer
from models.hypothesis_only_models.LastHiddenAndProbEntrLstmModel.Trainer import LastHiddenAndProbEntrLstmModelTrainer
from models.hypothesis_only_models.ProbEntropyModelV2.trainer import ProbEntropyModelTrainerV2
from models.hypothesis_only_models.AvgStdProbEntropyModel.Trainer import AvgStdPropEntropyModelTrainer

from models.hypothesis_only_models.HypothesisLstmModel.trainer import HypothesisLstmModelTrainer
from models.hypothesis_only_models.LastHiddenLstmModel.Trainer import TrainLastHiddenLSTMModel
from models.hypothesis_only_models.ProbEntropyModel.trainer import ProbEntropyModelTrainer
from models.source_hyp_models.EncDecLastHiddenModel.Trainer import EncDecLastHidenModelTrainer


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,

                        default='./configs/unigram-f1/hypothesis-only-models/full_dec_model.yml',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')

    parser.set_defaults(smoke_test=False)

    parser.set_defaults(on_hpc=False)

    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    model_type = config["model"]["type"]

    smoke_test = args.smoke_test
    if model_type == "hypothesis_lstm_model":
        train_model = HypothesisLstmModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "hypothesis_decoder_model":
        print("hypothesis_decoder_model")
        train_model = TrainLastHiddenLSTMModel(config, smoke_test)
        train_model()
    elif model_type == "prop_entropy_model":
        print("prop entropy model")
        train_model = ProbEntropyModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "prop_entropy_model_v2":
        print("prop_entropy_model_v2")
        train_model = ProbEntropyModelTrainerV2(config, smoke_test)
        train_model()
    elif model_type == "enc_dec_last_hidden_model":
        print("enc dec last hidden state model")
        train_model = EncDecLastHidenModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "avg_std_prop_entropy_model":
        print("avg_std_prob_entropy_model")
        train_model = AvgStdPropEntropyModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "last_hidden_prob_entropy_model":
        print("last_hidden_prob_entropy_model")
        train_model = LastHiddenAndProbEntrLstmModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "full_dec_model":
        print("full_dec_model")
        train_model = FullDecModelTrainer(config, smoke_test)
        train_model()
    else:
        raise ValueError("model type: {} not found".format(model_type))


if __name__ == '__main__':
    main()
