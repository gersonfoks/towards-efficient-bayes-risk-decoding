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
from models.reference_models.BasicCrossAttentionModel.trainer import BasicCrossAttentionModelTrainer
from models.reference_models.BasicReferenceLstmModel.trainer import BasicReferenceLstmModelTrainer
from models.reference_models.BasicReferenceLstmModelV2.trainer import BasicReferenceLstmModelV2Trainer
from models.reference_models.CometEncoddingModel.trainer import CometEncodingModelTrainer
from models.reference_models.FullDecCometModel.trainer import FullDecCometModelTrainer
from models.reference_models.FullDecUtilityModel.Trainer import FullDecUtilityModelTrainer
from models.reference_models.LastHiddenStateRefModel.trainer import LastHiddenStateRefModelTrainer
from models.source_hyp_models.EncDecLastHiddenModel.Trainer import EncDecLastHidenModelTrainer


model_trainers = {
    "hypothesis_lstm_model": HypothesisLstmModelTrainer,
    "hypothesis_decoder_model": TrainLastHiddenLSTMModel,
    "prop_entropy_model": ProbEntropyModelTrainer,
    "prop_entropy_model_v2": ProbEntropyModelTrainerV2,
    "enc_dec_last_hidden_model": EncDecLastHidenModelTrainer,
    "avg_std_prop_entropy_model": AvgStdPropEntropyModelTrainer,
    "last_hidden_prob_entropy_model": LastHiddenAndProbEntrLstmModelTrainer,
    "full_dec_model": FullDecModelTrainer,

    # Reference models
    "basic_ref_model": BasicReferenceLstmModelTrainer,
    "last_hidden_state_ref_model": LastHiddenStateRefModelTrainer,
    "basic_ref_model_v2": BasicReferenceLstmModelV2Trainer,
    "basic_cross_attention_model": BasicCrossAttentionModelTrainer,
    "full_dec_utility_model": FullDecUtilityModelTrainer,
    "comet_encoding_model": CometEncodingModelTrainer,
    "full_dec_comet_model": FullDecCometModelTrainer,
}

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,

                        default='./configs/unigram-f1/hypothesis-only-models/basic_ref_model.yml',
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
    print(model_type)
    train_model =model_trainers[model_type](config, smoke_test)
    train_model()


if __name__ == '__main__':
    main()
