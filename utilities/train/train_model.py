'''
This file contains all the functions that can be used to train a model
'''
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from models.hypothesis_only_models.AvgStdProbEntropyModel.AvgStdProbEntropyModelTrainer import \
    AvgStdPropEntropyModelTrainer
from models.hypothesis_only_models.HiddenStateModel.Trainer import HiddenStateModelTrainer
from models.hypothesis_only_models.HypothesisLstmModel.Collator import HypothesisLstmModelCollator
from models.hypothesis_only_models.HypothesisLstmModel.Preprocess import HypothesisLstmPreprocess
from models.hypothesis_only_models.HypothesisLstmModel.manager import HypothesisLstmModelManager
from models.hypothesis_only_models.HypothesisLstmModel.trainer import HypothesisLstmModelTrainer
from models.hypothesis_only_models.LastHiddenLstmModel.Trainer import TrainLastHiddenLSTMModel
from models.hypothesis_only_models.ProbEntropyModel.trainer import ProbEntropyModelTrainer

from models.source_hyp_models.EncDecLastHiddenModel.Trainer import EncDecLastHidenModelTrainer


def train_model_from_config(config, smoke_test=False):
    '''
    Selects which function to call to start training the model
    :param config:
    :return:
    '''

    model_type = config["model"]["type"]

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
    elif model_type == "enc_dec_last_hidden_model":
        print("enc dec last hidden state model")
        train_model = EncDecLastHidenModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "hidden_state_model":
        print("hidden state model")
        train_model = HiddenStateModelTrainer(config, smoke_test)
        train_model()
    elif model_type == "avg_std_prop_entropy_model":
        print("avg_std_prob_entropy_model")
        train_model = AvgStdPropEntropyModelTrainer(config, smoke_test)
        train_model()
    else:
        raise ValueError("model type: {} not found".format(model_type))


####

