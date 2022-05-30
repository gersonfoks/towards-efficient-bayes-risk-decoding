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
from models.hypothesis_only_models.LastHiddenLstmModel.Trainer import TrainLastHiddenLSTMModel
from models.hypothesis_only_models.ProbEntropyModel.PropEntropyModelTrainer import PropEntropyModelTrainer

from models.source_hyp_models.EncDecLastHiddenModel.Trainer import EncDecLastHidenModelTrainer
from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


def train_model_from_config(config, smoke_test=False):
    '''
    Selects which function to call to start training the model
    :param config:
    :return:
    '''

    model_type = config["model"]["type"]

    if model_type == "hypothesis_lstm_model":
        train_lstm_model(config, smoke_test)
    elif model_type == "hypothesis_decoder_model":
        print("hypothesis_decoder_model")
        train_model = TrainLastHiddenLSTMModel(config, smoke_test)
        train_model()
    elif model_type == "prop_entropy_model":
        print("prop entropy model")
        train_model = PropEntropyModelTrainer(config, smoke_test)
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

def train_lstm_model(config, smoke_test):
    # First get the model:

    model_manager = HypothesisLstmModelManager(config["model"])

    model = model_manager.create_model()

    # Next load the datasets

    train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

    # Next do the preprocessing
    #
    preprocess = HypothesisLstmPreprocess()

    train_dataset_preprocessed = preprocess(train_dataset)
    validation_dataset_preprocessed = preprocess(validation_dataset)

    # Get the collate functions

    collate_fn = HypothesisLstmModelCollator(model_manager.tokenizer)

    train_dataloader = DataLoader(train_dataset_preprocessed,
                                  collate_fn=collate_fn,
                                  batch_size=config["batch_size"], shuffle=True, )
    val_dataloader = DataLoader(validation_dataset_preprocessed,
                                collate_fn=collate_fn,
                                batch_size=config["batch_size"], shuffle=False, )

    # Start the training
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])

    max_epochs = 1 if smoke_test else config["max_epochs"]
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        progress_bar_refresh_rate=1,
        val_check_interval=0.5,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        logger=tb_logger,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        gradient_clip_val=2.0
    )

    # create the dataloaders
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

    path_manager = get_path_manager()

    model_path = path_manager.get_abs_path(config["save_model_path"])
    model_manager.save_model(model_path)

    model, manager = HypothesisLstmModelManager.load_model(model_path)

    # create the dataloaders
    trainer.validate(model, val_dataloader, )
