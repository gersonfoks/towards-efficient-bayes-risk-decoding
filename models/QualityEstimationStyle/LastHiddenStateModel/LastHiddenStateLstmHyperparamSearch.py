import numpy as np
import pytorch_lightning
from optuna.samplers import TPESampler

from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelManager import LastHiddenStateModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.helpers import load_data
from utilities.callbacks import CustomSaveCallback

from argparse import Namespace

import joblib
import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


class LastHiddenStateLstmHyperparamSearch:

    def __init__(self, smoke_test, utility='comet', seed=0):

        self.smoke_test = smoke_test
        self.utility = utility

        self.study_name = "last_hidden_state_lstm_model"

        self.log_dir = './logs/{}/'.format(self.study_name)
        self.save_location = './saved_models/{}/'.format(self.study_name)

        self.n_warmup_steps = 10
        self.n_trials = 30

        self.model_type = "last_hidden_state_model"

        self.possible_dims = {
            "small": [0, 1],
            "medium": [0, 256, 1],
            "large": [0, 256, 128, 1],
        }

        self.seed = seed
        np.random.seed(seed)
        pytorch_lightning.seed_everything(seed)

    def objective(self, trial: optuna.trial.Trial) -> float:

        max_epochs = 5 if self.smoke_test else 200  # More than enough, early stopping takes care that we stop on time

        # Create the configuration
        config = self.get_config(trial)

        model_manager = LastHiddenStateModelManager(config["model"])

        model = model_manager.create_model()

        tokenizer = model_manager.tokenizer
        nmt_model = model_manager.nmt_model



        train_dataloader, val_dataloader = load_data(config, nmt_model, tokenizer, seed=self.seed, smoke_test=self.smoke_test)

        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_dir)

        save_callback = CustomSaveCallback(model_manager, self.save_location + str(trial.number) + "/")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, verbose=True),
                       LearningRateMonitor(logging_interval="epoch"),
                       PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                       save_callback
                       ],
            logger=tb_logger,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=config["gradient_clip_val"]
        )

        # create the dataloaders

        trainer.logger.log_hyperparams(Namespace(**config["model"]))

        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        return save_callback.best_score.item()

    def __call__(self, ):
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner(n_warmup_steps=self.n_warmup_steps)
        )

        if self.smoke_test:
            self.n_trials = 5
        study = optuna.create_study(study_name=self.study_name, direction="minimize", pruner=pruner)
        study.optimize(self.objective, n_trials=self.n_trials, )

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("saving study")
        joblib.dump(study, "./study/{}.pkl".format(self.study_name))

    def get_config(self, trial):
        dataset_config = self.get_dataset_config()
        model_config = self.get_model_config(trial)
        accumulate_grad_batches = trial.suggest_categorical("accumulate_grad_batches", [2, 4, 8])

        config = {
            "model_name": 'basic_lstm',
            'accumulate_grad_batches': accumulate_grad_batches,
            "gradient_clip_val": trial.suggest_float("gradient clip val", 1.5, 5.0),
            "model": model_config,
            "dataset": dataset_config,
            "batch_size": model_config["batch_size"],

        }

        return config

    def get_model_config(self, trial):

        batch_size = 64

        feed_forward_size = trial.suggest_categorical("feed_forward_size", ["small", "medium", "large"])

        dims = self.possible_dims[feed_forward_size]

        hidden_state_size = trial.suggest_categorical("hidden_state_size", [256, 512, ])

        dims[0] = hidden_state_size * 2  # Because of the bidirectional part.

        return {

            "batch_size": batch_size,

            "type": "last_hidden_state_model",
            "lr": trial.suggest_float('lr', 1.0e-4, 1.0e-1, log=True),  # Not used
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-9, 1.0e-5, log=True),
            "dropout": trial.suggest_float("dropout", 0.01, 0.9, ),

            "hidden_state_size": hidden_state_size,
            "pooling": {
                "name": "lstm",
                "embedding_size": 512,
                "hidden_state_size": hidden_state_size,
            },

            "feed_forward_layers": {
                "dims": dims,
                "activation_function": "relu",
                "activation_function_last_layer": "tanh",
                "last_layer_scale": 2.5,

            },

            "nmt_model": {

                "name": 'Helsinki-NLP/opus-mt-de-en',
                "checkpoint": './saved_models/NMT/de-en-model/',
                "type": 'MarianMT'

            },

            "optimizer": {
                "type": "adam_with_lr_decay",
                "step_size": 1,
                "interval": "epoch",
                "gamma": trial.suggest_float("gamma", 0.5, 1.0)
            }

        }

    def get_dataset_config(self):
        return {
            "sampling_method": 'ancestral',
            "n_hypotheses": 10,
            "n_references": 100,
            "utility": self.utility,

        }
