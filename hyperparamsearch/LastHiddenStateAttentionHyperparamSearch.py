from models.QualityEstimationStyle.BasicLstmModel.BasicLstmTrainer import BasicLstmModelTrainer
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelTrainer import LastHiddenStateModelTrainer

from utilities.callbacks import CustomSaveCallback

from argparse import Namespace

import joblib
import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utilities.config.ConfigParser import ConfigParser


class LastHiddenStateAttentionHyperparamSearch:

    def __init__(self, config, smoke_test, utility='comet'):

        self.config = config
        self.smoke_test = smoke_test
        self.utility = utility

        self.trainer = LastHiddenStateModelTrainer

        self.study_name = "last_hidden_state_attention_study"

        self.log_dir = './logs/{}/'.format(self.study_name)
        self.save_location = './saved_models/{}/'.format(self.study_name)

        self.n_warmup_steps = 10
        self.n_trials = 30

        self.model_type = "last_hidden_state"

        self.possible_dims = {
            "small": [512, 1],
            "medium": [512, 256, 1],
            "large": [512, 256, 128, 1],
        }

    def objective(self, trial: optuna.trial.Trial) -> float:

        max_epochs = 5 if self.smoke_test else 200  # More than enough, early stopping takes care that we stop on time

        # Create the configuration
        config = self.get_config(trial)

        # Create the trainer
        model_trainer = self.trainer(config, smoke_test=self.smoke_test)
        model_trainer.load_common()

        # Load the dataframe
        train_dataframe, validation_dataframe = model_trainer.load_data()

        # Load the manager and model
        manager, model = model_trainer.load_manager_and_model()

        # Create the dataloaders
        train_dataloader, val_dataloader = model_trainer.get_dataloaders(train_dataframe, validation_dataframe)

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_dir)

        save_callback = CustomSaveCallback(manager, self.save_location + str(trial.number) + "/")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5),
                       LearningRateMonitor(logging_interval="epoch"),
                       PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                       save_callback
                       ],
            logger=tb_logger,
            accumulate_grad_batches=1,
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


        config = {
            "model_name": 'basic_lstm',
            "gradient_clip_val": trial.suggest_float('gradient_clip_val', 1.5, 5.0),
            "model": model_config,
            "dataset": dataset_config,
            "batch_size": model_config["batch_size"],
            "preprocess": {
                "name": "basic"
            },
            "collator": {
                "name": "nmt_collator"
            },


        }
        config_parser = ConfigParser(self.utility)
        config = config_parser.parse(config)
        return config

    def get_model_config(self, trial):

        batch_size = 64
        accumulate_grad_batches = trial.suggest_categorical("accumulate_grad_batches", [2,4, 8])


        feed_forward_size = trial.suggest_categorical("feed_forward_size", ["small", "medium", "large"])

        dims = self.possible_dims[feed_forward_size]

        return {

            "batch_size": batch_size,
            'accumulate_grad_batches': accumulate_grad_batches,
            "type": "last_hidden_state_model",
            "lr": trial.suggest_float('lr', 1.0e-4, 1.0e-1, log=True),  # Not used
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-9, 1.0e-5, log=True),
            "dropout": trial.suggest_float("dropout", 0.01, 0.9, ),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "hidden_state_size": 512,
            "pooling": {
                "name": "attention",
                "embedding_size": 512,
                "n_heads": trial.suggest_categorical('n_heads', [2,4,8]),
            },

            "feed_forward_layers": {
                "dims": dims,
                "activation_function": "relu",
                "activation_function_last_layer": "tanh",
                "last_layer_scale": 2.5,

            },

            "nmt_model": {

                "name": 'Helsinki-NLP/opus-mt-de-en',
                "checkpoint": 'NMT/tatoeba-de-en/model',
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
            "dir": 'predictive/tatoeba-de-en/data/raw/',
            "preproces_dir": 'predictive/tatoeba-de-en/data/preprocessed/',
            "sampling_method": 'ancestral',
            "n_hypotheses": 10,
            "n_references": 100,
            "repeated_indices": False,
            "utility": self.utility,

        }
