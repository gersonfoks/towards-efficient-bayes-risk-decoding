from models.QualityEstimationStyle.FullDecModel.FullDecModelTrainer import FullDecModelTrainer
from models.ReferenceStyle.ReferenceFullDecModel.ReferenceFullDecModelTrainer import ReferenceFullDecModelTrainer

from utilities.callbacks import CustomSaveCallback

from argparse import Namespace

import joblib
import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utilities.config.ConfigParser import ConfigParser


class ReferenceFullDecLstmModelHyperparamSearch:

    def __init__(self, config, smoke_test, utility='comet'):

        self.config = config
        self.smoke_test = smoke_test
        self.utility = utility

        self.trainer = ReferenceFullDecModelTrainer

        self.study_name = "ref_full_dec_lstm_study"

        self.log_dir = './logs/{}/'.format(self.study_name)
        self.save_location = './saved_models/{}/'.format(self.study_name)

        self.n_warmup_steps = 10
        self.n_trials = 10

        self.model_type = "ref_full_dec_model"

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

        model_trainer.load_tables()

        # Create the dataloaders
        train_dataloader, val_dataloader = model_trainer.get_dataloaders(train_dataframe, validation_dataframe)

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_dir)

        save_callback = CustomSaveCallback(manager, self.save_location + str(trial.number) + "/")
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
            accumulate_grad_batches=4,
            gradient_clip_val=3.6,
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
            "model_name": 'full_dec_lstm',

            "model": model_config,
            "dataset": dataset_config,
            "batch_size": model_config["batch_size"],
            "preprocess": {
                "name": "refs_full_dec"
            },
            "table_location": './FBR/predictive/tatoeba-de-en/data/lookup_tables/',
            "collator": {
                "name": "ref_full_dec_collator",
                "n_references": 5,
            },

        }

        return config

    def get_model_config(self, trial):

        batch_size = 64

        dims = [
            2048, 128, 1
        ]

        hidden_state_size = 128
        token_embedding_size = 128

        return {

            "batch_size": batch_size,
            "type": "full_dec_model",
            "lr": trial.suggest_float('lr', 1.0e-5, 1.0e-1, log=True),
            "weight_decay": 4.0e-9,
            "dropout": 0.18,
            "embedding_size": token_embedding_size,
            "n_statistics": 7,

            "token_pooling": {
                "name": "lstm",
                "embedding_size": token_embedding_size,
                "hidden_state_size": hidden_state_size,
            },
            "dec_pooling": {
                "name": "lstm",
                "embedding_size": 512,
                "hidden_state_size": 128,
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
