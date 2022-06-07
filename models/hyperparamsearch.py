from argparse import Namespace

import joblib
import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


class HyperparamSearch:

    def __init__(self, config, smoke_test, on_hpc=False):
        self.config = config
        self.smoke_test = smoke_test

        self.model_info = None

        self.study_name = None

        self.log_dir = None

        self.model_type = None

        self.n_warmup_steps = 5
        self.n_trials = 25

        self.batch_size = 128
        self.accumulate_grad_batches = 1
        self.on_hpc = on_hpc



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



    def get_model_config(self, trial):
        return {
            "type": self.model_type,
            "lr": trial.suggest_float("learning_rate", 1.0e-5, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-7, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.9, ),

            "feed_forward_layers": {
                "dims": [4, 32, 16, 1, ],
                "activation_function": "relu",
                "activation_function_last_layer": "sigmoid",

            },

            "optimizer": {
                "type": "adam_with_steps",
                "step_size": 1,
                "gamma": trial.suggest_float("learning_rate_decay", 0.25, 1.0, )
            },

            "nmt_model": {
                "model": {
                    "name": 'Helsinki-NLP/opus-mt-de-en',
                    "checkpoint": 'NMT/tatoeba-de-en/model',
                    "type": 'MarianMT'
                }
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
            "utility": "unigram-f1",

        }

    def load_model_and_manager(self, trial):
        model_config = self.get_model_config(trial)

        model_manager = self.model_info.manager(model_config)

        model = model_manager.create_model()

        return model, model_manager, model_config

    def load_dataset(self, trial, model, model_manager):
        raise NotImplementedError()

    def objective(self, trial: optuna.trial.Trial) -> float:
        model, manager, model_config = self.load_model_and_manager(trial)

        train_dataloader, val_dataloader = self.load_dataset(trial, model, manager)

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_dir)

        max_epochs = 1 if self.smoke_test else 30
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[EarlyStopping(monitor="val_loss", divergence_threshold=0.2),
                       LearningRateMonitor(logging_interval="epoch"),
                       PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            logger=tb_logger,
            accumulate_grad_batches=self.accumulate_grad_batches,
            gradient_clip_val=2.0
        )

        # create the dataloaders

        trainer.logger.log_hyperparams(Namespace(**model_config))

        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        return trainer.callback_metrics["val_loss"].item()
