
from os.path import exists


from datasets import Dataset

from torch.utils.data import DataLoader

from models.hyperparamsearch import HyperparamSearch
from models.hypothesis_only_models.AvgStdProbEntropyModel.info import AvgStdProbEntropyModelInfo
from models.reference_models.CrossAttentionModelV3.info import CrossAttentionModelV3Info
from models.reference_models.CrossAttentionModelV3.trainer import CrossAttentionModelV3Trainer

from utilities.PathManager import get_path_manager
from utilities.callbacks import CustomSaveCallback
from utilities.dataset.loading import load_dataset_for_training


from argparse import Namespace

import joblib
import optuna

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

class CrossAttentionV3CyclicHyperparamsearch:

    def __init__(self, config, smoke_test):

        self.config = config
        self.smoke_test = smoke_test

        self.model_info = CrossAttentionModelV3Info

        self.study_name = "cross_attention_v3_cyclic_study"

        self.log_dir = './logs/cross_attention_v3_cyclic_study/'
        self.save_location = './saved_models/cross_attention_v3_cyclic_study/'

        self.batch_size = 32


        self.n_warmup_steps = 10
        self.n_trials = 30

        self.model_type = "cross_attention_model_v3"


        self.possible_dims = {
            "small": [2048, 1],
            "medium": [2048, 512, 1],
            "large": [2048, 1024, 512, 256, 1],
        }

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

        model_size = trial.suggest_categorical("model_size", ["small", "medium", "large"])

        dims = self.possible_dims[model_size]

        base_lr = trial.suggest_float("max_lr",  1.0e-5, 1.0e-4)
        max_lr = trial.suggest_float("base_lr",  1.1e-4, 1.0e-2)

        return {
            "type": self.model_type,
            "lr": 0.1, # Not used
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-9,1.0e-5, log=True),
            "dropout": trial.suggest_float("dropout", 0.2, 0.7, ),
            "n_references": 3,
            "n_heads": trial.suggest_categorical("n_heads", [2,4,8]),
            "batch_norm": trial.suggest_categorical("batch_norm", [True, False]),
            "prob_entrop_embed_size": trial.suggest_categorical("prob_entrop_embed_size", [8, 32, 64]),
            "n_learnable_embeddings": trial.suggest_int("n_learnable_embeddings", 1, 3),


            "feed_forward_layers": {
                "dims": dims,
                "activation_function": "relu",
                "activation_function_last_layer": "sigmoid",

            },

            "optimizer": {
                "type": "cyclic_lr",
                "step_size_up": trial.suggest_categorical("step_size_up", [2000, 4000]),
                "mode": trial.suggest_categorical("mode", ["triangular", "triangular2"]),
                "base_lr": base_lr,
                "max_lr": max_lr,

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



    def objective(self, trial: optuna.trial.Trial) -> float:


        model_config = self.get_model_config(trial)
        dataset_config = self.get_dataset_config()
        max_epochs = 5 if self.smoke_test else 75
        config = {
            "model": model_config,
            "dataset": dataset_config,
            "batch_size": self.batch_size,


        }
        model_trainer = CrossAttentionModelV3Trainer(config, smoke_test=self.smoke_test)

        self.accumulate_grad_batches = trial.suggest_categorical("accumulate_batches", [2, 4, 8])

        manager = model_trainer.get_model_manager()
        model = model_trainer.get_model()


        train_dataloader, val_dataloader = model_trainer.get_dataloaders()

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.log_dir)



        save_callback = CustomSaveCallback(manager, self.save_location + str(trial.number) + "/")
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[EarlyStopping(monitor="val_loss", divergence_threshold=0.2, patience=15),
                       LearningRateMonitor(logging_interval="epoch"),
                       PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                       save_callback
                       ],
            logger=tb_logger,
            accumulate_grad_batches=self.accumulate_grad_batches,
        )

        # create the dataloaders

        trainer.logger.log_hyperparams(Namespace(**model_config))

        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        return save_callback.best_score.item()