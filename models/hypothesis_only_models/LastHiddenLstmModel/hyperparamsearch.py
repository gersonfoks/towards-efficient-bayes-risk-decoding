from argparse import Namespace

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from models.hypothesis_only_models.HiddenStateModel.info import HiddenStateModelInfo
from models.hypothesis_only_models.HypothesisLstmModel.Collator import HypothesisLstmModelCollator
from models.hypothesis_only_models.HypothesisLstmModel.Preprocess import HypothesisLstmPreprocess
from models.hypothesis_only_models.HypothesisLstmModel.manager import HypothesisLstmModelManager
from models.hypothesis_only_models.LastHiddenLstmModel.info import LastHiddenStateLstmModelInfo
from utilities.dataset.loading import load_dataset_for_training
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


class LastHiddenStateLstmModelHyperParamSearch:

    def __init__(self, config, smoke_test):
        self.config = config
        self.smoke_test = smoke_test

        self.manager = LastHiddenStateLstmModelInfo.manager
        self.preprocess = LastHiddenStateLstmModelInfo.preprocess
        self.collate = LastHiddenStateLstmModelInfo.collate
        self.model = LastHiddenStateLstmModelInfo.model

    def __call__(self, ):
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(self.objective, n_trials=25, )

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial: optuna.trial.Trial) -> float:
        # Here we have our hyperparameters

        # Next we create a config:

        model_config = {
            "type": "hidden_state_model",
            "lr": trial.suggest_float("learning_rate", 1.0e-5, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-7, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.9, ),

            "feed_forward_layers": {
                "dims": [1024, 512, 256, 128, 1],
                "activation_function": "relu",
                "activation_function_last_layer": "sigmoid",

            },

            "embedding": {
                "size": 512
            },
            "optimizer": {
                "type": "adam_with_steps",
                "step_size": 1,
                "gamma": trial.suggest_float("learning_rate_decay", 0.01, 0.9, )
            },

            "nmt_model": {
                "model": {
                    "name": 'Helsinki-NLP/opus-mt-de-en',
                    "checkpoint": 'NMT/tatoeba-de-en/model',
                    "type": 'MarianMT'
                }
            }

        }

        dataset_config = {
            "dir": 'predictive/tatoeba-de-en/data/raw/',
            "sampling_method": 'ancestral',
            "n_hypotheses": 10,
            "n_references": 100,
            "repeated_indices": False,
            "utility": "unigram-f1",

        }

        batch_size = 32
        accumulate_grad_batches = 4
        log_dir = './logs/last_hidden_state_model_hyperparam_search'

        model_manager = self.manager(model_config)

        model = model_manager.create_model()

        train_dataset, validation_dataset = load_dataset_for_training(dataset_config, self.smoke_test)

        # Next do the preprocessing
        #
        preprocess = self.preprocess()

        train_dataset_preprocessed = preprocess(train_dataset)
        validation_dataset_preprocessed = preprocess(validation_dataset)

        # Get the collate functions

        collate_fn = self.collate(model_manager.nmt_model, model_manager.tokenizer)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=batch_size, shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=batch_size, shuffle=False, )

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

        max_epochs = 1 if self.smoke_test else 30
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[LearningRateMonitor(logging_interval="epoch"),
                       PyTorchLightningPruningCallback(trial, monitor="val_loss")],
            logger=tb_logger,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=2.0
        )

        # create the dataloaders

        trainer.logger.log_hyperparams(Namespace(**model_config))

        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        return trainer.callback_metrics["val_loss"].item()
