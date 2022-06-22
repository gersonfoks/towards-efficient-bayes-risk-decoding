from os.path import exists

from datasets import Dataset
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from models.hypothesis_only_models.LastHiddenLstmModel.manager import LastHiddenLstmManager
from models.hypothesis_only_models.ProbEntropyModel.collator import ProbEntropyModelCollator

from models.hypothesis_only_models.ProbEntropyModel.manager import ProbEntropyBaseManager
from models.hypothesis_only_models.ProbEntropyModel.preprocess import ProbEntropyModelPreprocess
from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


class ProbEntropyModelTrainer:

    def __init__(self, config, smoke_test):
        self.config = config
        self.smoke_test = smoke_test

    def __call__(self):
        config = self.config
        smoke_test = self.smoke_test
        # First get the model:

        model_manager = ProbEntropyBaseManager(config["model"])
        model = model_manager.create_model()

        # Next load the datasets
        # First check if there is a preprocessed dataset if so, load that one
        path_manager = get_path_manager()

        dataset_config = config["dataset"]
        name = "{}unigram_f1_{}_{}".format(dataset_config["preproces_dir"], dataset_config["n_hypotheses"],
                                           dataset_config["n_references"])
        if smoke_test:
            name += "_smoke_test_"

        preprocessed_train_dataset_ref = path_manager.get_abs_path(name + "train.parquet")
        preprocessed_val_dataset_ref = path_manager.get_abs_path(name + "val.parquet")
        if exists(preprocessed_train_dataset_ref) and exists(preprocessed_val_dataset_ref):
            print("Loading preprocessed data")
            train_dataset_preprocessed = Dataset.from_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed = Dataset.from_parquet(preprocessed_val_dataset_ref)



        else:
            train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

            # Next do the preprocessing
            #
            preprocess = ProbEntropyModelPreprocess(model_manager.nmt_model, model_manager.tokenizer)

            train_dataset_preprocessed = preprocess(train_dataset)
            validation_dataset_preprocessed = preprocess(validation_dataset)

            train_dataset_preprocessed.to_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed.to_parquet(preprocessed_val_dataset_ref)

            # Save in parquet format.

        # Get the collate functions

        collate_fn = ProbEntropyModelCollator()

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
            gradient_clip_val=2.0,
            log_every_n_steps=250,  # Makes interpreting the training easier (as we see a running average)
            flush_logs_every_n_steps=500
        )

        # create the dataloaders
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(config["save_model_path"])
        model_manager.save_model(model_path)

        model, manager = ProbEntropyBaseManager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
