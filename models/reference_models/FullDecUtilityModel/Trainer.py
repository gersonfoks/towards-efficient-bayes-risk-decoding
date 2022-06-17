from os.path import exists

from datasets import Dataset
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from models.reference_models.FullDecUtilityModel.info import FullDecUtilityModelInfo
from utilities.PathManager import get_path_manager
from utilities.callbacks import CustomSaveCallback
from utilities.dataset.loading import load_dataset_for_training
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

class FullDecUtilityModelTrainer:

    def __init__(self, config, smoke_test):
        self.config = config
        self.smoke_test = smoke_test

        self.info = FullDecUtilityModelInfo

    def __call__(self):
        config = self.config
        smoke_test = self.smoke_test
        # First get the model:

        model_manager = self.info.manager(config["model"])

        model = model_manager.create_model()

        # Next load the datasets

        path_manager = get_path_manager()

        dataset_config = config["dataset"]
        name = "{}unigram_f1_full_dec_util_{}_{}".format(dataset_config["preproces_dir"], dataset_config["n_hypotheses"],
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
            preprocess = self.info.preprocess(model_manager.nmt_model, model_manager.tokenizer)

            train_dataset_preprocessed = preprocess(train_dataset)
            validation_dataset_preprocessed = preprocess(validation_dataset)

            # Save in parquet format.
            train_dataset_preprocessed.to_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed.to_parquet(preprocessed_val_dataset_ref)



        # Get the collate functions

        collate_fn = self.info.collate(model_manager.nmt_model, model_manager.tokenizer)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=config["batch_size"], shuffle=False, )

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])

        max_epochs = 10 if smoke_test else config["max_epochs"]
        custom_save_model_callback = CustomSaveCallback(model_manager, config["save_model_path"])

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[LearningRateMonitor(logging_interval="step"), custom_save_model_callback],
            logger=tb_logger,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=2.0,
        )



        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(config["save_model_path"])
        model_manager.save_model(model_path)

        model, manager = self.info.manager.load_model(model_path)

        trainer.validate(model, val_dataloader, )
