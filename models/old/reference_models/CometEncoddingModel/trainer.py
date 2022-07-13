from comet import download_model, load_from_checkpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from models.old.reference_models.CometEncoddingModel.info import CometEncodingModelInfo
from utilities.callbacks import CustomSaveCallback

from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utilities.wrappers.CometWrapper import CometWrapper


class CometEncodingModelTrainer:

    def __init__(self, config, smoke_test=False):
        self.config = config
        self.smoke_test = smoke_test
        self.model_info = CometEncodingModelInfo

    def __call__(self, ):
        # First get the model:

        config = self.config
        smoke_test = self.smoke_test
        model_manager = self.model_info.manager(config["model"])

        model = model_manager.create_model()

        # Next load the datasets

        train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

        # Next do the preprocessing

        # Load comet
        model_path = download_model("wmt20-comet-da")
        comet_model = load_from_checkpoint(model_path)

        comet_model.to("cuda")

        wrapped_model = CometWrapper(comet_model)

        comet_model.eval()

        train_path = "predictive/tatoeba-de-en/data/lookup_tables/comet_model_refs_train/"
        val_path = "predictive/tatoeba-de-en/data/lookup_tables/comet_model_refs_val/"

        train_preprocess = self.model_info.preprocess(wrapped_model, lookup_table_location=train_path)
        val_preprocess = self.model_info.preprocess(wrapped_model, lookup_table_location=val_path)

        train_dataset_preprocessed, train_ref_lookup_table, train_source_lookup_table = train_preprocess(train_dataset)
        val_dataset_preprocessed, val_ref_lookup_table, val_source_lookup_table = val_preprocess(validation_dataset)

        # Get the collate functions

        train_collate_fn = self.model_info.collator(train_source_lookup_table, train_ref_lookup_table, n_references=3)
        val_collate_fn = self.model_info.collator(val_source_lookup_table, val_ref_lookup_table, n_references=3)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=train_collate_fn,
                                      batch_size=config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(val_dataset_preprocessed,
                                    collate_fn=val_collate_fn,
                                    batch_size=config["batch_size"], shuffle=False, )

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])
        max_epochs = 10 if smoke_test else config["max_epochs"]
        custom_save_model_callback = CustomSaveCallback(model_manager, config["save_model_path"])

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            callbacks=[LearningRateMonitor(logging_interval="step"),
                       custom_save_model_callback],
            logger=tb_logger,
            accumulate_grad_batches=config["accumulate_grad_batches"],
        )

        # create the dataloaders
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(config["save_model_path"])
        model_manager.save_model(model_path)

        model, manager = self.model_info.manager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
