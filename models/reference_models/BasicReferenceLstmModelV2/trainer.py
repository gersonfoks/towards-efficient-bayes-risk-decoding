from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from models.reference_models.BasicReferenceLstmModel.info import BasicReferenceLstmModelInfo
from models.reference_models.BasicReferenceLstmModelV2.info import BasicReferenceLstmModelV2Info
from utilities.callbacks import CustomSaveCallback

from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


class BasicReferenceLstmModelV2Trainer:

    def __init__(self, config, smoke_test=False):
        self.config = config
        self.smoke_test = smoke_test
        self.model_info = BasicReferenceLstmModelV2Info

    def __call__(self,):
        # First get the model:

        config = self.config
        smoke_test = self.smoke_test
        model_manager = self.model_info.manager(config["model"])

        model = model_manager.create_model()

        # Next load the datasets

        train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

        # Next do the preprocessing
        #


        lookup_table_base_location = "predictive/tatoeba-de-en/data/lookup_tables/basic_ref_model_v2"

        if self.smoke_test:
            lookup_table_base_location += "_smoke_test"
        lookup_table_val_location = lookup_table_base_location + "_val/"
        lookup_table_train_location = lookup_table_base_location + "_train/"

        train_preprocess = self.model_info.preprocess(model_manager.nmt_model, model_manager.tokenizer, lookup_table_location=lookup_table_train_location)
        val_preprocess = self.model_info.preprocess(model_manager.nmt_model, model_manager.tokenizer, lookup_table_location=lookup_table_val_location)
        train_dataset_preprocessed, lookup_table_train = train_preprocess(train_dataset, )
        validation_dataset_preprocessed, lookup_table_val = val_preprocess(validation_dataset,)

        # Get the collate functions

        train_collate_fn = self.model_info.collator(lookup_table_train, model_manager.tokenizer, n_references=3)
        val_collate_fn = self.model_info.collator(lookup_table_val, model_manager.tokenizer, n_references=3)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=train_collate_fn,
                                      batch_size=config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
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
                       EarlyStopping(monitor="val_loss", divergence_threshold=0.2, patience=5),
                       custom_save_model_callback],
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

        model, manager = self.model_info.manager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
