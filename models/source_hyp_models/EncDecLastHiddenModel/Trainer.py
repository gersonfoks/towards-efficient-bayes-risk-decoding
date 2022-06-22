from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from models.hypothesis_only_models.LastHiddenLstmModel.Collator import LastHiddenLstmCollator
from models.hypothesis_only_models.LastHiddenLstmModel.Preprocess import LastHiddenLstmPreprocess
from models.hypothesis_only_models.LastHiddenLstmModel.manager import LastHiddenLstmManager
from models.source_hyp_models.EncDecLastHiddenModel.Collator import EncDecLastHiddenCollator
from models.source_hyp_models.EncDecLastHiddenModel.Preprocess import EncDecLastHiddenModelPreprocess
from models.source_hyp_models.EncDecLastHiddenModel.manager import EncDecLastHiddenBaseManager
from utilities.PathManager import get_path_manager
from utilities.callbacks import CustomSaveCallback
from utilities.dataset.loading import load_dataset_for_training
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

class EncDecLastHidenModelTrainer:

    def __init__(self, config, smoke_test):
        self.config = config
        self.smoke_test = smoke_test

    def __call__(self):
        config = self.config
        smoke_test = self.smoke_test
        # First get the model:

        model_manager = EncDecLastHiddenBaseManager(config["model"])

        model = model_manager.create_model()

        # Next load the datasets

        train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

        # Next do the preprocessing
        #
        preprocess = EncDecLastHiddenModelPreprocess()

        train_dataset_preprocessed = preprocess(train_dataset)
        validation_dataset_preprocessed = preprocess(validation_dataset)

        # Get the collate functions

        collate_fn = EncDecLastHiddenCollator(model_manager.nmt_model, model_manager.tokenizer)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=config["batch_size"], shuffle=False, )

        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])

        max_epochs = 5 if smoke_test else config["max_epochs"]

        custom_save_model_callback = CustomSaveCallback(model_manager, config["save_model_path"])

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            val_check_interval=1.0,
            callbacks=[LearningRateMonitor(logging_interval="step"), EarlyStopping(monitor="val_loss", divergence_threshold=0.2, patience=5 ), custom_save_model_callback],
            logger=tb_logger,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=2.0,
            log_every_n_steps=250, # Makes interpreting the training easier (as we see a running average)
            flush_logs_every_n_steps=500
        )


        # create the dataloaders
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(config["save_model_path"])
        model_manager.save_model(model_path)

        model, manager = EncDecLastHiddenBaseManager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
