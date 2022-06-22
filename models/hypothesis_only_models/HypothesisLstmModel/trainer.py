from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader

from models.hypothesis_only_models.HypothesisLstmModel.Collator import HypothesisLstmModelCollator
from models.hypothesis_only_models.HypothesisLstmModel.Preprocess import HypothesisLstmPreprocess
from models.hypothesis_only_models.HypothesisLstmModel.manager import HypothesisLstmBaseManager
from utilities.dataset.loading import load_dataset_for_training
from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl


class HypothesisLstmModelTrainer:

    def __init__(self, config, smoke_test=False):
        self.config = config
        self.smoke_test = smoke_test

    def __call__(self, config, smoke_test):
        # First get the model:

        model_manager = HypothesisLstmBaseManager(config["model"])

        model = model_manager.create_model()

        # Next load the datasets

        train_dataset, validation_dataset = load_dataset_for_training(config["dataset"], smoke_test)

        # Next do the preprocessing
        #
        preprocess = HypothesisLstmPreprocess()

        train_dataset_preprocessed = preprocess(train_dataset)
        validation_dataset_preprocessed = preprocess(validation_dataset)

        # Get the collate functions

        collate_fn = HypothesisLstmModelCollator(model_manager.tokenizer)

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
            callbacks=[LearningRateMonitor(logging_interval="epoch")],
            logger=tb_logger,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            gradient_clip_val=2.0
        )

        # create the dataloaders
        trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(config["save_model_path"])
        model_manager.save_model(model_path)

        model, manager = HypothesisLstmBaseManager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
