from comet import download_model, load_from_checkpoint
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader


from models.reference_models.FullDecCometModel.info import FullDecCometModelInfo
from utilities.callbacks import CustomSaveCallback

from utilities.PathManager import get_path_manager


from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl



class BaseTrainer:

    def __init__(self, config, smoke_test=False):
        self.config = config
        self.smoke_test = smoke_test
        self.model_info = None

    def get_model_manager(self):
        return self.model_info.manager(self.config["model"])

    def get_model(self):
        return self.model_manager.create_model()


    def get_dataloaders(self):
        raise NotImplementedError()

    def __call__(self, ):
        # First get the model:


        self.model_manager = self.get_model_manager()

        self.model = self.get_model()


        train_dataloader, val_dataloader = self.get_dataloaders()



        # Start the training
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.config["log_dir"])
        max_epochs = 10 if self.smoke_test else self.config["max_epochs"]
        custom_save_model_callback = CustomSaveCallback(self.model_manager, self.config["save_model_path"])

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            gradient_clip_val=2.0,
            callbacks=[LearningRateMonitor(logging_interval="step"),
                       custom_save_model_callback],
            logger=tb_logger,
            accumulate_grad_batches=self.config["accumulate_grad_batches"],
        )

        # create the dataloaders
        trainer.fit(self.model, train_dataloader, val_dataloaders=val_dataloader, )

        path_manager = get_path_manager()

        model_path = path_manager.get_abs_path(self.config["save_model_path"])
        self.model_manager.save_model(model_path)

        # Reload (as a sanity check
        model, manager = self.model_info.manager.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
