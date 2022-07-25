from datasets import Dataset
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader

from utilities.callbacks import CustomSaveCallback

from utilities.PathManager import get_path_manager

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utilities.dataset.loading import load_dataset_for_training
from utilities.factories.CollatorFactory import CollatorFactory
from utilities.factories.PreprocessFactory import PreprocessFactory
from utilities.misc import load_nmt_model
from utilities.wrappers.NmtWrapper import NMTWrapper


class BaseTrainer:

    def __init__(self, config, smoke_test=False):
        self.config = config
        self.smoke_test = smoke_test
        self.manager_class = None
        self.model = None

        self.train_dataframe = None
        self.validation_dataframe = None

    def load_common(self):
        self.nmt_model, self.tokenizer = load_nmt_model(self.config["model"]["nmt_model"], pretrained=True)
        self.wrapped_nmt = NMTWrapper(self.nmt_model, self.tokenizer)

    def load_manager_and_model(self):
        self.model_manager = self.manager_class(self.config["model"])
        self.model = self.model_manager.create_model()
        return   self.model_manager,self.model,

    def load_data(self):
        # First we load the data
        train_dataframe, validation_dataframe = load_dataset_for_training(self.config["dataset"], self.smoke_test)

        # Next we do preprocessing
        preprocess_factory = PreprocessFactory(self.config["preprocess"])

        preprocessor = preprocess_factory.get_preprocessor()
        train_dataframe = preprocessor(train_dataframe)
        validation_dataframe = preprocessor(validation_dataframe)

        train_dataframe = train_dataframe
        validation_dataframe = validation_dataframe

        # Lastly we put it into a dataset
        self.train_dataset = Dataset.from_pandas(train_dataframe)
        self.validation_dataset = Dataset.from_pandas(validation_dataframe)

        return self.train_dataset, self.validation_dataset

    def load_tables(self):
        pass

    def get_dataloaders(self, train_dataset, val_dataset):
        # First we get the collators
        collator_factory = CollatorFactory(self.config["collator"], self.wrapped_nmt)
        train_collator, val_collator = collator_factory.get_collators()

        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=train_collator,
                                      batch_size=self.config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(val_dataset,
                                    collate_fn=val_collator,
                                    batch_size=self.config["batch_size"], shuffle=False, )

        return train_dataloader, val_dataloader

    def __call__(self, ):
        # First get the model:

        # Load a common things we might need
        print("start loading common objects")
        self.load_common()

        # Load the data
        print("start loading the dataset")
        train_dataset, val_dataset = self.load_data()



        # Load the manager and the model
        print("start loading the model")
        model_manager, model = self.load_manager_and_model()

        # Next load the tables (if needed, for models that use tables)
        self.load_tables()

        # Get the dataloaders
        print("Getting the dataloaders")
        train_dataloader, val_dataloader = self.get_dataloaders(train_dataset, val_dataset)

        # Start the training
        print("Starting the training")
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=self.config["log_dir"])
        max_epochs = 10 if self.smoke_test else self.config["max_epochs"]
        custom_save_model_callback = CustomSaveCallback(model_manager, self.config["save_model_path"])

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=1,
            progress_bar_refresh_rate=1,
            gradient_clip_val=self.config["gradient_clip_val"],
            callbacks=[LearningRateMonitor(logging_interval="step"),
                       EarlyStopping("val_loss", patience=10, verbose=True,),
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
        model, manager = self.manager_class.load_model(model_path)

        # create the dataloaders
        trainer.validate(model, val_dataloader, )
