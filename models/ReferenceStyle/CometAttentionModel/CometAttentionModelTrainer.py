from datasets import Dataset
from torch.utils.data import DataLoader

from models.Base.BaseTrainer import BaseTrainer

from models.ReferenceStyle.CometAttentionModel.CometAttentionModelManager import CometAttentionModelManager


from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from utilities.factories.CollatorFactory import CollatorFactory
from utilities.factories.PreprocessFactory import PreprocessFactory
from utilities.preprocess.TablePreprocessor import TokenStatisticsLookupTableCreator, RefTableCreator


class CometAttentionModelTrainer(BaseTrainer):

    def __init__(self, config, smoke_test=False):
        super().__init__(config, smoke_test)

        self.manager_class = CometAttentionModelManager
        self.path_manager = get_path_manager()

        self.validation_ref_table = None
        self.train_ref_table = None



    def load_data(self):
        # First we load the data
        train_dataframe, validation_dataframe = load_dataset_for_training(self.config["dataset"], self.smoke_test)

        # Next we create the source tables (for looking up references)

        base = self.path_manager.get_abs_path('NMT/tatoeba-de-en/data/')
        validation_table_creator = RefTableCreator(base, 'validation_predictive' ,self.config["dataset"]["n_references"] , self.config["dataset"]["sampling_method"], self.smoke_test)
        train_table_creator = RefTableCreator(base, 'train_predictive' ,self.config["dataset"]["n_references"] , self.config["dataset"]["sampling_method"], self.smoke_test)

        self.validation_ref_table = validation_table_creator(validation_dataframe)
        self.train_ref_table = validation_table_creator(train_dataframe)

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

    def get_dataloaders(self, train_dataset, val_dataset):
        # First we get the collators
        collator_factory = CollatorFactory(self.config["collator"], self.wrapped_nmt, tables=[
            (self.train_table,self.train_ref_table),
            (self.val_table, self.validation_ref_table)
        ])
        train_collator, val_collator = collator_factory.get_collators()

        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=train_collator,
                                      batch_size=self.config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(val_dataset,
                                    collate_fn=val_collator,
                                    batch_size=self.config["batch_size"], shuffle=False, )

        return train_dataloader, val_dataloader

    def get_train_and_val_table_location(self):
        path_manager = get_path_manager()

        train_table_location = self.config["table_location"] + 'train_{}_{}/'.format(self.config["dataset"]["n_hypotheses"], self.config["dataset"]["n_references"])
        val_table_location = self.config["table_location"] + 'val_{}_{}/'.format(self.config["dataset"]["n_hypotheses"], self.config["dataset"]["n_references"])

        if self.smoke_test:
            train_table_location += "smoke_test/"
            val_table_location += "smoke_test/"

        return path_manager.get_abs_path(train_table_location), path_manager.get_abs_path(val_table_location)

    def load_tables(self):

        train_table_location, val_table_location = self.get_train_and_val_table_location()

        train_lookup_table_creator = TokenStatisticsLookupTableCreator(
            self.wrapped_nmt, train_table_location
        )
        self.train_table = train_lookup_table_creator(self.train_dataset)

        val_lookup_table_creator = TokenStatisticsLookupTableCreator(
            self.wrapped_nmt, val_table_location
        )

        self.val_table = val_lookup_table_creator(self.validation_dataset)










