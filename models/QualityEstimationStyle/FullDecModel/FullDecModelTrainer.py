from torch.utils.data import DataLoader

from models.Base.BaseTrainer import BaseTrainer
from models.QualityEstimationStyle.FullDecModel.FullDecModelManager import FullDecModelManager

from utilities.PathManager import get_path_manager
from utilities.factories.CollatorFactory import CollatorFactory
from utilities.preprocess.TablePreprocessor import TokenStatisticsLookupTableCreator


class FullDecModelTrainer(BaseTrainer):

    def __init__(self, config, smoke_test=False):
        super().__init__(config, smoke_test)

        self.manager_class = FullDecModelManager

    def get_dataloaders(self, train_dataset, val_dataset):
        # First we get the collators
        collator_factory = CollatorFactory(self.config["collator"], self.wrapped_nmt, tables=[
            self.train_table,
            self.val_table
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
            self.wrapped_nmt, train_table_location, load=False
        )
        self.train_table = train_lookup_table_creator(self.train_dataset)

        val_lookup_table_creator = TokenStatisticsLookupTableCreator(
            self.wrapped_nmt, val_table_location, load=False
        )

        self.val_table = val_lookup_table_creator(self.validation_dataset)






