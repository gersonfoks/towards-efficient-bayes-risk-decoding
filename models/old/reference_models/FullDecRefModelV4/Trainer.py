
from torch.utils.data import DataLoader

from models.Base.BaseTrainer import BaseTrainer
from models.old.reference_models.FullDecRefModelV4.info import FullDecRefModelV4Info

from utilities.PathManager import get_path_manager

from utilities.dataset.loading import load_dataset_for_training


class FullDecRefModelV4Trainer(BaseTrainer):

    def __init__(self, config, smoke_test):
        super().__init__(config, smoke_test)
        self.model_info = FullDecRefModelV4Info

        self.path_manager = get_path_manager()

    def get_dataloaders(self):


        train_dataset, validation_dataset = load_dataset_for_training(self.config["dataset"], self.smoke_test)

        base_path = './predictive/tatoeba-de-en/data/lookup_tables/{}_{}_prob_entropy_refs'.format(
            self.config['dataset']["n_hypotheses"], self.config['dataset']["n_hypotheses"])
        if self.smoke_test:
            base_path += "_smoke_test"
        train_path = base_path + '_train/'
        val_path = base_path + '_val/'
        train_path = self.path_manager.get_abs_path(train_path)
        val_path = self.path_manager.get_abs_path(val_path)

        train_preprocess = self.model_info.preprocess(self.model_manager.nmt_model, self.model_manager.tokenizer, table_location=train_path)
        val_preprocess = self.model_info.preprocess(self.model_manager.nmt_model, self.model_manager.tokenizer, table_location=val_path)

        train_dataset_preprocessed, train_lookup_table = train_preprocess(train_dataset)
        validation_dataset_preprocessed, val_lookup_table = val_preprocess(validation_dataset)

        train_collate_fn = self.model_info.collate(self.model_manager.nmt_model, self.model_manager.tokenizer, train_lookup_table,
                                             n_references=self.config["model"]["n_references"])

        val_collate_fn = self.model_info.collate(self.model_manager.nmt_model, self.model_manager.tokenizer, val_lookup_table,
                                                   n_references=self.config["model"]["n_references"])

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=train_collate_fn,
                                      batch_size=self.config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=val_collate_fn,
                                    batch_size=self.config["batch_size"], shuffle=False, )

        return train_dataloader, val_dataloader
