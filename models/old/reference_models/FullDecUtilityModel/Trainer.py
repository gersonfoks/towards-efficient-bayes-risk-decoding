from os.path import exists

from datasets import Dataset
from torch.utils.data import DataLoader

from models.Base.BaseTrainer import BaseTrainer
from models.old.reference_models.FullDecUtilityModel.info import FullDecUtilityModelInfo
from utilities.PathManager import get_path_manager

from utilities.dataset.loading import load_dataset_for_training

class FullDecUtilityModelTrainer(BaseTrainer):

    def __init__(self, config, smoke_test):
        super().__init__(config, smoke_test)
        self.model_info = FullDecUtilityModelInfo


    def get_dataloaders(self):
        path_manager = get_path_manager()

        dataset_config = self.config["dataset"]
        name = "{}unigram_f1_full_dec_util_{}_{}".format(dataset_config["preproces_dir"],
                                                         dataset_config["n_hypotheses"],
                                                         dataset_config["n_references"])
        if self.smoke_test:
            name += "_smoke_test_"

        preprocessed_train_dataset_ref = path_manager.get_abs_path(name + "train.parquet")
        preprocessed_val_dataset_ref = path_manager.get_abs_path(name + "val.parquet")
        if exists(preprocessed_train_dataset_ref) and exists(preprocessed_val_dataset_ref):
            print("Loading preprocessed data")
            train_dataset_preprocessed = Dataset.from_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed = Dataset.from_parquet(preprocessed_val_dataset_ref)

        else:
            print("preprocessing")
            train_dataset, validation_dataset = load_dataset_for_training(self.config["dataset"], self.smoke_test)

            # Next do the preprocessing
            #
            preprocess = self.model_info.preprocess(self.model_manager.nmt_model, self.model_manager.tokenizer)

            train_dataset_preprocessed = preprocess(train_dataset)
            validation_dataset_preprocessed = preprocess(validation_dataset)

            # Save in parquet format.
            train_dataset_preprocessed.to_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed.to_parquet(preprocessed_val_dataset_ref)

        # Get the collate functions

        collate_fn = self.model_info.collate(self.model_manager.nmt_model, self.model_manager.tokenizer,
                                       n_references=self.config["model"]["n_references"])

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=self.config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=self.config["batch_size"], shuffle=False, )

        return train_dataloader, val_dataloader


