from os.path import exists


from datasets import Dataset

from torch.utils.data import DataLoader


from models.hyperparamsearch import HyperparamSearch
from models.hypothesis_only_models.FullDecHiddenLstmModel.info import FullDecHiddenLstmInfo

from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training



class FullDecHyperparamSearch(HyperparamSearch):

    def __init__(self, config, smoke_test):
        super().__init__(config, smoke_test)
        self.config = config
        self.smoke_test = smoke_test

        self.model_info = FullDecHiddenLstmInfo

        self.model_type = "full_dec_model"

        self.study_name = "full_dec_study"

        self.log_dir = "./logs/full_dec_model_hyperparamsearch"


    def get_model_config(self, trial):
        return {
            "type": self.model_type,
            "lr": trial.suggest_float("learning_rate", 1.0e-5, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-7, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.9, ),

            "feed_forward_layers": {
                "dims": [2048, 1024, 512, 256, 129, 1 ],
                "activation_function": "relu",
                "activation_function_last_layer": "sigmoid",

            },

            "optimizer": {
                "type": "adam_with_steps",
                "step_size": 1,
                "gamma": trial.suggest_float("learning_rate_decay", 0.25, 1.0, )
            },

            "nmt_model": {
                "model": {
                    "name": 'Helsinki-NLP/opus-mt-de-en',
                    "checkpoint": 'NMT/tatoeba-de-en/model',
                    "type": 'MarianMT'
                }
            }

        }

    def load_dataset(self, trial, model, model_manager):

        dataset_config = self.get_dataset_config()
        name = "{}unigram_f1_{}_{}".format(dataset_config["preproces_dir"], dataset_config["n_hypotheses"],
                                           dataset_config["n_references"])
        if self.smoke_test:
            name += "_smoke_test_"

        path_manager = get_path_manager()
        preprocessed_train_dataset_ref = path_manager.get_abs_path(name + "train.parquet")
        preprocessed_val_dataset_ref = path_manager.get_abs_path(name + "val.parquet")
        if exists(preprocessed_train_dataset_ref) and exists(preprocessed_val_dataset_ref):
            print("Loading preprocessed data")
            train_dataset_preprocessed = Dataset.from_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed = Dataset.from_parquet(preprocessed_val_dataset_ref)



        else:
            train_dataset, validation_dataset = load_dataset_for_training(dataset_config, self.smoke_test)

            # Next do the preprocessing
            #
            preprocess = self.model_info.preprocess(model_manager.nmt_model, model_manager.tokenizer)

            train_dataset_preprocessed = preprocess(train_dataset)
            validation_dataset_preprocessed = preprocess(validation_dataset)

            train_dataset_preprocessed.to_parquet(preprocessed_train_dataset_ref)
            validation_dataset_preprocessed.to_parquet(preprocessed_val_dataset_ref)

            # Save in parquet format.

        # Get the collate functions

        collate_fn = self.model_info.collate(model_manager.nmt_model, model_manager.tokenizer)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=self.batch_size, shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=self.batch_size, shuffle=False, )


        return train_dataloader, val_dataloader




