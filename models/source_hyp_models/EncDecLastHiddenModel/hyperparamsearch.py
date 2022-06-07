
from torch.utils.data import DataLoader


from models.hyperparamsearch import HyperparamSearch

from models.source_hyp_models.EncDecLastHiddenModel.Collator import EncDecLastHiddenCollator
from models.source_hyp_models.EncDecLastHiddenModel.Preprocess import EncDecLastHiddenModelPreprocess
from models.source_hyp_models.EncDecLastHiddenModel.info import EncDecLastHiddenModelInfo
from utilities.dataset.loading import load_dataset_for_training


class EncDecLastHiddenModelHyperparamsearch(HyperparamSearch):

    def __init__(self, config, smoke_test):
        super().__init__(config, smoke_test)
        self.config = config
        self.smoke_test = smoke_test


        self.model_info = EncDecLastHiddenModelInfo

        self.study_name = "enc_dec_last_hidden_study"

        self.model_type = "enc_dec_last_hidden_model"

        self.log_dir = './logs/enc_dec_last_hidden_hyperparamsearch'

        self.batch_size = 32
        self.accumulate_grad_batches = 4

    def get_model_config(self, trial):
        return {
            "type": self.model_type,
            "lr": trial.suggest_float("learning_rate", 1.0e-5, 0.1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1.0e-7, 0.1, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.9, ),

            "feed_forward_layers": {
                "dims": [2048, 1024, 512, 256, 128, 1 ],
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
        train_dataset, validation_dataset = load_dataset_for_training(dataset_config, self.smoke_test)

        # Next do the preprocessing
        #
        preprocess = EncDecLastHiddenModelPreprocess()

        train_dataset_preprocessed = preprocess(train_dataset)
        validation_dataset_preprocessed = preprocess(validation_dataset)

        # Get the collate functions

        collate_fn = EncDecLastHiddenCollator(model_manager.nmt_model, model_manager.tokenizer)

        train_dataloader = DataLoader(train_dataset_preprocessed,
                                      collate_fn=collate_fn,
                                      batch_size=self.batch_size, shuffle=True, )
        val_dataloader = DataLoader(validation_dataset_preprocessed,
                                    collate_fn=collate_fn,
                                    batch_size=self.batch_size, shuffle=False, )

        return train_dataloader, val_dataloader
