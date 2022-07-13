from datasets import Dataset
from torch.utils.data import DataLoader

from models.Base.BaseTrainer import BaseTrainer
from models.reference_models.CrossAttentionModel.info import CrossAttentionModelInfo
from models.reference_models.CrossAttentionModelV2.info import CrossAttentionModelV2Info
from models.reference_models.CrossAttentionModelV3.info import CrossAttentionModelV3Info
from models.reference_models.CrossAttentionModelV4.info import CrossAttentionModelV4Info

from utilities.PathManager import get_path_manager
from utilities.dataset.loading import load_dataset_for_training
from utilities.factories.PreprocessFactory import PreprocessFactory
from utilities.wrappers.NmtWrapper import NMTWrapper


class CrossAttentionModelV4Trainer(BaseTrainer):

    def __init__(self, config, smoke_test=False):
        super().__init__(config, smoke_test)
        self.config = config
        self.smoke_test = smoke_test
        self.model_info = CrossAttentionModelV4Info
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

        wrapped_nmt_model = NMTWrapper(self.model_manager.nmt_model, self.model_manager.tokenizer)
        train_preprocess_factory = PreprocessFactory(self.config["preprocess"],
                                                     {
                                                         "wrapped_nmt_model": wrapped_nmt_model,
                                                         "table_location": train_path
                                                     }
                                                     )
        train_preprocessor = train_preprocess_factory.get_preprocessor()
        val_preprocess_factory = PreprocessFactory(self.config["preprocess"],
                                                   {
                                                       "wrapped_nmt_model": wrapped_nmt_model,
                                                       "table_location": val_path
                                                   }
                                                   )
        val_preprocessor = val_preprocess_factory.get_preprocessor()

        train_dataframe, train_lookup_tables = train_preprocessor(train_dataset)


        validation_dataframe, val_lookup_tables = val_preprocessor(validation_dataset)

        train_dataset = Dataset.from_pandas(train_dataframe)
        validation_dataset = Dataset.from_pandas(validation_dataframe)

        train_collate_fn = self.model_info.collator(self.model_manager.nmt_model, self.model_manager.tokenizer,
                                                    train_lookup_tables[0],
                                                    n_references=self.config["model"]["n_references"])

        val_collate_fn = self.model_info.collator(self.model_manager.nmt_model, self.model_manager.tokenizer,
                                                  val_lookup_tables[0],
                                                  n_references=self.config["model"]["n_references"])

        train_dataloader = DataLoader(train_dataset,
                                      collate_fn=train_collate_fn,
                                      batch_size=self.config["batch_size"], shuffle=True, )
        val_dataloader = DataLoader(validation_dataset,
                                    collate_fn=val_collate_fn,
                                    batch_size=self.config["batch_size"], shuffle=False, )

        return train_dataloader, val_dataloader
