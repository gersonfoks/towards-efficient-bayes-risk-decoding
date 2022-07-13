import torch
from datasets import Dataset

from models.old.reference_models.BasicReferenceLstmModelV2.Preprocess import create_hypothesis_ids
from utilities.LookUpTable import LookUpTable
from utilities.PathManager import get_path_manager
from utilities.wrappers.CometWrapper import CometWrapper
from utilities.wrappers.NmtWrapper import NMTWrapper


class FullDecCometModelPreprocess:

    def __init__(self, nmt_wrapper: NMTWrapper, comet_model: CometWrapper, max_seq_length=75,
                 lookup_table_location='predictive/tatoeba-de-en/data/lookup_tables/comet_model_refs/'):

        self.nmt_wrapper = nmt_wrapper
        self.max_seq_length = max_seq_length

        self.batch_size = 32

        self.features = [
            "mean_log_prob",
            "std_log_prob",
            "mean_entropy",
            "std_entropy"
        ]

        self.comet_model = comet_model

        self.lookup_table_location = lookup_table_location
        self.path_manager = get_path_manager()

    def __call__(self, data):
        # Add list of references
        data = data.reset_index()

        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(create_hypothesis_ids, batched=True, batch_size=32).to_pandas()

        dataset["ref_ids"] = dataset["hypotheses_ids"]
        dataset["ref_counts"] = dataset["count"]

        dataset["source_index"] = data.index
        source_dataset = Dataset.from_pandas(dataset[["source_index", "source"]])

        # Explode the dataset
        dataset = dataset.explode(["hypotheses", "utilities", "count", "hypotheses_ids"], ignore_index=True).rename(
            columns={
                "hypotheses": "hypothesis",
                "utilities": "utility",
                "hypotheses_ids": "hypothesis_id"
            })

        main_dataset = dataset[
            ["source_index", "source", "hypothesis_id", "hypothesis", "ref_ids", "ref_counts", "utility"]]
        # First we need to add the probabilities and entropy to the hypothesis

        hyp_dataset = Dataset.from_pandas(dataset[["hypothesis_id", "hypothesis", "utility", "source"]])

        hypothesis_lookup_table = self.create_hypothesis_lookup_table(hyp_dataset)
        source_lookup_table = self.create_source_lookup_table(source_dataset)

        # Add probs and entropy
        main_dataset = Dataset.from_pandas(main_dataset)
        main_dataset = main_dataset.map(self.nmt_wrapper.map_to_log_probs_and_entropy, batch_size=self.batch_size,
                                        batched=True).to_pandas()

        main_dataset["score"] = main_dataset["utility"]

        main_dataset = Dataset.from_pandas(main_dataset)

        return main_dataset, hypothesis_lookup_table, source_lookup_table

    @torch.no_grad()
    def h_to_comet_embedding(self, x):
        embedding = self.comet_model.to_embedding(x["hypothesis"])

        return {
            "embedding": embedding.cpu().numpy(),
            **x
        }

    @torch.no_grad()
    def s_to_comet_embedding(self, x):
        embedding = self.comet_model.to_embedding(x["source"])
        return {
            "embedding": embedding.cpu().numpy(),
            **x
        }

    def create_hypothesis_lookup_table(self, hyp_dataset):
        hypothesis_lookup_table_location = self.lookup_table_location + 'hypothesis/'

        hypothesis_lookup_table_location = self.path_manager.get_abs_path(hypothesis_lookup_table_location)

        if LookUpTable.exists(hypothesis_lookup_table_location):
            hypothesis_look_up_table = LookUpTable.load(hypothesis_lookup_table_location)
        else:
            hyp_dataset = hyp_dataset.map(self.h_to_comet_embedding, batch_size=self.batch_size,
                                          batched=True).to_pandas()
            hypothesis_look_up_table = LookUpTable(hyp_dataset, index="hypothesis_id",
                                                   features=["hypothesis", "embedding"])

            hypothesis_look_up_table.save(hypothesis_lookup_table_location)
        return hypothesis_look_up_table

    def create_source_lookup_table(self, source_dataset):
        source_lookup_table_location = self.lookup_table_location + 'source/'

        source_lookup_table_location = self.path_manager.get_abs_path(source_lookup_table_location)

        if LookUpTable.exists(source_lookup_table_location):
            source_lookup_table = LookUpTable.load(source_lookup_table_location)
        else:
            source_dataset = source_dataset.map(self.s_to_comet_embedding, batch_size=self.batch_size,
                                                batched=True).to_pandas()
            source_lookup_table = LookUpTable(source_dataset, index="source_index",
                                              features=["embedding", "source"])

            source_lookup_table.save(source_lookup_table_location)
        return source_lookup_table

    # def create_prob_entropy_lookup_table(self, hyp_dataset):
    #     location = self.lookup_table_location + 'prob_and_entropy/'
    #     look_up_table_location = self.path_manager.get_abs_path(location)
    #
    #     if LookUpTable.exists(look_up_table_location):
    #         look_up_table = LookUpTable.load(look_up_table_location)
    #     else:
    #         hyp_dataset = hyp_dataset.map(self.nmt_wrapper.map_to_log_probs_and_entropy, batch_size=self.batch_size,
    #                                       batched=True).to_pandas()
    #         look_up_table = LookUpTable(hyp_dataset, index="hypothesis_id",
    #                                     features=["hypothesis", "utility", ] + self.features)
    #
    #         look_up_table.save(look_up_table_location)
    #     return look_up_table
