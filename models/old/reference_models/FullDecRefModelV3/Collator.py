import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq
import numpy as np

from models.Base.BaseCollator import BaseCollator, pick_random_references
from utilities.wrappers.NmtWrapper import NMTWrapper


class FullDecRefModelV2Collator(BaseCollator):
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, prob_entropy_lookup_table, max_seq_length=75, device="cuda",
                 n_references=3):
        super().__init__(nmt_model, tokenizer, max_seq_length, device)
        self.device = device
        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt", )

        self.n_references = n_references

        self.prob_entropy_lookup_table = prob_entropy_lookup_table
        self.nmt_wrapper = NMTWrapper(nmt_model, tokenizer)

    def get_features(self, batch):
        # First we get the features for the hypothesis
        hypothesis_features = self.get_hypothesis_features(batch)
        # Then we get those for the references
        reference_features = self.get_reference_features(batch)

        return {**hypothesis_features, **reference_features}

    def pack_entropy_and_probs(self, features):
        return pack_sequence(
            [torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1)], dim=-1)
             for b in features], enforce_sorted=False)

    def get_hypothesis_features(self, batch):
        # First get the input for the nmt model
        sources = self.get_sources(batch)
        hypotheses = self.get_hypotheses(batch)
        nmt_in, sequence_lengths = self.nmt_wrapper.prepare_for_nmt(sources, hypotheses)

        # Next we get the prob and entropies

        hypotheses_ids = [b["hypothesis_id"] for b in batch]

        features = self.prob_entropy_lookup_table.get_features(hypotheses_ids)

        # Need to stack the entropy and
        hypotheses_prob_entropy = self.pack_entropy_and_probs(features)

        return {
            "sequence_lengths": sequence_lengths,
            "hypotheses_prob_entropy": hypotheses_prob_entropy,
            **nmt_in,

        }

    def get_reference_features(self, batch):

        reference_ids = self.get_reference_ids(batch)

        # Next get the features

        reference_features = []
        references = []
        for ref in reference_ids:
            features = self.prob_entropy_lookup_table.get_features(ref)
            references.append([f["hypothesis"] for f in features])

            packed_hyp_prob_ref = self.pack_entropy_and_probs(features)
            reference_features.append(packed_hyp_prob_ref)
        return {
            "references": references,
            "references_prob_entropy": reference_features
        }


    def get_reference_ids(self, batch):

        return np.array([pick_random_references(b["reference_ids"], b["reference_counts"], self.n_references) for b in batch]).T




    # def __call__(self, batch):
    #
    #     hypotheses = [b["hypotheses"] for b in batch]
    #     sources = [b["source"] for b in batch]
    #
    #     score = torch.tensor([b["score"] for b in batch])
    #
    #     # We get the lengths of the sequences (used for packing later)
    #     with self.tokenizer.as_target_tokenizer():
    #         labels = self.tokenizer(hypotheses, truncation=True, max_length=self.max_seq_length)
    #         sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])
    #
    #     model_inputs = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length)
    #     # Setup the tokenizer for targets
    #
    #     model_inputs["labels"] = labels["input_ids"]
    #
    #     x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
    #          zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]
    #
    #     # Next we create the inputs for the NMT model
    #     x_new = self.data_collator(x).to("cuda")
    #
    #     log_prob_entropy = pack_sequence(
    #         [torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1)], dim=-1)
    #          for b in batch], enforce_sorted=False)
    #
    #     references = [self.get_references(b["references"], b["reference_counts"]).tolist() for b in batch]
    #
    #     features = {
    #         "references": references,
    #         "sequence_lengths": sequence_lengths,
    #         "log_prob_entropy": log_prob_entropy,
    #         **x_new,
    #
    #     }
    #     return sources, hypotheses, features, score
