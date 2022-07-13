import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq
import numpy as np
import torch.nn.functional as F
from models.Base.BaseCollator import BaseCollator, pick_random_references
from utilities.wrappers.NmtWrapper import NMTWrapper


class FullDecRefModelV4Collator(BaseCollator):
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

    def get_tokens(self, sentences):
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(sentences, truncation=True, max_length=self.max_seq_length, padding=True, return_tensors="pt",)["input_ids"]


        return labels



    def get_hypothesis_features(self, batch):
        # First get the input for the nmt model
        sources = self.get_sources(batch)
        hypotheses = self.get_hypotheses(batch)
        nmt_in, sequence_lengths = self.nmt_wrapper.prepare_for_nmt(sources, hypotheses)

        # Next we get the prob and entropies

        count_vector = self.get_tokens(hypotheses)

        hypotheses_ids = [b["hypothesis_id"] for b in batch]

        features = self.prob_entropy_lookup_table.get_features(hypotheses_ids)

        # Need to stack the entropy and
        hypotheses_prob_entropy = self.pack_entropy_and_probs(features)


        return {
            "sequence_lengths": sequence_lengths,
            "hypothesis_tokens": count_vector,
            "hypotheses_prob_entropy": hypotheses_prob_entropy,
            **nmt_in,

        }

    def get_reference_features(self, batch):

        references = self.get_references(batch)

        # Next get the features

        reference_tokens = []

        for ref in references:

            count_vec = self.get_tokens(ref.tolist())
            reference_tokens.append(count_vec)




        return {
            "references": references,
            "reference_tokens": reference_tokens
        }


    def get_references(self, batch):

        return np.array([pick_random_references(b["references"], b["reference_counts"], self.n_references) for b in batch]).T

    def pack_entropy_and_probs(self, features):
        return pack_sequence(
            [torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1)], dim=-1)
             for b in features], enforce_sorted=False)
