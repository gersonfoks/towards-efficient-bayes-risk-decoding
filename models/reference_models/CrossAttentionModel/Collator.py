import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np

from models.Base.BaseCollator import BaseCollator, pick_random_references


class CrossAttentionModelCollator(BaseCollator):
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, prob_entropy_lookup_table, max_seq_length=75, device="cuda",
                 n_references=3):
        super().__init__(nmt_model, tokenizer, max_seq_length, device)
        self.device = device
        self.n_references = n_references
        self.prob_entropy_lookup_table = prob_entropy_lookup_table

    def get_references(self, batch):
        return np.array(
            [pick_random_references(b["references"], b["reference_counts"], self.n_references) for b in batch]).T

    def get_features(self, batch):
        # First we get the features for the hypothesis
        hypothesis_features = self.get_hypothesis_features(batch)
        # Then we get those for the references
        reference_features = self.get_reference_features(batch)

        return {**hypothesis_features, **reference_features}

    def get_tokens(self, sentences):
        with self.tokenizer.as_target_tokenizer():
            s_tokenized = self.tokenizer(sentences, truncation=True, max_length=self.max_seq_length, padding=True,
                                    return_tensors="pt", )

        return s_tokenized

    def get_hypothesis_features(self, batch):
        # First get the input for the nmt model
        hypotheses = self.get_hypotheses(batch)


        h_tokenized = self.get_tokens(hypotheses)


        hypotheses_ids = [b["hypothesis_id"] for b in batch]

        features = self.prob_entropy_lookup_table.get_features(hypotheses_ids)

        # Need to stack the entropy and
        hypotheses_prob_entropy = self.pad_prob_and_entropy(features)

        return {

            "tokenized_hypotheses": h_tokenized,
            "hypotheses_prob_entropy": hypotheses_prob_entropy,

        }

    def get_reference_features(self, batch):
        reference_ids = self.get_reference_ids(batch)

        # Next get the features

        reference_tokens = []

        reference_features = []

        for ref in reference_ids:
            features = self.prob_entropy_lookup_table.get_features(ref)

            references = [f["hypothesis"] for f in features]
            r_tokenized = self.get_tokens(references)
            reference_tokens.append(r_tokenized)

            packed_hyp_prob_ref = self.pad_prob_and_entropy(features)
            reference_features.append(packed_hyp_prob_ref)

        return {
            "references_tokenized": reference_tokens,
            "reference_prob_entropy": reference_features
        }


    def pad_prob_and_entropy(self, features):
        return pad_sequence(
            [torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1)], dim=-1)
             for b in features], batch_first=True)

    def get_reference_ids(self, batch):

        return np.array([pick_random_references(b["reference_ids"], b["reference_counts"], self.n_references) for b in batch]).T


