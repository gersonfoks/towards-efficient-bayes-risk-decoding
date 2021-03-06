import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np

from models.Base.BaseCollator import BaseCollator, pick_random_references


class TopProbModelV2Collator(BaseCollator):
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
        return self.get_hypothesis_features(batch)

    def get_tokens(self, sentences):
        with self.tokenizer.as_target_tokenizer():
            s_tokenized = self.tokenizer(sentences, truncation=True, max_length=self.max_seq_length, padding=True,
                                         return_tensors="pt", )

        return s_tokenized

    def get_hypothesis_features(self, batch):
        # First get the input for the nmt model

        hypotheses_ids = [b["hypothesis_id"] for b in batch]

        features = self.prob_entropy_lookup_table.get_features(hypotheses_ids)

        # Need to stack the entropy and
        top_k_probs, h_probs = self.pack_features(features)

        return {

            "top_k_probs": top_k_probs,
            "h_probs": h_probs,

        }

    def pack_features(self, features):
        top_k_probs = [torch.concat([torch.tensor(np.stack(f["top_k_prob"])).float()], dim=-1) for f in features]
        h_probs = [torch.concat([torch.tensor(f["prob"].reshape(-1, 1)).float()], dim=-1) for f in features]

        return pack_sequence(top_k_probs, enforce_sorted=False, ), pack_sequence(h_probs, enforce_sorted=False, )

