import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np


class BasicReferenceLstmModelV2Collator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, lookup_table, tokenizer, n_references=3, max_seq_length=75, device="cuda"):
        self.lookup_table = lookup_table
        self.tokenizer = tokenizer
        self.n_references = n_references
        self.max_seq_length = max_seq_length

        self.device = device

        self.features = [
            "mean_log_prob",
            "std_log_prob",
            "mean_entropy",
            "std_entropy"

        ]

    def get_references(self, references, counts):
        probs = np.array(counts) / np.sum(counts)

        selected_refs = np.random.choice(references, self.n_references, replace=True, p=probs)

        return selected_refs

    def __call__(self, batch):

        sources = [b["source"] for b in batch]

        ids = [b["hypothesis_id"] for b in batch]

        hypotheses_features = [self.lookup_table[id] for id in ids]
        hypotheses = [f["hypothesis"] for f in hypotheses_features]

        hypotheses_features = torch.tensor([[ref[feature] for feature in self.features] for ref in hypotheses_features])

        score = torch.tensor([b["score"] for b in batch])

        with self.tokenizer.as_target_tokenizer():
            tokenized_hypotheses = self.tokenizer(hypotheses, truncation=True,
                                                  max_length=self.max_seq_length)["input_ids"]

        # Next we need to pick the references:
        ref_ids_list = np.array([self.get_references(b["ref_ids"], b["ref_counts"]) for b in batch])

        # Next we need to transpose it
        ref_ids_list = ref_ids_list.T
        references = []
        probs_and_entropy = []
        for ref_ids in ref_ids_list:
            refs = [self.lookup_table[id] for id in ref_ids]
            references.append([ref["hypothesis"] for ref in refs])

            probs_and_entropy.append(torch.tensor([[ref[feature] for feature in self.features] for ref in refs]))


        #concatonate all the probs and entropies

        probs_and_entropy = torch.concat([hypotheses_features] + probs_and_entropy, dim=-1).to("cuda")



        # Get the references out


        packed_references = []
        with self.tokenizer.as_target_tokenizer():
            for ref in references:
                tokenized_ref = self.tokenizer(ref, truncation=True,
                                               max_length=self.max_seq_length)["input_ids"]

                packed_ref = pack_sequence([torch.tensor(h) for h in tokenized_ref], enforce_sorted=False, ).to(
                    self.device)

                packed_references.append(packed_ref)

        # Get the probs and entropies




        # Next we create a packed sequence:
        packed_hypotheses = pack_sequence([torch.tensor(h) for h in tokenized_hypotheses], enforce_sorted=False, ).to(
            self.device)

        features = {
            "tokenized_hypotheses": packed_hypotheses,
            "tokenized_references": packed_references,
            "probs_and_entropy": probs_and_entropy
        }
        return sources, hypotheses, features, score
