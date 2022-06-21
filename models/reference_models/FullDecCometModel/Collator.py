import torch

import numpy as np
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq


class FullDecCometModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, source_lookup_table, reference_lookup_table, n_references=3,
                 max_seq_length=75, device="cuda"):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.reference_lookup_table = reference_lookup_table
        self.source_lookup_table = source_lookup_table

        self.n_references = n_references
        self.max_seq_length = max_seq_length

        self.device = device

        self.features = [
            "mean_log_prob",
            "std_log_prob",
            "mean_entropy",
            "std_entropy"

        ]

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt", )

    def get_references(self, references, counts):
        probs = np.array(counts) / np.sum(counts)

        selected_refs = np.random.choice(references, self.n_references, replace=True, p=probs)

        return selected_refs

    def __call__(self, batch):
        hypotheses = [b["hypothesis"] for b in batch]

        source = [b["source"] for b in batch]
        score = torch.tensor([b["score"] for b in batch])

        comet_features = self.get_comet_features(batch)
        other_features = self.get_other_features(batch)

        features = {**comet_features, **other_features}

        return source, hypotheses, features, score

    def get_comet_features(self, batch):
        h_emb = torch.stack(
            [torch.tensor(self.reference_lookup_table[b["hypothesis_id"]]["embedding"]) for b in batch]).to("cuda")

        s_emb = torch.stack([torch.tensor(self.source_lookup_table[b["source_index"]]["embedding"]) for b in batch]).to(
            "cuda")

        # Next we need to pick the references:
        references = np.array([self.get_references(b["ref_ids"], b["ref_counts"]) for b in batch])

        # Next we need to transpose it
        references = references.T

        reference_embeddings = [

        ]
        for ref_ids in references:
            r_emb = torch.stack(
                [torch.tensor(self.reference_lookup_table[ref_id]["embedding"]) for ref_id in ref_ids]).to("cuda")

            reference_embeddings.append(r_emb)

        features = {
            "hypothesis_embedding": h_emb,
            "source_embedding": s_emb,
            "reference_embeddings": reference_embeddings
        }

        return features

    def get_other_features(self, batch):
        hypotheses = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypotheses, truncation=True, max_length=self.max_seq_length)
            sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])

        model_inputs = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length)
        # Setup the tokenizer for targets

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        # Next we create the inputs for the NMT model
        x_new = self.data_collator(x).to("cuda")

        log_prob_entropy = pack_sequence(
            [torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1)], dim=-1)
             for b in batch], enforce_sorted=False)

        features = {
            "sequence_lengths": sequence_lengths,
            "log_prob_entropy": log_prob_entropy,
            **x_new,

        }

        return features
