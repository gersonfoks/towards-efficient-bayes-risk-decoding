import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np
from transformers import DataCollatorForSeq2Seq


class TokenStatisticsCollator:
    '''
    Prepares the token statistics
    '''

    def __init__(self, lookup_table, max_seq_length=75, device="cuda"):

        self.max_seq_length = max_seq_length

        self.device = device
        self.lookup_table = lookup_table

        self.features = [
            "entropy",
            "top_5",
            "prob"
        ]



    def __call__(self, batch):

        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]

        utility = torch.tensor([b["utility"] for b in batch])

        hypothesis_ids = [b["hypothesis_id"] for b in batch]

        features = self.lookup_table.get_features(hypothesis_ids)

        entropy = [torch.tensor(f["entropy"]) for f in features]

        top_5 = [torch.tensor([x.astype(float) for x in f["top_5"]]) for f in features]
        prob = [torch.tensor(f["prob"]) for f in features]




        top_5 = pad_sequence(top_5, batch_first=True)

        entropy = pad_sequence(entropy, batch_first=True)
        prob = pad_sequence(prob, batch_first=True)



        attention_mask = (prob != 0.0).long()


        stacked_features = torch.concat([prob.unsqueeze(-1), entropy.unsqueeze(-1), top_5], dim=-1).float()


        features = {

            "token_statistics": stacked_features,
            "attention_mask": attention_mask,
        }
        return sources, hypothesis, features, utility