import time

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np

class BasicCollator:
    """
    A very basic collator
    Prepares the input for the basic model
    """

    def __init__(self, device="cuda", include_id=False):
        self.device = device

        self.incude_id = include_id

    def __call__(self, batch):

        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]

        utility = torch.tensor([b["utility"] for b in batch]).to("cuda").float()

        tokenized_source = pad_sequence([torch.tensor(b["tokenized_source"]) for b in batch], batch_first=True)
        tokenized_hypothesis = pad_sequence([torch.tensor(b["tokenized_hypothesis"]) for b in batch], batch_first=True)


        source_attention_mask = tokenized_source != 0
        hypothesis_attention_mask = tokenized_hypothesis != 0

        features = {
            "tokenized_source": tokenized_source.to("cuda"),
            "tokenized_hypothesis": tokenized_hypothesis.to("cuda"),
            "source_attention_mask": source_attention_mask,
            "hypothesis_attention_mask": hypothesis_attention_mask,

        }

        if self.incude_id:
            features["id"] = [b["index"] for b in batch]


        return sources, hypothesis, features, utility
