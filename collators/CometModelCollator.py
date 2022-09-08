import time

import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np

class CometModelCollator:
    """
    A collator for the comet model
    """

    def __init__(self,  source_embedding_table, hypotheses_embedding_table,  device="cuda",include_source_id=False):
        self.device = device

        self.source_embedding_table = source_embedding_table
        self.hypotheses_embedding_table = hypotheses_embedding_table


        self.include_source_id = include_source_id

    def __call__(self, batch):

        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]

        utility = torch.tensor([b["utility"] for b in batch]).to("cuda").float()

        source_ids = [b["index"] for b in batch]
        hypothesis_ids = [b["hypothesis_id"] for b in batch]

        source_embeddings = self.source_embedding_table.get_embeddings(source_ids)
        hypothesis_embeddings = self.hypotheses_embedding_table.get_embeddings(hypothesis_ids)

        features = {
            "source_embedding": source_embeddings.to("cuda"),
            "hypothesis_embedding": hypothesis_embeddings.to("cuda"),

        }

        if self.include_source_id:
            features["source_index"] = [b["source_index"] for b in batch]


        return sources, hypothesis, features, utility
