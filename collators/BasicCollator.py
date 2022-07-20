import time

import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np

class BasicCollator:

    def __init__(self, tokenizer, max_seq_length=75, device="cuda"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device

    def __call__(self, batch):



        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]


        utility = torch.tensor([b["utility"] for b in batch]).to("cuda")

        tokenized_source = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length, padding=True,
                                    return_tensors="pt",)


        # Make sure that we don't use the same tokens
        with self.tokenizer.as_target_tokenizer():
            tokenized_hypothesis = self.tokenizer(hypothesis, truncation=True, max_length=self.max_seq_length, padding=True,
                                    return_tensors="pt",)




        features = {
            "tokenized_source": tokenized_source.to("cuda"),
            "tokenized_hypothesis": tokenized_hypothesis.to("cuda")
        }

        end = time.time()

        return sources, hypothesis, features, utility
