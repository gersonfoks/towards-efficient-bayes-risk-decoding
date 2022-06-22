import torch

import numpy as np



def pick_random_references(references, counts, n_references):
    probs = np.array(counts) / np.sum(counts)

    selected_refs = np.random.choice(references, n_references, replace=True, p=probs)

    return selected_refs




class BaseCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, device="cuda", ):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


        self.device = device


    def get_hypotheses(self, batch):
        return [b["hypothesis"] for b in batch]

    def get_sources(self, batch):

        return [b["source"] for b in batch]

    def get_score(self, batch):

        return torch.tensor([b["utility"] for b in batch])

    def get_features(self, batch):
        raise NotImplementedError()


    def __call__(self, batch):

        hypotheses = self.get_hypotheses(batch)
        sources = self.get_sources(batch)

        score = self.get_score(batch)

        features = self.get_features(batch)

        return sources, hypotheses, features, score
