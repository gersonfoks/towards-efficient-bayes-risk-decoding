import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq


class ProbEntropyModelCollatorV2:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, device="cuda"):


        self.device = device


    def __call__(self, batch):
        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])



        log_prob_entropy = pack_sequence([torch.concat([torch.tensor(b["log_prob"]).unsqueeze(-1), torch.tensor(b["entropy"]).unsqueeze(-1) ], dim=-1) for b in batch] ,enforce_sorted=False)


        features = {
           "log_prob_entropy": log_prob_entropy,

        }
        return sources, hypotheses, features, score
