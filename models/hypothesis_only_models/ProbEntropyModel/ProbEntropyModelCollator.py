import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq


class ProbEntropyModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, device="cuda"):


        self.device = device


    def __call__(self, batch):
        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])


        # We get the lenghts of the sequences (used for packing later)

        # Setup the tokenizer for targets

        log_prob = pack_sequence([torch.tensor(b["log_prob"]) .unsqueeze(-1)for b in batch] ,enforce_sorted=False)
        entropy = pack_sequence([torch.tensor(b["entropy"]).unsqueeze(-1) for b in batch] ,enforce_sorted=False)

        features = {
           "log_prob": log_prob,
            "entropy": entropy
        }
        return sources, hypotheses, features, score
