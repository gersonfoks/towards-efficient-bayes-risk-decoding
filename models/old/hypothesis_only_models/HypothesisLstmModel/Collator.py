import torch
from torch.nn.utils.rnn import pack_sequence


class HypothesisLstmModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, tokenizer, max_seq_length=75, device="cuda"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device

    def __call__(self, batch):
        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])

        with self.tokenizer.as_target_tokenizer():
            tokenized_hypotheses = self.tokenizer(hypotheses, truncation=True,
                                                  max_length=self.max_seq_length)["input_ids"]


        # Next we create a packed sequence:
        packed_hypotheses = pack_sequence([torch.tensor(h) for h in tokenized_hypotheses], enforce_sorted=False,).to(self.device)

        features = {
            "tokenized_hypotheses": packed_hypotheses
        }
        return sources, hypotheses, features, score
