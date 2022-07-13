import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np

class BasicReferenceLstmModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, tokenizer, n_references=3, max_seq_length=75, device="cuda"):
        self.tokenizer = tokenizer
        self.n_references = n_references
        self.max_seq_length = max_seq_length

        self.device = device

    def get_references(self, references, counts):
        probs = np.array(counts) / np.sum(counts)

        selected_refs = np.random.choice(references, self.n_references, replace=True, p=probs)

        return selected_refs


    def __call__(self, batch):

        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])

        with self.tokenizer.as_target_tokenizer():
            tokenized_hypotheses = self.tokenizer(hypotheses, truncation=True,
                                                  max_length=self.max_seq_length)["input_ids"]

        # Next we need to pick the references:
        references = np.array([self.get_references(b["references"], b["reference_counts"]) for b in batch])

        #Next we need to transpose it
        references = references.T

        packed_references = []
        with self.tokenizer.as_target_tokenizer():
            for ref in references:

                tokenized_ref = self.tokenizer(ref.tolist(), truncation=True,
                                                  max_length=self.max_seq_length)["input_ids"]

                packed_ref = pack_sequence([torch.tensor(h) for h in tokenized_ref], enforce_sorted=False,).to(self.device)

                packed_references.append(packed_ref)



        # Then we need to pack them

        # Next we create a packed sequence:
        packed_hypotheses = pack_sequence([torch.tensor(h) for h in tokenized_hypotheses], enforce_sorted=False,).to(self.device)

        features = {
            "tokenized_hypotheses": packed_hypotheses,
            "tokenized_references": packed_references
        }
        return sources, hypotheses, features, score
