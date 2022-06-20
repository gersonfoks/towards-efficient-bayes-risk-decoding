import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np

class CometEncodingModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, source_lookup_table, reference_lookup_table, n_references=3, max_seq_length=75, device="cuda"):

        self.reference_lookup_table = reference_lookup_table
        self.source_lookup_table = source_lookup_table
        self.n_references = n_references
        self.max_seq_length = max_seq_length

        self.device = device

    def get_references(self, references, counts):
        probs = np.array(counts) / np.sum(counts)

        selected_refs = np.random.choice(references, self.n_references, replace=True, p=probs)

        return selected_refs


    def __call__(self, batch):



        h_emb = torch.stack([torch.tensor(self.reference_lookup_table[b["hypothesis_id"]]["embedding"]) for b in batch]).to("cuda")

        s_emb = torch.stack([torch.tensor(self.source_lookup_table[b["source_index"]]["embedding"]) for b in batch]).to("cuda")


        score = torch.tensor([b["score"] for b in batch])



        # Next we need to pick the references:
        references = np.array([self.get_references(b["ref_ids"], b["ref_counts"]) for b in batch])

        #Next we need to transpose it
        references = references.T

        reference_embeddings = [

        ]
        for ref_ids in references:
            r_emb = torch.stack([torch.tensor(self.reference_lookup_table[ref_id]["embedding"]) for ref_id in ref_ids]).to("cuda")

            reference_embeddings.append(r_emb)







        # Then we need to pack them

        # Next we create a packed sequence:

        features = {
            "hypothesis_embedding": h_emb,
            "source_embedding": s_emb,
            "reference_embeddings": reference_embeddings
        }
        return None, None, features, score
