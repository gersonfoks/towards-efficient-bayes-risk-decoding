import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np
from transformers import DataCollatorForSeq2Seq


class UnigramCountCollator:
    '''
    Prepares the model for the nmt input
    '''

    def __init__(self, reference_table, tokenizer, max_seq_length=75, device="cuda", include_source_id=False,
                 n_model_references=5):

        self.reference_table = reference_table
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device

        self.include_source_id = include_source_id

        self.n_model_references = n_model_references

    def get_utilities(self, batch):

        resulting_utilities = []
        references = []
        for b in batch:

            reference_info = self.reference_table[b["index"]]


            probs = reference_info["references_count"]/np.sum(reference_info["references_count"])

            indices = np.random.choice(len(reference_info["references"]), p=probs, replace=True, size=self.n_model_references)

            chosen_utilities = b["utilities"][indices]

            chosen_refs = reference_info["references"][indices].tolist()

            resulting_utilities.append(np.mean(chosen_utilities))

            references.append(chosen_refs)

        return references, torch.tensor(resulting_utilities).to("cuda").float()

    def __call__(self, batch):
        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]

        utility = torch.tensor([b["utility"] for b in batch]).float()

        # We get the lengths of the sequences (used for packing later)
        with self.tokenizer.as_target_tokenizer():
            hypothesis_ids = self.tokenizer(hypothesis, truncation=True, max_length=self.max_seq_length, padding=True, return_tensors='pt').input_ids


        references, ref_utilities = self.get_utilities(batch)

        references_ids = []

        for refs in references:

            refs_ids = self.tokenizer(refs, truncation=True, padding=True, max_length=self.max_seq_length, return_tensors='pt').input_ids.to("cuda")

            references_ids.append(refs_ids)



        features = {
            "hypothesis_ids": hypothesis_ids.to("cuda"),
            'references_ids': references_ids,
            "mean_utilities": ref_utilities,

        }

        if self.include_source_id:
            features["source_index"] = [b["source_index"] for b in batch]
        return sources, hypothesis, features, utility
