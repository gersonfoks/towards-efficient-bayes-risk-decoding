import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np
from transformers import DataCollatorForSeq2Seq


class BasicReferenceCollator:
    '''
    Prepares the model for the nmt input
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, device="cuda", include_source_id=False,
                 n_ref_utilities=5):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt", )

        self.include_source_id = include_source_id

        self.n_ref_utilities = n_ref_utilities

    def get_utilities(self, batch):

        resulting_utilities = []
        for b in batch:
            indices = np.random.choice(len(b["utilities"]),p=b["probs"], replace=True, size=self.n_ref_utilities)

            chosen_utilities = b["utilities"][indices]
            
            resulting_utilities.append(np.mean(chosen_utilities))

        return torch.tensor(resulting_utilities).to("cuda")


    def __call__(self, batch):
        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]

        utility = torch.tensor([b["utility"] for b in batch]).float()

        # We get the lengths of the sequences (used for packing later)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypothesis, truncation=True, max_length=self.max_seq_length)
            sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])

        model_inputs = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length)
        # Setup the tokenizer for targets

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        # Next we create the inputs for the NMT model
        x_new = self.data_collator(x).to("cuda")

        ref_utilities = self.get_utilities(batch)

        features = {
            "sequence_lengths": sequence_lengths,
            "mean_utilities": ref_utilities,
            **x_new

        }

        if self.include_source_id:
            features["source_index"] = [b["source_index"] for b in batch]
        return sources, hypothesis, features, utility
