import torch
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import numpy as np
from transformers import DataCollatorForSeq2Seq


class RefFullDecCollator:
    '''
    Prepares the token statistics
    '''

    def __init__(self, nmt_model, tokenizer, lookup_table, max_seq_length=75, device="cuda", n_references=3):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device
        self.lookup_table = lookup_table

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt", )

        self.features = [
            "entropy",
            "top_5",
            "prob"
        ]

        self.n_references = n_references

    def pick_reference(self, b):

        p = b["utilities_count"]/np.sum(b["utilities_count"])
        indices = np.random.choice(len(b["utilities_count"]), size=self.n_references, p=p)

        return indices

    def prepare_for_nmt(self, sources, hypothesis):
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

        features = {
            "sequence_lengths": sequence_lengths,
            **x_new
        }

        return features



    def __call__(self, batch):
        hypothesis = [b["hypothesis"] for b in batch]
        sources = [b["source"] for b in batch]


        utility = torch.tensor([b["utility"] for b in batch])


        hypothesis_ids = [b["hypothesis_id"] for b in batch]

        features = self.lookup_table.get_features(hypothesis_ids)

        entropy = [torch.tensor(f["entropy"]) for f in features]

        top_5 = [torch.tensor([x.astype(float) for x in f["top_5"]]) for f in features]
        prob = [torch.tensor(f["prob"]) for f in features]

        top_5 = pad_sequence(top_5, batch_first=True)

        entropy = pad_sequence(entropy, batch_first=True)
        prob = pad_sequence(prob, batch_first=True)

        attention_mask = (prob != 0.0).long()

        stacked_features = torch.concat([prob.unsqueeze(-1), entropy.unsqueeze(-1), top_5], dim=-1).float()

        features_for_nmt = self.prepare_for_nmt(sources, hypothesis)
        #Lastly we pick references and get their mean utility
        reference_indices = [self.pick_reference(b) for b in batch]

        utilities = torch.tensor([np.array(b["ref_utilities"])[i] for i, b in zip(reference_indices, batch)])



        mean_utilities = torch.mean(utilities, dim=-1)



        features = {
            "utilities": utilities,
            "mean_utilities": mean_utilities,
            "token_statistics": stacked_features,
            "attention_mask_tokens": attention_mask,
            **features_for_nmt
        }
        return sources, hypothesis, features, utility