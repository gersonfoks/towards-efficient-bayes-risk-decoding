import torch
from torch.nn.utils.rnn import pack_sequence
import numpy as np
from transformers import DataCollatorForSeq2Seq


class LastHiddenStateRefModelCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, n_references=3, max_seq_length=75, device="cuda"):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.n_references = n_references
        self.max_seq_length = max_seq_length

        self.device = device

        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                                    padding=True, return_tensors="pt", )

    def get_references(self, references, counts):
        probs = np.array(counts) / np.sum(counts)

        selected_refs = np.random.choice(references, self.n_references, replace=True, p=probs)

        return selected_refs


    def __call__(self, batch):

        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])

        references = np.array([self.get_references(b["references"], b["reference_counts"]) for b in batch])
        references = references.T

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypotheses, truncation=True,
                                                  max_length=self.max_seq_length)
            hypotheses_sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])

        model_inputs = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length)

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        x_new = self.data_collator(x).to("cuda")

        # Next we need to pick the references:

        #Next we need to transpose it


        reference_inputs = []
        ref_sequence_lengths = []
        with self.tokenizer.as_target_tokenizer():
            for ref in references:

                tokenized_ref = self.tokenizer(ref.tolist(), truncation=True,
                                                  max_length=self.max_seq_length)
                seq_lenghts = torch.tensor([len(t) for t in tokenized_ref["input_ids"]])

                model_inputs["labels"] = tokenized_ref["input_ids"]

                ref_input_temp = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
                     zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

                ref_input = self.data_collator(ref_input_temp).to("cuda")

                reference_inputs.append(ref_input)

                ref_sequence_lengths.append(seq_lenghts)






        # Then we need to pack them

        # Next we create a packed sequence:

        features = {
            "hypotheses_sequence_lengths": hypotheses_sequence_lengths,
            **x_new,
            "ref_sequence_lengths": ref_sequence_lengths,
            "reference_inputs": reference_inputs
        }
        return sources, hypotheses, features, score
