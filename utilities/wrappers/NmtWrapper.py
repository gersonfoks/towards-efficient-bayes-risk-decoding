from time import time

import torch
from transformers import DataCollatorForSeq2Seq


class NMTWrapper:
    """"
    A wrapper to have handy functions for the nmt model
    """

    def __init__(self, nmt_model, tokenizer, max_seq_length=75):
        self.nmt_model = nmt_model.to("cuda")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.data_collator = DataCollatorForSeq2Seq(model=nmt_model, tokenizer=tokenizer,
                                                    padding=True, return_tensors="pt", max_length=max_seq_length)
        self.batch_size = 32


    def prepare_for_nmt(self, sources, hypotheses):
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypotheses, truncation=True, max_length=self.max_seq_length)
            sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])

        model_inputs = self.tokenizer(sources, truncation=True, max_length=self.max_seq_length)
        # Setup the tokenizer for targets

        model_inputs["labels"] = labels["input_ids"]

        x = [{"labels": l, "input_ids": i, "attention_mask": a} for (l, i, a) in
             zip(model_inputs["labels"], model_inputs["input_ids"], model_inputs["attention_mask"])]

        # Next we create the inputs for the NMT model
        x_new = self.data_collator(x).to("cuda")

        return x_new, sequence_lengths

    def prepare_batch(self, batch):


        sources = batch["source"]
        hypotheses = batch["hypothesis"]

        with self.tokenizer.as_target_tokenizer():
            hyp_tokenized = self.tokenizer(hypotheses, max_length=self.max_seq_length, )
            lengths = [len(ids) for ids in hyp_tokenized["input_ids"]]

        source_tokenized = self.tokenizer(sources, max_length=self.max_seq_length)

        features = [
            {
                "input_ids": input_id,
                "attention_mask": att_mask,
                "labels": label
            }
            for input_id, att_mask, label in
            zip(source_tokenized.input_ids, source_tokenized.attention_mask, hyp_tokenized.input_ids)
        ]

        nmt_in = self.data_collator(features).to("cuda")
        return nmt_in, lengths

    def timed_forward(self, source, hypothesis):

        # Preperation is not taken into account for fair comparison
        nmt_in, _ = self.prepare_for_nmt([source], [hypothesis])

        # Keep track of time it takes to go through the model
        start_time = time()
        self.nmt_model.forward(**nmt_in)
        end_time = time()
        return end_time - start_time