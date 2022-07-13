import torch
from torch.nn.utils.rnn import pack_sequence
from transformers import DataCollatorForSeq2Seq


class LastHiddenLstmCollator:
    '''
    Tokenizes the hypotheses and put them into a packed sequence (as we work with lstms)
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, device="cuda"):
        self.nmt_model = nmt_model
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.device = device
        self.data_collator = DataCollatorForSeq2Seq(model=self.nmt_model, tokenizer=self.tokenizer,
                                               padding=True, return_tensors="pt",)

    def __call__(self, batch):
        hypotheses = [b["hypotheses"] for b in batch]
        sources = [b["source"] for b in batch]


        score = torch.tensor([b["score"] for b in batch])


        # We get the lenghts of the sequences (used for packing later)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(hypotheses, truncation=True, max_length=self.max_seq_length)
            sequence_lengths = torch.tensor([len(x) for x in labels["input_ids"]])

        model_inputs = self.tokenizer(sources, truncation=True,  max_length=self.max_seq_length)
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
        return sources, hypotheses, features, score
