from datasets import Dataset
import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq

from utilities.LookUpTable import LookUpTable


def create_hypothesis_ids(x):
    hypotheses_ids = []
    for ids, hypotheses in zip(x["index"], x["hypotheses"]):
        hypotheses_ids.append([
            "{}_{}".format(ids, i) for i in range(len(hypotheses))
        ])
    return {
        "hypotheses_ids": hypotheses_ids,
        **x
    }


class BasicReferenceLstmModelV2Preprocess:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75):
        self.nmt_model = nmt_model.to("cuda")
        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length
        self.data_collator = DataCollatorForSeq2Seq(model=nmt_model, tokenizer=tokenizer,
                                                    padding=True, return_tensors="pt", max_length=max_seq_length)
        self.batch_size = 32

        self.features = [
            "mean_log_prob",
            "std_log_prob",
            "mean_entropy",
            "std_entropy"

        ]

    def __call__(self, data):
        # Add hypothesis id

        data = data.reset_index()
        dataset = Dataset.from_pandas(data)
        dataset = dataset.map(create_hypothesis_ids, batched=True, batch_size=32).to_pandas()

        dataset["ref_ids"] = dataset["hypotheses_ids"]
        dataset["ref_counts"] = dataset["count"]

        # Explode the dataset
        dataset = dataset.explode(["hypotheses", "utilities", "count", "hypotheses_ids"], ignore_index=True).rename(
            columns={
                "hypotheses": "hypothesis",
                "utilities": "utility",
                "index": "source_index",
                "hypotheses_ids": "hypothesis_id"
            })

        main_dataset = dataset[["source", "hypothesis_id", "ref_ids", "ref_counts", "utility"]]
        # First we need to add the probabilities and entropy to the hypothesis

        hyp_dataset = Dataset.from_pandas(dataset[["hypothesis_id", "hypothesis", "utility", "source"]])

        hyp_dataset = hyp_dataset.map(self.map_to_log_probs_and_entropy, batch_size=self.batch_size,
                                      batched=True).to_pandas()
        look_up_table = LookUpTable(hyp_dataset, index="hypothesis_id", features=["hypothesis", "utility", ] + self.features)

        #
        data["references"] = data["hypotheses"]
        data["reference_counts"] = data["count"]

        main_dataset["score"] = main_dataset["utility"]

        main_dataset = Dataset.from_pandas(main_dataset)

        return main_dataset, look_up_table

    def map_to_log_probs_and_entropy(self, batch):
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

        log_prob, entropy = self.get_log_prob_and_entropy(nmt_in, lengths)

        avg_log_prob = [np.mean(lp) for lp in log_prob]
        std_log_prob = [np.std(lp) for lp in log_prob]
        avg_entr = [np.mean(entr) for entr in entropy]
        std_entr = [np.std(entr) for entr in entropy]

        return {"log_probs": log_prob,
                "entropy": entropy,
                "mean_log_prob": avg_log_prob,
                "std_log_prob": std_log_prob,
                "mean_entropy": avg_entr,
                "std_entropy": std_entr,
                **batch}

    def shorten_by_length(self, vector, lengths):
        return [v[:l].cpu().numpy() for v, l in zip(vector, lengths)]

    @torch.no_grad()
    def get_log_prob_and_entropy(self, nmt_in, lengths):
        nmt_out = self.nmt_model(**nmt_in)
        logits = nmt_out["logits"]

        hyp_input_ids = (nmt_in["labels"] * (nmt_in["labels"] != -100)).long()  # Get the indices

        ids = hyp_input_ids.unsqueeze(dim=-1)

        log_softmax = torch.nn.LogSoftmax(dim=-1)

        log_probs_all_tokens = log_softmax(logits)

        # Get the chosen log probabilities
        log_probs = log_probs_all_tokens.gather(-1, ids).squeeze(dim=-1)
        probs_all_tokens = torch.exp(log_probs_all_tokens)

        entropy = - torch.sum(log_probs_all_tokens * probs_all_tokens, dim=-1)

        log_probs = self.shorten_by_length(log_probs, lengths)
        entropy = self.shorten_by_length(entropy, lengths)

        return log_probs, entropy
