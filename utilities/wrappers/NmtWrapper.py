import numpy as np
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

    @torch.no_grad()
    def map_to_top_k_prob(self, batch, k):
        nmt_in, lengths = self.prepare_batch(batch)

        nmt_out = self.nmt_model(**nmt_in)
        logits = nmt_out["logits"]

        hyp_input_ids = (nmt_in["labels"] * (nmt_in["labels"] != -100)).long()  # Get the indices

        ids = hyp_input_ids.unsqueeze(dim=-1)

        log_softmax = torch.nn.LogSoftmax(dim=-1)

        log_probs_all_tokens = log_softmax(logits)

        # Get top 10
        top_k = torch.topk(log_probs_all_tokens,  k, dim=-1, )


        log_probs_selected_tokens = log_probs_all_tokens.gather(-1, ids).squeeze(dim=-1)
        log_probs_selected_tokens = self.shorten_by_length(log_probs_selected_tokens, lengths)
        top_k = self.shorten_by_length(top_k.values, lengths)

        return {"top_k_prob": top_k,
                "prob": log_probs_selected_tokens
                }

    @torch.no_grad()
    def map_to_log_probs_and_entropy(self, batch):
        nmt_in, lengths = self.prepare_batch(batch)

        log_prob, entropy = self.get_log_prob_and_entropy(nmt_in, lengths)

        avg_log_prob = [np.mean(lp) for lp in log_prob]
        std_log_prob = [np.std(lp) for lp in log_prob]
        avg_entr = [np.mean(entr) for entr in entropy]
        std_entr = [np.std(entr) for entr in entropy]
        return {"log_prob": log_prob,
                "entropy": entropy,
                "mean_log_prob": avg_log_prob,
                "std_log_prob": std_log_prob,
                "mean_entropy": avg_entr,
                "std_entropy": std_entr,
                **batch}

    def shorten_by_length(self, vector, lengths):
        return [v[:l].cpu().numpy().tolist() for v, l in zip(vector, lengths)]

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
