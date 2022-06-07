import torch
from datasets import Dataset
import numpy as np
from transformers import DataCollatorForSeq2Seq


def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score


class ProbEntropyModelPreprocess:
    '''
    Splits the dataset hypotheses column and takes the average of the unigram f1 scores
    '''

    def __init__(self, nmt_model, tokenizer, max_seq_length=75, ):
        self.nmt_model = nmt_model
        self.nmt_model.eval()
        self.nmt_model =self.nmt_model.to("cuda")
        self.tokenizer = tokenizer

        self.max_seq_length = max_seq_length

        self.data_collator = DataCollatorForSeq2Seq(model=nmt_model, tokenizer=tokenizer,
                                                    padding=True, return_tensors="pt", max_length=max_seq_length)
        self.batch_size = 32

    def __call__(self, data):
        df_exploded = self.explode_dataset(data)

        df_exploded["score"] = df_exploded["utilities"]

        dataset = Dataset.from_pandas(df_exploded)

        # Lastly we want to calculate the probs and entropy of all the tokens

        dataset = dataset.map(self.map_to_log_probs_and_entropy, batch_size=self.batch_size, batched=True)


        return dataset

    def explode_dataset(self, data):
        df_exploded = data.explode(["hypotheses", "utilities", "count", ], ignore_index=True)

        return df_exploded

    def map_to_log_probs_and_entropy(self, batch):
        sources = batch["source"]
        hypotheses = batch["hypotheses"]

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
        return {"log_prob": log_prob,
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

        hyp_input_ids = (nmt_in["labels"] * (nmt_in["labels"] != -100)).long() # Get the indices

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
