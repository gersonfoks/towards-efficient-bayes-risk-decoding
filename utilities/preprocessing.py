import numpy as np


class CustomTokenizer:

    def __init__(self, tokenizer, max_length=75):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_sources(self, sources):
        tokenized_source = self.tokenizer(sources, max_length=self.max_length,truncation=True )

        return tokenized_source

    def tokenize_hypotheses(self, hypotheses):
        with self.tokenizer.as_target_tokenizer():
            tokenized_hypotheses = self.tokenizer(hypotheses, truncation=True, max_length=self.max_length, )
        return tokenized_hypotheses


class SourceTokenizer:

    def __init__(self, tokenizer):
        self.custom_tokenizer = CustomTokenizer(tokenizer)

    def __call__(self, x):
        x["tokenized_source"] = self.custom_tokenizer.tokenize_sources(x["source"]).input_ids
        return x


class TargetTokenizer:

    def __init__(self, tokenizer):
        self.custom_tokenizer = CustomTokenizer(tokenizer)

    def __call__(self, x):
        x["tokenized_hypothesis"] = self.custom_tokenizer.tokenize_sources(x["hypothesis"]).input_ids

        return x






