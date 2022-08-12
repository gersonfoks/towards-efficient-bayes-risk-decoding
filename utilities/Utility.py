# Code and inspiration taken from: https://github.com/Roxot/mbr-nmt/blob/a419775b638c4b09e962acad71c4468269b0197a/mbr_nmt/utility.py#L250
from typing import List

from comet import download_model, load_from_checkpoint
from nltk.util import ngrams

from utilities.misc import load_nmt_model
from utilities.wrappers.CometWrapper import CometWrapper


class DecoderTokenizer:

    def __init__(self, nmt_tokenizer, max_seq_length=75):
        self.nmt_tokenizer = nmt_tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, sentences):
        with self.nmt_tokenizer.as_target_tokenizer():
            tokenized_sentences = self.nmt_tokenizer(sentences, truncation=True,
                                                     max_length=self.max_seq_length)["input_ids"]
        return tokenized_sentences


class Utility:

    def __call__(self, source: str, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        raise NotImplementedError()

    def call_batched_fast(self, source: str, hypotheses: List[str], refs: List[str]):
        raise NotImplementedError()

    def call_batched(self, sources: List[str], hypotheses: List[str], refs: List[List[str]]):
        raise NotImplementedError()


def load_utility(utility, nmt_model=None, tokenizer=None):
    '''
    Loads the utility function
    '''
    if utility == "unigram-f1":
        # Get the nmt model tokenizer
        config = {

            "name": 'Helsinki-NLP/opus-mt-de-en',
            "checkpoint": './saved_models/NMT/de-en-model/',
            "type": 'MarianMT'

        }
        if tokenizer == None:
            nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

        tokenizer = DecoderTokenizer(tokenizer)
        return NGramF(1, tokenize=True, tokenizer=tokenizer)
    elif utility == "comet":
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)
        model.to("cuda")
        model.eval()
        wrapped_model = CometWrapper(model)
        return CometUtility(wrapped_model, )

    else:
        raise ValueError("utility: {} not found!".format(utility))


class CometUtility(Utility):

    def __init__(self, wrapped_model, ):
        self.wrapped_model = wrapped_model

    def call_batched_fast(self, source: str, hypotheses: List[str], refs: List[str]):
        '''
            A fast call for the batch. Hypotheses and references should be for the given source.
        '''
        return self.wrapped_model.fast_predict_batched(source, hypotheses, refs)


class NGramF(Utility):

    def __init__(self, n, tokenize=False, tokenizer=None):
        Utility.__init__(self)
        self.n = n
        self.tokenize = tokenize
        self.tokenizer = tokenizer

    def __call__(self, source: str, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        if self.tokenize:
            hyp = self.tokenizer(hyp)
            ref = self.tokenizer(ref)
        assert isinstance(hyp, str) and isinstance(ref, str)
        hyp_set = set(ngrams(hyp.split(' '), self.n))
        ref_set = set(ngrams(ref.split(' '), self.n))
        matches = hyp_set.intersection(ref_set)
        n = len(matches)
        p = n / len(hyp_set) if len(hyp_set) else 0.
        r = n / len(ref_set) if len(ref_set) else 0.
        return 0. if (p + r) == 0. else 2. * p * r / (p + r)

    def call_batched_fast(self, source: str, hypotheses: List[str], refs: List[str]):
        scores = []

        # First we tokenize:
        if self.tokenize:
            hypotheses = self.tokenizer(hypotheses)

            hypotheses = [set(ngrams(hyp, self.n)) for hyp in hypotheses]

            refs = self.tokenizer(refs)
            refs = [set(ngrams(ref, self.n)) for ref in refs]

        for hyp_set in hypotheses:
            scores_for_hyp = []
            for ref_set in refs:
                matches = hyp_set.intersection(ref_set)
                n = len(matches)
                p = n / len(hyp_set) if len(hyp_set) else 0.
                r = n / len(ref_set) if len(ref_set) else 0.
                s = 0. if (p + r) == 0. else 2. * p * r / (p + r)
                scores_for_hyp.append(s)
            scores.append(scores_for_hyp)
        return scores

    def call_batched(self, sources: List[str], hypotheses: List[str], refs: List[List[str]]):

        scores = []
        for s, h, r in zip(sources, hypotheses, refs):
            scores.append(self.call_batched_fast(s, [h], r))
        return scores
