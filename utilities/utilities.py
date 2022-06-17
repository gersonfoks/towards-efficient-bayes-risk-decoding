# Code and inspiration taken from: https://github.com/Roxot/mbr-nmt/blob/a419775b638c4b09e962acad71c4468269b0197a/mbr_nmt/utility.py#L250
from typing import List

from nltk.util import ngrams



class Utility:

    def __init__(self):
        # Whether this utility supports batching with self.sentence_scores(hyps, refs)
        self.supports_batching = False

        # Whether this utility requires tokenization as pre-processing step (requires self.tokenizer to be set).
        self.requires_tokenization = False
        self.tokenizer = None

    def sentence_scores(self, hyps, refs):
        """
        :param hyps: list of strings, system hypotheses.
        :param refs: list of strings, single reference per input.

        Returns a list of sentence-level scores. Required if self.supports_batching == True.
        """
        pass

    def __call__(self, source: str, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        returns the utility score of a single hypothesis, reference pair as float.
        """
        pass


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

