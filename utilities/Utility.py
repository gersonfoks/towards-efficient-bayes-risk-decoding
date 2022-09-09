# Code and inspiration taken from: https://github.com/Roxot/mbr-nmt/blob/a419775b638c4b09e962acad71c4468269b0197a/mbr_nmt/utility.py#L250
from typing import List

import sacrebleu
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

    elif utility == 'chrf':
        return ChrF()

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

        # scores = Parallel(n_jobs=8)(delayed(self.call_batched_fast(s, [h], r)) for s, h, r in zip(sources, hypotheses, refs))
        return scores


class ChrF(Utility):

    def __init__(self):
        Utility.__init__(self)
        self.n_word_order = 2

    def __call__(self, source: str, hyp: str, ref: str):
        """
        :param hyp: string, system hypothesis, tokens separated by spaces
        :param ref: string, single reference, tokens separated by spaces
        """
        return sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score

    def call_batched_fast(self, source: str, hypotheses: List[str], refs: List[str]):
        # scores = []
        #
        # for hyp in hypotheses:
        #     scores_for_hyp = [sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score/100 for ref in refs]
        #
        #     scores.append(scores_for_hyp)
        from joblib import Parallel, delayed

        # scores = []
        #
        #
        #
        #
        # for hyp in hypotheses:
        #     scores_for_hyp = Parallel(n_jobs=4)(
        #         delayed(lambda : sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100)() for ref in refs)
        #
        #     scores.append(scores_for_hyp)

        def get_score(hyp):
            return [sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score / 100 for ref in refs]

        scores =  Parallel(n_jobs=8)(
                    delayed(get_score)(hyp) for hyp in hypotheses)
        #


        return scores

    def call_batched(self, sources: List[str], hypotheses: List[str], refs: List[List[str]]):

        scores = []
        for s, h, r in zip(sources, hypotheses, refs):
            scores.append(self.call_batched_fast(s, [h], r))
        # scores = Parallel(n_jobs=8)(
        #     delayed(self.call_batched_fast(s, [h], r)) for s, h, r in zip(sources, hypotheses, refs))
        return scores
