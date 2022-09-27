import math

import pandas as pd
import transformers
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel
from typing import List
import numpy as np

def load_nmt_model(config, pretrained=False):
    '''
    Loads the model described in the config,
    :param config:
    :param pretrained: if true we load a pretrained model found at the given checkpoint
    :return:
    '''
    model_name = config["name"]
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    if pretrained:
        model = MarianMTModel.from_pretrained(config["checkpoint"])
    else:
        configuration = transformers.AutoConfig.from_pretrained(model_name)
        model = MarianMTModel(configuration)

    return model, tokenizer


def load_tatoeba_dataframe(split):
    '''
    Loads the tatoeba split as a pandas dataframe
    :param split: the split to load
    :return:
    '''

    base_dir = './data/tatoeba_splits/'



    df = pd.read_parquet(base_dir + '{}.parquet'.format(split))

    return df




def preprocess_tokenize(examples, tokenizer, prefix="", source_lang="de", target_lang="en",):
    '''
    Preprocessess for the NMT model
    :param examples: examples to preprocess
    :param tokenizer: the tokenizer to use
    :param prefix: the prefix to add
    :param source_lang: the source language
    :param target_lang: the target language
    :return:
    '''
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs,  truncation=True, )
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def batch_sample(model, tokenizer, texts, n_samples=96, batch_size=32, sampling_method='ancestral'):
    samples = []
    n_loops = math.ceil(n_samples/batch_size)

    last_batch_size = n_samples % batch_size
    for i in range(n_loops):
        # Make sure we generate enough samples by dynamic allocating
        n = batch_size
        if i == n_loops- 1:
            n = last_batch_size
            if n == 0:
                n = batch_size
        tokenized = tokenizer(texts, return_tensors="pt", padding=True, ).to("cuda")

        if sampling_method == 'ancestral':
            sample = model.generate(
                    **tokenized,
                    do_sample=True,
                    num_beams=1,
                    top_k=0,
                    num_return_sequences=n,
                    max_length=75,
                )
        elif sampling_method == 'beam':
            sample = model.generate(
                **tokenized,
                do_sample=True,
                num_beams=5,
                num_return_sequences=n,
                max_length=75,
            )
        else:
            raise "sample_method not found: {}".format(sampling_method)

        decoded_samples = tokenizer.batch_decode(sample, skip_special_tokens=True)
        samples += decoded_samples

    return samples


def batch(iterable, n=1, l=None):
    '''
    Batches an iterable
    '''
    if not l:
        l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




def load_bayes_risk_dataframe(sampling_method, n_hypotheses, n_references, split, utility='comet', seed=0,  smoke_test=False, prepend='.'):
    '''
    Loads a bayes risk dataset
    :param sampling_method: the method which the samples were generated
    :param n_hypotheses: the number of hypotheses used
    :param n_references: The number of references used
    :param utility: The utility used
    :param seed: Random seed number
    :param smoke_test: True if we want to use the smoke_test (small) dataset
    :return:
    '''

    ref = '{}/data/{}/{}_{}_{}_{}_{}'.format(prepend, utility, sampling_method, n_hypotheses, n_references,split, seed )

    if smoke_test:
        ref += '_smoke_test'
    ref += '.parquet'

    df = pd.read_parquet(ref)

    return df


def map_to_utility(x):

    utilities = x["utilities"]
    references_count = np.array(x["references_count"])
    all_utilities = [

    ]
    for util in utilities:

        utility = np.sum(references_count * util, axis=-1) / np.sum(references_count)
        all_utilities.append(utility)


    return all_utilities


def translate(model, tokenizer, source_texts, batch_size=32, method="ancestral"):
    translations = []
    total = math.ceil(len(source_texts) /batch_size)
    for lines in tqdm(batch(source_texts, n=batch_size), total=total):
        samples = sample_model(model, tokenizer, lines, method=method)
        decoded_samples = tokenizer.batch_decode(samples, skip_special_tokens=True)

        translations += decoded_samples
    return translations


def sample_model(model, tokenizer, source_texts, method="ancestral",):
    sample = None

    tokenized = tokenizer(source_texts, return_tensors="pt", padding=True, ).to("cuda")
    if method == "ancestral":
        sample = model.generate(
            **tokenized,
            do_sample=True,
            num_beams=1,
            top_k=0,
        )
    elif method == "beam":
        sample = model.generate(
            **tokenized,
            do_sample=True,
            num_beams=5,
            early_stopping=True
        )

    else:
        raise(ValueError("Method not implemented: {}".format(method)))

    return sample


