import math

import pandas as pd
import transformers
from transformers import MarianTokenizer, MarianMTModel


def load_nmt_model(config, pretrained=False):
    '''
    Loads the model described in the config,
    :param config:
    :param pretrained: if true we load a pretrained model found at the given checkpoint
    :return:
    '''
    model_name = config["model"]["name"]
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    if pretrained:
        model = MarianMTModel.from_pretrained(config["model"]["checkpoint"])
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
    for i in range(math.ceil(n_samples/batch_size)):
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
            raise "sample_method not found: {}".format(sample_method)

        decoded_samples = tokenizer.batch_decode(sample, skip_special_tokens=True)
        samples += decoded_samples
    return samples
