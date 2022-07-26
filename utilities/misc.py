import math
import os

import transformers
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel

# Temp hack
from utilities.PathManager import get_path_manager



def load_nmt_model(config, pretrained=False):
    '''
    Loads the model described in the config,
    :param config:
    :param pretrained: if we load a pretrained model or not
    :return:
    '''

    path_manager = get_path_manager()


    model_name = config["name"]

    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Load the base model
    Base = None
    if config["type"] == "MarianMT":
        Base = MarianMTModel
    else:
        raise ValueError("Base model not found: {}".format(config["type"]))

    model = None
    if pretrained:


        config_path = path_manager.get_abs_path(config["checkpoint"])

        model = Base.from_pretrained(config_path)
    else:
        configuration = transformers.AutoConfig.from_pretrained(model_name)
        model = Base(configuration)

    return model, tokenizer


def load_comet_model():
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    model.to("cuda")
    model.eval()

    return model


def batch(iterable, n=1):
    '''
    Batches an iterable
    '''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]




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


def batch_sample(model, tokenizer, texts, n_samples=96, batch_size=32):
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

        sample = model.generate(
                **tokenized,
                do_sample=True,
                num_beams=1,
                top_k=0,
                num_return_sequences=n

            )

        decoded_samples = tokenizer.batch_decode(sample, skip_special_tokens=True)
        samples += decoded_samples
    return samples


def get_utility_scores(sources, hypotheses, targets, utilities):
    scores = {

    }
    for util in utilities:
        scores[util.name] = util.call_batched(sources, hypotheses, targets)

    return scores