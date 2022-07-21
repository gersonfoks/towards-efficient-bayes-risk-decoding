import pandas as pd

from utilities.misc import get_utility_scores
from utilities.utilities import load_utility
import numpy as np

df = pd.read_parquet('./model_predictions/beam_search_translations.parquet')

sources = df["source"].astype("str").to_list()
hypotheses = df["translations"].astype("str").to_list()
targets = df["target"].astype("str").to_list()




unigram_f1 = load_utility('unigram-f1')
comet_utility = load_utility('comet')

utilities = [
    unigram_f1,
    comet_utility
]


utility_scores = get_utility_scores(sources, hypotheses, targets, utilities)


mean_util_scores = {
    k: np.mean(v) for k,v in utility_scores.items()
}

print(mean_util_scores)

