
import numpy as np


from models.old.hypothesis_only_models.ProbEntropyModel.preprocess import ProbEntropyModelPreprocess


def calc_score(x):
    utils = np.array(x["utilities"])
    utilities_count = np.array(x["utilities_count"])
    score = float(np.sum(utils * utilities_count) / np.sum(utilities_count))
    return score


ProbEntropyModelPreprocessV2 = ProbEntropyModelPreprocess
