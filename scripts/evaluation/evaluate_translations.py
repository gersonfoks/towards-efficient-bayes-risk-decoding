import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint

from utilities.Utility import NGramF, ChrF
from utilities.misc import load_nmt_model, translate
from utilities.wrappers.CometWrapper import CometWrapper


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--sampling-method', type=str,

                        default='beam',
                        help='config to load model from')
    args = parser.parse_args()
    df = pd.read_parquet('./model_predictions/nmt_outputs/{}_translations.parquet'.format(args.sampling_method))


    results = {

    }

    sources = df["source"].to_list()
    translations = df["translations"].to_list()
    targets = df["target"].to_list()

    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    model.to("cuda")
    model.eval()
    wrapped_model = CometWrapper(model)

    comet_scores = wrapped_model.batch_predict(sources, translations, targets)

    results["comet_mean"] = np.mean(comet_scores)
    results["comet_std"] = np.std(comet_scores)

    utility = NGramF(n=1)

    unigram_f1_scores = utility.evaluate(sources, translations, targets)



    results["unigram_f1_mean"] = np.mean(unigram_f1_scores)
    results["unigram_f1_std"] = np.std(unigram_f1_scores)

    utility = ChrF()

    unigram_f1_scores = utility.evaluate(sources, translations, targets)

    results["chrf_mean"] = np.mean(unigram_f1_scores)
    results["chrf_std"] = np.std(unigram_f1_scores)

    result_ref = './results/{}/'.format(args.sampling_method)

    Path(result_ref).mkdir(parents=True, exist_ok=True)
    summary_ref = result_ref + 'summary.json'

    with open(summary_ref, 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    main()
