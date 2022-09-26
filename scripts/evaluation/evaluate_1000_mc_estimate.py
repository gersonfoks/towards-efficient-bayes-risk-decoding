import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint

from utilities.Utility import NGramF, ChrF, BleurtUtility
from utilities.misc import map_to_utility

from utilities.wrappers.CometWrapper import CometWrapper


def get_best_translation(x):



    max_idx = np.argmax(x["utility"])

    return x["hypotheses"][max_idx]




def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--utility', type=str,
                        default='unigram-f1',
                        help='utility to use')
    args = parser.parse_args()
    df = pd.read_parquet('./data/{}/ancestral_100_1000_test_0.parquet'.format(args.utility))



    print(df)

    print(list(df.columns))
    df["utility"] = df[["utilities", 'references_count']].apply(map_to_utility, axis=1)
    df["translations"] = df[['hypotheses', 'utility']].apply(get_best_translation, axis=1)

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

    chrf_scores = utility.evaluate(sources, translations, targets)

    results["chrf_mean"] = np.mean(chrf_scores)
    results["chrf_std"] = np.std(chrf_scores)

    bleurt_metric = BleurtUtility()

    bleurt_scores = bleurt_metric.evaluate(sources, translations, targets)


    results["bleurt_mean"] = np.mean(bleurt_scores)



    result_ref = './results/{}/'.format(args.utility)

    Path(result_ref).mkdir(parents=True, exist_ok=True)
    summary_ref = result_ref + '1000_mc_summary.json'

    with open(summary_ref, 'w') as fp:
        json.dump(results, fp)


if __name__ == '__main__':
    main()
