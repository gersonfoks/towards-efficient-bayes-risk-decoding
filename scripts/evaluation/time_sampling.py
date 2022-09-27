import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch

from utilities.misc import load_nmt_model, translate, batch_sample, batch


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--sampling-method', type=str,

                        default='ancestral',
                        help='which sampling method to use')

    parser.add_argument('--n-sources', type=int,

                        default=300,
                        help='how many sources we use for calculating the speed')



    config = {
        'name': 'Helsinki-NLP/opus-mt-de-en',
        'checkpoint': './saved_models/NMT/de-en-model/',
        'type': 'MarianMT',

    }




    args = parser.parse_args()
    df = pd.read_parquet('./data/tatoeba_splits/test.parquet')[:args.n_sources]

    df["source"] = df[["translation"]].apply(lambda x: x["translation"]["de"], axis=1)
    df["target"] = df[["translation"]].apply(lambda x: x["translation"]["en"], axis=1)

    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    nmt_model = nmt_model.to('cuda').eval()
    sources = df["source"].astype("str").to_list()


    results = {

    }
    ms = [
        1,2,3,4,5,10,25,50,100, 1000
    ]
    with torch.no_grad():
        for n in ms:
            times = []
            for s in batch(sources, 1):
                start_time = time()
                batch_sample(nmt_model, tokenizer, s, n_samples=n ,sampling_method=args.sampling_method)
                time_diff = time() - start_time
                times.append(time_diff)
            print(args.n_sources)
            print('mean_time_{}: {}'.format(n, np.mean(times)))

            results[n] = np.mean(times)

    print(results)


    result_ref = "./results/reference_generation.json"


    with open(result_ref, 'w') as f:
        json.dump(results, f)





if __name__ == '__main__':
    main()
