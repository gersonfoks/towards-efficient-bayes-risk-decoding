import argparse
from pathlib import Path

import pandas as pd

from utilities.misc import load_nmt_model, translate



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--sampling-method', type=str,

                        default='beam',
                        help='config to load model from')
    config = {
        'name': 'Helsinki-NLP/opus-mt-de-en',
        'checkpoint': './saved_models/NMT/de-en-model/',
        'type': 'MarianMT',

    }


    df = pd.read_parquet('./data/tatoeba_splits/test.parquet')

    args = parser.parse_args()

    df["source"] = df[["translation"]].apply(lambda x: x["translation"]["de"], axis=1)
    df["target"] = df[["translation"]].apply(lambda x: x["translation"]["en"], axis=1)

    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    nmt_model = nmt_model.to('cuda')
    sources = df["source"].astype("str").to_list()



    df["translations"] = translate(nmt_model, tokenizer, sources, method=args.sampling_method)


    print(df)

    base_dir = "./model_predictions/nmt_outputs/"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_dir + '{}_translations.parquet'.format(args.sampling_method))




if __name__ == '__main__':
    main()
