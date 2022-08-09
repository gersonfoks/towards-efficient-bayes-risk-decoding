import argparse
from collections import Counter

import numpy as np
import pandas as pd
import pytorch_lightning
import yaml
from datasets import Dataset

from torch.utils.data import DataLoader

import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq

from utilities.misc import load_nmt_model, load_tatoeba_dataframe, preprocess_tokenize, batch_sample


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Generate samples for a given split')
    parser.add_argument('--config', type=str, default='./configs/nmt/helsinki-de-en.yml',
                        help='config to load model from')
    parser.add_argument('--n-samples', type=int, default=10, help='number of references for each source')
    parser.add_argument('--split', type=str, default="validation_predictive",
                        help="Which split to generate samples for (train_predictive, validation_predictive or test")
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")
    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true uses only 100 sources for fast development')

    parser.set_defaults(smoke_test=False)

    parser.add_argument('--sampling-method', type=str, default="ancestral", help='sampling method for the hypothesis')

    args = parser.parse_args()

    # Add seeding for reproducability
    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print("Loading model")
    nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

    model = nmt_model.to("cuda")
    model.eval()



    save_file_base = './data/samples/{}_{}_{}_{}'.format(args.sampling_method, args.n_samples, args.split, args.seed)
    if args.smoke_test:
        save_file_base += '_smoke_test'
    save_file = save_file_base + '.parquet'


    preprocess_function = lambda x: preprocess_tokenize(x, tokenizer)

    dataframe = load_tatoeba_dataframe(args.split)

    if args.smoke_test:
        dataframe = dataframe.loc[:100]
    dataset = Dataset.from_pandas(dataframe)
    dataset = dataset.map(
        preprocess_function, batched=True)




    data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer,
                                           padding=True, return_tensors="pt")

    keys = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    def collate_fn(batch):
        new_batch = [{k: s[k] for k in keys} for s in batch]
        x_new = data_collator(new_batch)

        sources = [s["translation"]['de'] for s in batch]
        targets = [s["translation"]['en'] for s in batch]

        return x_new, (sources, targets)




    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1, shuffle=False)
    result = {
        "source": [],
        "target": [],
        "samples": [],
        "sample_count": []
    }
    with torch.no_grad():

        for i, (x, (source, target)) in tqdm(enumerate(dataloader), total=len(dataloader), ):
            sample = batch_sample(model, tokenizer, source, n_samples=args.n_samples, batch_size=250, sampling_method=args.sampling_method )

            # Add each sample

            counter_samples = dict(Counter(sample))

            result["source"] += source
            result["target"] += target
            samples = []
            sample_count =[]
            for key, val in counter_samples.items():
                samples.append(key)
                sample_count.append(val)

            result["samples"].append(samples)
            result["sample_count"].append(sample_count)


    df = pd.DataFrame.from_dict(result)
    print("Saving to {}".format(save_file))
    df.to_parquet(save_file)

    print("Done!")


if __name__ == '__main__':
    main()
