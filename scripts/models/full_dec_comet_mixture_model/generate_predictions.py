### A simple script which we can use to train a model
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
from tqdm import tqdm

from models.MixtureModels.FullDecCometMixtureModel.FullDecCometMixtureModelManager import \
    FullDecCometMixtureModelManager
from models.MixtureModels.FullDecMixtureModel.helpers import load_test_data


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Get the predictions of a basic lstm model ')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/comet/full_dec_comet_model_gaussian_2/best/',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use

    model, model_manager = FullDecCometMixtureModelManager.load_model(args.model_path)



    # Load the dataset
    test_df, test_dataloader = load_test_data(model_manager.nmt_model, model_manager.tokenizer, smoke_test=args.smoke_test,
                                              seed=args.seed)

    outputs = {}

    outputs["locs"] = []
    outputs["scales"] = []
    outputs["logit_weights"] = []

    if model.distribution == 'student-t':
        outputs["degrees_of_freedom"] = []

    all_sources = [

    ]
    all_hypotheses = [

    ]
    indices = [

    ]

    model = model.to("cuda").eval()

    print("start gathering predictions")
    for x in tqdm(test_dataloader):

        sources, hypotheses, features, scores = x
        indices += features["source_index"]
        batch_out = model.predict(x)
        all_sources += sources
        all_hypotheses += hypotheses
        outputs["locs"] += batch_out["locs"].cpu().numpy().tolist()
        outputs["scales"] += batch_out["scales"].cpu().numpy().tolist()
        outputs["logit_weights"] += batch_out["logit_weights"].cpu().numpy().tolist()
        if model.distribution == 'student-t':
            outputs["degrees_of_freedom"] += batch_out["outputs"].cpu().numpy().tolist()
    #
    results = pd.DataFrame({
        "source": all_sources,
        "hypothesis": all_hypotheses,

        "source_index": indices,
        **outputs

    })

    grouped_result = {
        "source": [],
        "hypotheses": [],
        "locs": [],
        "scales": [],
        "logit_weights": [],
    }

    if model.distribution == 'student-t':
        grouped_result["degrees_of_freedom"] = []

    indices = []

    for i, x in test_df.iterrows():
        index = x["source_index"]

        indices.append(index)
        source = x["source"]
        temp = results[results["source_index"] == index]
        grouped_result["source"] += [source]
        grouped_result["hypotheses"].append(temp["hypothesis"].to_list())
        #grouped_result["predictions"].append(temp["prediction"].to_list())
        grouped_result["locs"].append(temp["locs"].to_list())
        grouped_result["scales"].append(temp["scales"].to_list())
        grouped_result["logit_weights"].append(temp["logit_weights"].to_list())
        if model.distribution == 'student-t':
            grouped_result["degrees_of_freedom"].append(temp["degrees_of_freedom"].to_list())


    grouped_results = pd.DataFrame(grouped_result)

    base_dir = './model_predictions/{}/'.format(args.utility)
    #
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    grouped_results.to_parquet(base_dir + '{}_predictions.parquet'.format(model.name))


if __name__ == '__main__':
    main()
