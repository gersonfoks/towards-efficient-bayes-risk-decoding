### A simple script which we can use to train a model
import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from models.QualityEstimationStyle.BasicLstmModel.BasicLstmModelManager import BasicLstmModelManager
from models.QualityEstimationStyle.BasicLstmModel.BasicLstmTrainer import BasicLstmModelTrainer

from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelManager import LastHiddenStateModelManager
from models.QualityEstimationStyle.LastHiddenStateModel.LastHiddenStateModelTrainer import LastHiddenStateModelTrainer

from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelManager import TokenStatisticsModelManager
from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelTrainer import TokenStatisticsModelTrainer
from utilities.config.ConfigParser import ConfigParser
from utilities.dataset.loading import load_dataset_for_training
from utilities.factories.CollatorFactory import CollatorFactory
from utilities.factories.PreprocessFactory import PreprocessFactory
from utilities.misc import load_nmt_model
from utilities.preprocess.TablePreprocessor import TokenStatisticsLookupTableCreator
from utilities.wrappers.NmtWrapper import NMTWrapper


def get_dataloader(config, model_config, smoke_test=False):
    n_hypotheses = 100
    n_references = 1000
    sampling_method = 'ancestral'
    utility = 'comet'

    # Load the dataset
    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", n_hypotheses, n_references,
                                                       sampling_method, utility, )
    validation_df = validation_dataset_loader.load(type="pandas").data
    if smoke_test:
        validation_df = validation_df.iloc[:100]

    # Need this for later
    validation_df["source_index"] = validation_df.index


    # Preprocess
    preprocess_factory = PreprocessFactory(config["preprocess"])

    preprocessor = preprocess_factory.get_preprocessor()

    preprocessed_val_df = preprocessor(validation_df)

    val_dataset = Dataset.from_pandas(preprocessed_val_df)
    nmt_model, tokenizer = load_nmt_model(model_config["nmt_model"], pretrained=True)
    wrapped_nmt_model = NMTWrapper(nmt_model, tokenizer)
    # Get the table



    # Load the nmt model


    collator_factory = CollatorFactory(config["collator"]
                                       ,
                                       wrapped_nmt_model=wrapped_nmt_model,

                                       )
    train_collator, val_collator = collator_factory.get_collators()

    val_dataloader = DataLoader(val_dataset,
                                collate_fn=val_collator,
                                batch_size=64, shuffle=False, )

    return validation_df, val_dataloader


def get_predictions(model, val_dataloader, val_df):
    predictions = [

    ]
    all_sources = [

    ]
    all_hypotheses = [

    ]
    indices = [

    ]

    model = model.to("cuda").eval()
    for x in tqdm(val_dataloader):
        sources, hypotheses, features, scores = x
        indices += features["index"]
        batch_out = model.predict(x)
        all_sources += sources
        all_hypotheses += hypotheses
        predictions += batch_out["predictions"].cpu().numpy().tolist()

    results = pd.DataFrame({
        "source": all_sources,
        "hypothesis": all_hypotheses,
        "prediction": predictions,
        "source_index": indices

    })



    grouped_result = {
        "source": [],
        "hypotheses": [],
        "predictions": [],
    }
    for i, x in val_df.iterrows():
        index = x["source_index"]
        source = x["source"]
        temp = results[results["source_index"] == index]
        grouped_result["source"] += [source]
        grouped_result["hypotheses"].append(temp["hypothesis"].to_list())
        grouped_result["predictions"].append(temp["prediction"].to_list())

    grouped_results = pd.DataFrame(grouped_result)

    return grouped_results



def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--model-path', type=str,
                        default='./saved_models/last_hidden_state_attention_comet/best/',
                        help='config to load model from')

    parser.add_argument('--config', type=str,
                        default='./configs/last_hidden_state_attention.yml',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')


    args = parser.parse_args()

    # Load the model
    pl_path = args.model_path + 'pl_model.pt'
    checkpoint = torch.load(pl_path)
    model_config = checkpoint["config"]

    model_config["utility"] = "comet"

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_parser = ConfigParser("comet")
    config = config_parser.parse(config)


    # Instantiate the model manager and the model
    model, manager = LastHiddenStateModelManager.load_model(args.model_path)



    val_df, val_dataloader = get_dataloader(config, model_config, smoke_test=args.smoke_test)

    predictions = get_predictions(model, val_dataloader, val_df)

    base_dir = './model_predictions/'

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(base_dir + '{}_predictions.parquet'.format(config["model_name"]))


if __name__ == '__main__':
    main()
