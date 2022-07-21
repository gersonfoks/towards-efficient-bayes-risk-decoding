from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader
from utilities.misc import load_nmt_model, translate

config = {
    'name': 'Helsinki-NLP/opus-mt-de-en',
    'checkpoint': 'NMT/tatoeba-de-en/model',
    'type': 'MarianMT',

}

nmt_model, tokenizer = load_nmt_model(config, pretrained=True)

dataset_dir = 'predictive/tatoeba-de-en/data/raw/'
dataset_loader = BayesRiskDatasetLoader('validation_predictive', 100, 1000,
                                        'ancestral', 'comet', develop=False,
                                        base=dataset_dir)


dataset = dataset_loader.load('pandas').data

nmt_model = nmt_model.to('cuda')
sources = dataset["source"].astype("str").to_list()

print(list(dataset.columns))

dataset["translations"] = translate(nmt_model, tokenizer, sources, method='beam')

dataset.to_parquet('./model_predictions/beam_search_translations.parquet')
