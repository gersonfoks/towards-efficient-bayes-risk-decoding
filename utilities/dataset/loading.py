from custom_datasets.BayesRiskDataset.BayesRiskDatasetLoader import BayesRiskDatasetLoader


def load_dataset_for_training(dataset_config, smoke_test=False):
    '''
    Loads the dataset into a pandas dataframe
    :param dataset_config:
    :return:
    '''
    n_hypotheses = dataset_config["n_hypotheses"]
    n_references = dataset_config["n_references"]
    utility = dataset_config["utility"]
    sampling_method = dataset_config["sampling_method"]

    dataset_dir = dataset_config["dir"]
    train_dataset_loader = BayesRiskDatasetLoader("train_predictive", n_hypotheses, n_references,
                                                  sampling_method, utility, develop=False,
                                                  base=dataset_dir)

    validation_dataset_loader = BayesRiskDatasetLoader("validation_predictive", n_hypotheses, n_references,
                                                       sampling_method, utility, develop=False,
                                                       base=dataset_dir)

    if smoke_test:
        train_dataset = train_dataset_loader.load(type="pandas").data.iloc[:16].copy()
        validation_dataset = validation_dataset_loader.load(type="pandas").data.iloc[:16].copy()

    else:
        train_dataset = train_dataset_loader.load(type="pandas").data
        validation_dataset = validation_dataset_loader.load(type="pandas").data


    return train_dataset, validation_dataset