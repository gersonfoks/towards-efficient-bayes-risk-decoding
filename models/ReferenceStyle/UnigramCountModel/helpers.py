import pandas as pd
from torch.utils.data import DataLoader
from collators.UnigramCountCollator import UnigramCountCollator
from custom_datasets.BayesRiskDataset import BayesRiskDataset
from utilities.misc import load_bayes_risk_dataframe

import numpy as np


def prepare_dataframe(df,):
    '''
    Prepares a dataframe such that it can be used to train the model
    :param df: dataframe to prepare

    :return:
    '''

    df.reset_index(inplace=True)
    df = df.explode(column=['hypotheses', 'utilities', ])
    df.reset_index(drop=True, inplace=True)
    df.rename({"hypotheses": "hypothesis"}, inplace=True, axis=1)

    df["utility"] = df[["utilities", 'references_count']].apply(lambda x: np.sum(
        np.array(x["utilities"]) * np.array(x["references_count"]) / np.sum(np.array(x["references_count"]))), axis=1)

    df["probs"] = df[['references_count']].apply(lambda x:
                                                 np.array(np.array(x["references_count"]) / np.sum(
                                                     np.array(x["references_count"]))), axis=1)

    return df


def load_data(config, nmt_model, tokenizer, seed=0, smoke_test=False, utility='unigram-f1', n_model_references=5):
    print("Preparing the data")

    train_df = load_bayes_risk_dataframe(config["dataset"]["sampling_method"],
                                         config["dataset"]["n_hypotheses"],
                                         config["dataset"]["n_references"],
                                         'train_predictive',
                                         seed=seed,
                                         smoke_test=smoke_test,
                                         utility=utility
                                         )
    # We need to add the references back
    references_file = './data/{}/{}_{}_train_predictive_{}_references'.format(utility,
                                                                   config["dataset"]["sampling_method"],
                                                                   config["dataset"]["n_references"],
                                                                   seed
                                                                   )
    if smoke_test:
        references_file += '_smoke_test'
    references_file += '.parquet'
    train_df_references = pd.read_parquet(references_file)


    # Construct the table

    train_ref_table = {
        r["index"]: {
            "references_count": r["references_count"],

            "references": r["references"],
        } for _, r in train_df_references.iterrows()
    }



    val_df = load_bayes_risk_dataframe(config["dataset"]["sampling_method"],
                                       config["dataset"]["n_hypotheses"],
                                       config["dataset"]["n_references"],
                                       'validation_predictive',
                                       seed=seed,
                                       smoke_test=smoke_test,
                                       utility=utility
                                       )

    references_file = './data/{}/{}_{}_validation_predictive_{}_references'.format(utility,
                                                                        config["dataset"]["sampling_method"],
                                                                        config["dataset"]["n_references"],
                                                                        seed
                                                                        )
    if smoke_test:
        references_file += '_smoke_test'
    references_file += '.parquet'
    val_df_references = pd.read_parquet(references_file)
    val_ref_table = {
        r["index"]: {
            "references_count": r["references_count"],

            "references": r["references"],
        } for _, r in val_df_references.iterrows()
    }


    # We need to add the references back
    train_df = prepare_dataframe(train_df,)
    val_df = prepare_dataframe(val_df,)

    train_dataset = BayesRiskDataset(train_df)
    val_dataset = BayesRiskDataset(val_df)

    train_collator = UnigramCountCollator(train_ref_table, tokenizer, n_model_references=n_model_references)
    val_collator = UnigramCountCollator(val_ref_table, tokenizer, n_model_references=n_model_references)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=train_collator,
                                  batch_size=config["batch_size"], shuffle=True, )
    val_dataloader = DataLoader(val_dataset,
                                collate_fn=val_collator,
                                batch_size=config["batch_size"], shuffle=False, )

    return train_dataloader, val_dataloader


def load_test_data(nmt_model, tokenizer, utility="comet", seed=0, smoke_test=False, n_model_references=5):
    print("Preparing the data")
    test_df = load_bayes_risk_dataframe("ancestral",
                                        100,
                                        1000,
                                        'test',
                                        seed=seed,
                                        smoke_test=smoke_test,
                                        utility=utility
                                        )

    references_file = './data/{}/{}_{}_test_{}_references'.format(utility,
                                                                              'ancestral',
                                                                              1000,
                                                                              seed
                                                                              )
    if smoke_test:
        references_file += '_smoke_test'
    references_file += '.parquet'
    test_df_references = pd.read_parquet(references_file)

    # Construct the table

    test_ref_table = {
        r["index"]: {
            "references_count": r["references_count"],

            "references": r["references"],
        } for _, r in test_df_references.iterrows()
    }

    # Add the index
    test_df = test_df.reset_index()
    test_df["source_index"] = test_df["index"]

    temp = prepare_dataframe(test_df)

    test_dataset = BayesRiskDataset(temp)

    collator = UnigramCountCollator(test_ref_table, tokenizer, n_model_references=n_model_references, include_source_id=True)

    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=collator,
                                 batch_size=32, shuffle=False, )

    return test_df, test_dataloader


def load_data_for_timing(nmt_model, tokenizer, seed=0, smoke_test=False, n_model_references=1, n_sources=100):
    print("Preparing the data")
    test_df = load_bayes_risk_dataframe("ancestral",
                                         100,
                                         1000,
                                         'test',
                                         seed=seed,
                                         smoke_test=smoke_test,
                                        utility='comet'
                                         )[:n_sources]

    references_file = './data/{}/{}_{}_test_{}_references'.format('comet',
                                                                  'ancestral',
                                                                  1000,
                                                                  seed
                                                                  )
    if smoke_test:
        references_file += '_smoke_test'
    references_file += '.parquet'
    test_df_references = pd.read_parquet(references_file)[:n_sources]

    # Construct the table

    test_ref_table = {
        r["index"]: {
            "references_count": r["references_count"],

            "references": r["references"],
        } for _, r in test_df_references.iterrows()
    }

    test_df = test_df.reset_index()
    test_df["source_index"] = test_df["index"]

    test_df = prepare_dataframe(test_df)

    test_dataset = BayesRiskDataset(test_df)

    collator = UnigramCountCollator(test_ref_table, tokenizer, n_model_references=100, include_source_id=True)

    test_dataloader = DataLoader(test_dataset,
                                  collate_fn=collator,
                                  batch_size=1, shuffle=False, )


    return test_dataloader