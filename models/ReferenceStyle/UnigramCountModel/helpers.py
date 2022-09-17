import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

from collators.BasicReferenceCollator import BasicReferenceCollator
from collators.UnigramCountCollator import UnigramCountCollator
from custom_datasets.BayesRiskDataset import BayesRiskDataset
from utilities.misc import load_bayes_risk_dataframe
from utilities.preprocessing import SourceTokenizer, TargetTokenizer
import numpy as np


def prepare_dataframe(df, tokenizer):
    '''
    Prepares a dataframe such that it can be used to train the model
    :param df: dataframe to prepare
    :param tokenizer: tokenizer used to tokenize the source and hypothesis
    :return:
    '''
    dataset = Dataset.from_pandas(df)
    # 1) Tokenize the dataset

    source_tokenize = SourceTokenizer(tokenizer)

    dataset = dataset.map(source_tokenize, batched=True)

    df = dataset.to_pandas()
    df.reset_index(inplace=True)
    df = df.explode(column=['hypotheses', 'utilities', ])
    df.reset_index(drop=True, inplace=True)
    df.rename({"hypotheses": "hypothesis"}, inplace=True, axis=1)

    dataset = Dataset.from_pandas(df)

    target_tokenize = TargetTokenizer(tokenizer)

    dataset = dataset.map(target_tokenize, batched=True)
    df = dataset.to_pandas()

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
            "utilities": r["utilities"][0],
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
            "utilities": r["utilities"][0],
            "references": r["references"],
        } for _, r in val_df_references.iterrows()
    }


    # We need to add the references back
    train_df = prepare_dataframe(train_df, tokenizer)
    val_df = prepare_dataframe(val_df, tokenizer)

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
