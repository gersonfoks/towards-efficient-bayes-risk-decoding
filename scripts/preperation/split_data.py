
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Load the dataset

seed = 1

dataset = load_dataset("tatoeba", lang1="de", lang2="en", )["train"]

n_samples = len(dataset)

train_MLE_size = int(0.9 * n_samples)
evaluation_MLE_size = 2500
evaluation_predictive_size = 2500
test_size = 2500
# This leaves about 7.5% for training the predictive model.
train_predictive_size = n_samples - train_MLE_size - evaluation_predictive_size - evaluation_MLE_size - test_size



indices = np.arange(stop=len(dataset))
np.random.seed(seed)
np.random.shuffle(indices)

train_MLE_end = train_MLE_size
train_predictive_end = train_MLE_end + train_predictive_size
validation_MLE_end = evaluation_MLE_size + train_predictive_end
evaluation_predictive_end = evaluation_predictive_size + validation_MLE_end

train_MLE_indices = indices[:train_MLE_end]
train_predictive_indices = indices[train_MLE_end:train_predictive_end]
validation_MLE_indices = indices[train_predictive_end: validation_MLE_end]
validation_predictive_indices = indices[validation_MLE_end: evaluation_predictive_end]
test_indices = indices[evaluation_predictive_end:]

# safe the splits

splits = {
    "train_NMT": train_MLE_indices,
    "train_predictive": train_predictive_indices,
    "validation_NMT": validation_MLE_indices,
    "validation_predictive": validation_predictive_indices,
    "test": test_indices
}

Path('./data').mkdir(parents=True, exist_ok=True)

with open("./data/splits_tatoeba.pkl", "wb") as f:
    pickle.dump(splits, f)

with open("./data/splits_tatoeba.pkl", "rb") as f:
    splits = pickle.load(f)
    print(splits)


# Save the dataset
df = dataset.to_pandas()

Path('./data/tatoeba_splits/').mkdir(parents=True, exist_ok=True)
for split_name, indices in splits.items():
    temp_df = df.loc[indices]
    temp_df.reset_index(inplace=True, drop=True)
    temp_df.drop(["id"], axis=1, inplace=True)

    temp_df.to_parquet('./data/tatoeba_splits/{}.parquet'.format(split_name))
