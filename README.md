# Towards Efficient bayes risk decoding

This project contains the code for "Towards Efficient Bayes Risk Decoding".

To reduce the cost of reproducibilty
The models and data used can be found at:
...

If you want to reproduce from scratch, follow these steps:

#### Splitting the data and training the nmt model

1. Create a split of the data by running: `python -m scripts.preperation.split_data`
2. Train the NMT model by running: `python -m scripts.preperation.train_nmt_model` (WARNING: training this model costs
   about 5 days, it is adviced to use the one provided)

#### Generating the data for the predictive model

3. Then generate samples by running `python -m scripts.generate_samples --n-samples=m --name="name"` with m being any
   number
   you like, use name to identify your samples (we may want to use independent samples so we need to be able to identify
   them)
4. Generate the bayes risk for different datasets by
   running: `python -m scripts.create_bayes_risk_dataset --hyp-data=<dataset ref> --ref-data=<dataset ref>`

#### Training the models

In the scripts.model module we have scripts for training the models

simply run:
`python -m scripts.models.<model_name>.train_model` to train that model
These scripts read a config and start training that model.
See the configs and files for details about what is used during training.

#### Hyperparameter search

For the hyperparameter search run:
`python -m scripts.models.<model_name>.hyperparam_search`

#### Evaluation.

To save computational resources we first generate predictions from each model and store those.

Run (TODO insert)




For evaluation a couple of notebooks are provided as they give immediate visual inspection.

See the directory notebooks for this.










