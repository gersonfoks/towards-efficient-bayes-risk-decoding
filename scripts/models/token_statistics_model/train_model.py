'''
File to train the basic model
'''
import argparse

import numpy as np
import pytorch_lightning
import yaml

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from models.QualityEstimationStyle.LastHiddenStateModel.helpers import load_data
from models.QualityEstimationStyle.TokenStatisticsModel.TokenStatisticsModelManager import TokenStatisticsModelManager
from utilities.callbacks import CustomSaveCallback

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='Train a model according with parameters specified in the config file ')
    parser.add_argument('--config', type=str,
                        default='./configs/predictive/token_statistics_model.yml',
                        help='config to load model from')

    parser.add_argument('--smoke-test', dest='smoke_test', action="store_true",
                        help='If true does a small test run to check if everything works')
    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    parser.set_defaults(smoke_test=False)

    args = parser.parse_args()

    np.random.seed(args.seed)
    pytorch_lightning.seed_everything(args.seed)

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    smoke_test = args.smoke_test

    # We first load the model as the model also has the tokenizer that we want to use

    model_manager = TokenStatisticsModelManager(config["model"])

    model = model_manager.create_model()

    tokenizer = model_manager.tokenizer
    nmt_model = model_manager.nmt_model

    ### First load the dataset
    # We can use the same dataloader as for the last hidden state model
    train_dataloader, val_dataloader = load_data(config, nmt_model, tokenizer, seed=args.seed, smoke_test= args.smoke_test)


    ### Train

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config["log_dir"])
    max_epochs = 10 if smoke_test else config["max_epochs"]
    custom_save_model_callback = CustomSaveCallback(model_manager, config["save_model_path"])

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        progress_bar_refresh_rate=1,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[LearningRateMonitor(logging_interval="step"),
                   EarlyStopping("val_loss", patience=config["patience"], verbose=True, ),
                   custom_save_model_callback],
        logger=tb_logger,
        accumulate_grad_batches=config["accumulate_grad_batches"],

    )

    # Fit the model
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader, )

    ### Evaluate
    trainer.validate(model, val_dataloader, )

if __name__ == '__main__':
    main()
