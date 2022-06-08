import os
import shutil

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback

from utilities.PathManager import get_path_manager


class CustomSaveCallback(Callback):

    def __init__(self, manager, save_location, target_value="val_loss", keep_top=3):
        self.save_location = save_location
        self.manager = manager
        # Keep track of scores
        self.scores_model_pairs = []
        self.target_value = target_value
        self.keep_top = keep_top
        self.path_manager = get_path_manager()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        if self.target_value in trainer.callback_metrics.keys():
            epoch_number = trainer.current_epoch
            self.scores_model_pairs.append((trainer.callback_metrics[self.target_value], epoch_number))

            if len(self.scores_model_pairs) < self.keep_top:
                self.save_current_epoch(epoch_number)

            else:

                sorted_scores = sorted(self.scores_model_pairs, key=lambda x: x[0])

                top_epochs = [s[1] for s in sorted_scores[:self.keep_top]]
                # Save the model if it is in the top 3

                if epoch_number in top_epochs:
                    self.save_current_epoch(epoch_number)

                # Delete other models
                losers = [s[1] for s in sorted_scores[self.keep_top:]]

                for loser in losers:
                    save_location = self.save_location + '{}/'.format(loser)
                    save_location = self.path_manager.get_abs_path(save_location)
                    if os.path.exists(save_location) and os.path.isdir(save_location):
                        shutil.rmtree(save_location)

    def save_current_epoch(self, epoch_number):
        save_location = self.path_manager.get_abs_path(self.save_location + '{}/'.format(epoch_number))

        self.manager.save_model(save_location)
