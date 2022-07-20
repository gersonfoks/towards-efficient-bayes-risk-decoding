import os
import shutil
from distutils.dir_util import copy_tree

import pytorch_lightning as pl

from pytorch_lightning.callbacks import Callback


class CustomSaveCallback(Callback):

    def __init__(self, manager, save_location, target_value="val_loss", keep_top=3):
        self.save_location = save_location
        self.manager = manager
        # Keep track of scores
        self.scores_model_pairs = []
        self.target_value = target_value
        self.keep_top = keep_top
        self.best_score = None

        self.is_first = True


    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Skip the sanity check
        if self.is_first:
            self.is_first = False
        else:
            if self.target_value in trainer.callback_metrics.keys():
                epoch_number = trainer.current_epoch
                self.scores_model_pairs.append((trainer.callback_metrics[self.target_value], epoch_number))

                if len(self.scores_model_pairs) <= self.keep_top:
                    self.save_current_epoch(epoch_number)

                else:

                    sorted_scores = sorted(self.scores_model_pairs, key=lambda x: x[0])


                    top_epochs = [s[1] for s in sorted_scores[:self.keep_top]]
                    # Save the model if it is in the top 3

                    if epoch_number in top_epochs:
                        self.save_current_epoch(epoch_number)

                    # Delete other models
                    losers = [s[1] for s in sorted_scores[self.keep_top+1:]]



                    for loser in losers:
                        save_location = self.save_location + '{}/'.format(loser)
                        if os.path.exists(save_location) and os.path.isdir(save_location):
                            shutil.rmtree(save_location)



    def save_current_epoch(self, epoch_number):
        save_location = self.save_location + '{}/'.format(epoch_number)

        self.manager.save_model(save_location)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        sorted_scores = sorted(self.scores_model_pairs, key=lambda x: x[0])

        self.best_score = sorted_scores[0][0]
        print(sorted_scores)
        text_location = self.save_location + "overview.txt"
        with open(text_location, 'w+') as f:
            f.write(str(sorted_scores))


        best_location = self.save_location + '{}/'.format(sorted_scores[0][1])

        copy_tree(best_location,  self.save_location + 'best/')