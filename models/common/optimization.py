import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR


def get_optimizer_function(config):

    if config["optimizer"]["type"] == "adam":
        def initializer(x):
            lr_config = {
                "optimizer": torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"]),

            }
            return lr_config

        return initializer
    if config["optimizer"]["type"] == "adam_with_warmup":

        def initializer(x):

            num_warmup_steps = config["optimizer"]["warmup_steps"]

            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])

            # When to start the decay
            start_step_decay = config["optimizer"]["start_decay"]

            def lr_lambda(current_step: int):

                if current_step <= num_warmup_steps:
                    return current_step / num_warmup_steps
                # Waiting a number of steps before decaying
                elif current_step <= start_step_decay:
                    return 1.0
                else:
                    return (current_step - start_step_decay) ** (-0.5)

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": LambdaLR(optimizer, lr_lambda),
                    "interval": "step",
                }

            }

            return lr_config
        return initializer

    if config["optimizer"]["type"] == "adam_with_steps":

        def initializer(x):


            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            step_size = config["optimizer"]["step_size"]
            gamma = config["optimizer"]["gamma"]
            # When to start the decay

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": StepLR(optimizer, step_size=step_size, gamma=gamma),
                    "interval": "epoch",
                }

            }

            return lr_config

        return initializer

    if config["optimizer"]["type"] == "adam_reduce_every_n_steps":
        def initializer(x):
            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            step_size = config["optimizer"]["step_size"]
            gamma = config["optimizer"]["gamma"]
            # When to start the decay

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": StepLR(optimizer, step_size=step_size, gamma=gamma),
                    "interval": "step",
                }

            }

            return lr_config

        return initializer
