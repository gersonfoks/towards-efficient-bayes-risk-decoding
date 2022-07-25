import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR, CyclicLR


def get_optimizer_function(config):


    if config["optimizer"]["type"] == "adam":
        def initializer(x):
            lr_config = {
                "optimizer": torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"]),

            }
            return lr_config

        return initializer
    if config["optimizer"]["type"] == "inv_square_root":

        def initializer(x):

            num_warmup_steps = config["optimizer"]["warmup_steps"]

            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            plateau_steps = config["optimizer"]["plateau_steps"]
            # When to start the decay
            start_step_decay = plateau_steps + num_warmup_steps

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

    if config["optimizer"]["type"] == "adam_with_lr_decay":

        def initializer(x):

            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])
            step_size = config["optimizer"]["step_size"]
            gamma = config["optimizer"]["gamma"]
            # When to start the decay

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": StepLR(optimizer, step_size=step_size, gamma=gamma),
                    "interval": config["optimizer"]["interval"],
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

    elif config["optimizer"]["type"] == "one_cylce_lr":
        print("using one cycle lr")
        def initializer(x):
            optimizer = torch.optim.Adam(x, lr=config["lr"], weight_decay=config["weight_decay"])


            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": OneCycleLR(optimizer, max_lr=config["optimizer"]["max_lr"],
                                            steps_per_epoch=config["optimizer"]["steps_per_epoch"],
                                            epochs=config["optimizer"]["epochs"]

                                            ),
                    "interval": "step",
                }

            }

            return lr_config

        return initializer

    elif config["optimizer"]["type"] == "cyclic_lr":
        print("using cyclic rl")

        def initializer(x):
            optimizer = torch.optim.Adam(x, weight_decay=config["weight_decay"],)

            lr_config = {
                "optimizer": optimizer,
                "lr_scheduler": {

                    "scheduler": CyclicLR(optimizer,
                                          base_lr= config["optimizer"]["base_lr"],
                                          max_lr=config["optimizer"]["max_lr"],

                                          step_size_up = config["optimizer"]["step_size_up"],
                                          mode = config["optimizer"]["mode"],
                                            cycle_momentum=False,

                                          ),
                    "interval": "step",
                }

            }

            return lr_config

        return initializer
