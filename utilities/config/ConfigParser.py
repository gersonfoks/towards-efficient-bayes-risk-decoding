import yaml


class ConfigParser:

    def __init__(self, utility, default_config_ref='./configs/defaults.yml'):
        self.utility = utility
        with open(default_config_ref, "r") as file:
            self.defaul_config = yaml.load(file, Loader=yaml.FullLoader)


    def parse(self, config):




        # Add log dir and save dir
        config["log_dir"] = './logs/{}_{}/'.format(config["model_name"], self.utility)
        config["save_model_path"] = './saved_models/{}_{}/'.format(config["model_name"], self.utility)

        # Add utility
        config["dataset"]["utility"] = self.utility


        return config


