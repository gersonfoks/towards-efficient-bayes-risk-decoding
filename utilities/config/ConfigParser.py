import yaml


class ConfigParser:

    def __init__(self, utility, default_config_ref='./configs/defaults.yml'):
        self.utility = utility
        with open(default_config_ref, "r") as file:
            self.defaul_config = yaml.load(file, Loader=yaml.FullLoader)


    def parse(self, config):


        # Fill in the default values
        config = self.fill_in_defaults(config)

        # Add log dir and save dir
        config["log_dir"] = './logs/{}_{}/'.format(config["model_name"], self.utility)
        config["save_model_path"] = './saved_models/{}_{}/'.format(config["model_name"], self.utility)

        # Add utility
        config["dataset"]["utility"] = self.utility

        ### Add actiction function last layer
        if 'activation_function_last_layer' not in config["model"]["feed_forward_layers"].keys():

            if self.utility == "comet":
                config["model"]["feed_forward_layers"]['activation_function_last_layer'] = 'tanh'
                config["model"]["feed_forward_layers"]['last_layer_scale'] = 5 # Min and max values are in practice between -5 and 5
            else:
                config["model"]["feed_forward_layers"]['activation_function_last_layer'] = None
                config["model"]["feed_forward_layers"]['last_layer_scale'] = 0


        return config


    def fill_in_defaults(self, config):
        for key, item in self.defaul_config.items():
            if key not in config.keys():
                config[key] = item
        return config

