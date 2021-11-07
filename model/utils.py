import yaml

def read_config(config_path):
    with open(config_path, "r") as stream:
        config = yaml.full_load(stream)

    return config

