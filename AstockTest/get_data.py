from ruamel.yaml import YAML
import qlib
from qlib.utils import init_instance_by_config, flatten_dict

def get_yaml_config(file_path=r"AstockTest\data_set.yaml"):
    yaml = YAML(typ="safe", pure=True)
    with open(file_path, "r") as f:
        rendered_yaml = f.read()
    config = yaml.load(rendered_yaml)
    return config

def get_dataset(config):
    dataset = init_instance_by_config(config["task"]["dataset"])
    return dataset

if __name__ == "__main__":
    qlib.init(provider_uri ="~/.qlib/qlib_data/my_data_2019/", region="cn")
    config = get_yaml_config()
    dataset = get_dataset(config)
    print(config)
    #print(dataset)