import yaml
import os
from robotarium_gym.utilities.misc import run_env, objectview
import argparse


def main():
    module_dir = os.path.dirname(__file__)
    if module_dir.split("/")[-1] != "robotarium_gym":
        module_dir = ""
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save the images. Show fig must be true. Set to None not to save')
    args = parser.parse_args()

    if module_dir == "":
        config_path = "config.yaml"
    else:
        config_path = os.path.join(module_dir, "scenarios", args.scenario, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    run_env(config, module_dir, args.save_dir)

if __name__ == '__main__':
    main()