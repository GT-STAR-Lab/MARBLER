import yaml
import os
from misc import run_env, objectview
import argparse


def main():
    module_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    args = parser.parse_args()

    config_path = "config.npy"# os.path.join(module_dir, "scenarios", args.scenario, "config.npy")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    run_env(config, module_dir)

if __name__ == '__main__':
    main()