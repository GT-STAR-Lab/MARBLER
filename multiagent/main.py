import yaml
import os
from multiagent.robotarium_env.utilities import run_env, objectview
import argparse


def main():
    module_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='pcp', help='scenario name')
    args = parser.parse_args()

    config_path = os.path.join(module_dir, "scenarios", args.scenario, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = objectview(config)

    run_env(config, module_dir)

if __name__ == '__main__':
    main()