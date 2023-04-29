import os
import time
import argparse
import shutil
import robotarium_gym.main
import robotarium_gym.utilities.roboEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--out_dir', type=str, default = '')
    args = parser.parse_args()

    if args.out_dir == '':
        args.out_dir = f"robotarium_submission{int(time.time())}"
    os.mkdir(args.out_dir)

    base_path = os.path.dirname(robotarium_gym.main.__file__)
    possible_imports = []

    #Copy main and base
    shutil.copy(f'{base_path}/main.py', args.out_dir)
    shutil.copy(f'{base_path}/scenarios/base.py', args.out_dir)
    possible_imports.append(f"robotarium_gym.scenarios.base")

    #Copy utilities
    utils = os.listdir(f'{base_path}/utilities')
    utils.remove("__pycache__")
    for f in utils:
        shutil.copy(f'{base_path}/utilities/{f}', args.out_dir)
        if '.py' in f:
            possible_imports.append(f"robotarium_gym.utilities.{f[:-3]}")

    #Copy Scenario base files
    scenarios = os.listdir(f'{base_path}/scenarios/{args.scenario}')
    scenarios.remove("__pycache__")
    scenarios.remove("models")
    for f in scenarios:
        shutil.copy(f'{base_path}/scenarios/{args.scenario}/{f}', args.out_dir)
        if '.py' in f:
            possible_imports.append(f"robotarium_gym.scenarios.{args.scenario}.{f[:-3]}")

    #Copy Scenario models
    models = os.listdir(f'{base_path}/scenarios/{args.scenario}/models')
    for f in models:
        shutil.copy(f'{base_path}/scenarios/{args.scenario}/models/{f}', args.out_dir)

    files = os.listdir(args.out_dir)
    files = [x for x in files if x.endswith('.py')]
    print(files)
    for f in files:
        with open(f, 'r') as file:
            data = file.read()
        for p in possible_imports:
            data = data.replace(p, p.split(".")[-1])
        with open(f, 'w') as file:
            file.write(data)


if __name__ == '__main__':
    main()