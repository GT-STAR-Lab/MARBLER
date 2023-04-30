import os
import time
import argparse
import shutil
import robotarium_gym.main
import robotarium_gym.utilities.roboEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='PredatorCapturePrey', help='scenario name')
    parser.add_argument('--name', type=str, default = '', help="Name to append to robotarium_submission")
    args = parser.parse_args()

    file_conversions = {".th":".tiff", ".json":".mat", ".yaml":".npy"}

    if args.name == '':
        args.out_dir = f"robotarium_submission{int(time.time())}"
    else:
        args.out_dir = f"robotarium_submission{args.name}"
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
        out = f
        for k in file_conversions:
            out = out.replace(k, file_conversions[k])
        shutil.copy(f'{base_path}/utilities/{f}', f"{args.out_dir}/{out}")
        if '.py' in f:
            possible_imports.append(f"robotarium_gym.utilities.{f[:-3]}")

    #Copy Scenario base files
    scenarios = os.listdir(f'{base_path}/scenarios/{args.scenario}')
    scenarios.remove("__pycache__")
    scenarios.remove("models")
    for f in scenarios:
        out = f
        for k in file_conversions:
            out = out.replace(k, file_conversions[k])
        shutil.copy(f'{base_path}/scenarios/{args.scenario}/{f}', f"{args.out_dir}/{out}")
        if '.py' in f:
            possible_imports.append(f"robotarium_gym.scenarios.{args.scenario}.{f[:-3]}")

    #Copy Scenario models
    models = os.listdir(f'{base_path}/scenarios/{args.scenario}/models')
    for f in models:
        out = f
        for k in file_conversions:
            out = out.replace(k, file_conversions[k])
        shutil.copy(f'{base_path}/scenarios/{args.scenario}/models/{f}', f"{args.out_dir}/{out}")

    #Fix the imports
    files = [f'{args.out_dir}/{x}' for x in os.listdir(args.out_dir) if x.endswith('py')]
    for f in files:
        with open(f, 'r') as file:
            data = file.read()
        if "main.py" in f:
            data = data.replace('PredatorCapturePrey', args.scenario)
        for p in possible_imports:
            data = data.replace(p, p.split(".")[-1])
        for k in file_conversions:
            data = data.replace(k, file_conversions[k])
        with open(f, 'w') as file:
            file.write(data)


if __name__ == '__main__':
    main()