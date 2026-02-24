

import argparse
import pprint
import yaml
import torch
from src.utils.io import MODEL_RUNS_BASE_DIR, PROCESSED_DIR, resolve_path
from src.train import main as app_main
  

parser = argparse.ArgumentParser(
    description="Minimal JEPA training pipeline for patient sequences")
parser.add_argument(
    "--cfg",        type=str, 
    required=True, help="Name of configs/* file to load")


def main():
    args = parser.parse_args()
    
    cfg_fp = resolve_path(f"configs/{args.cfg.removesuffix('.yaml').removesuffix('.yml')}.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    params = {}
    with open(cfg_fp, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print(f'loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    app_main(device, params)


if __name__ == "__main__":
    main()
    
    