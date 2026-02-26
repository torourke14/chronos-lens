

import argparse
import torch
from src.utils.io import create_run
from src.training.train import main as app_main
  

parser = argparse.ArgumentParser(
    description="Minimal JEPA training pipeline for patient sequences")
parser.add_argument(
    "--model", type=str, required=True, 
    help="""Name of model to run (subdir of 'experiments'). If model exists, 
            a new model artifacts directory will be populated. To create a new
            model, specify a new folder name 'expirements/[model_name]' with 
            config.yaml in it, and then specify --model=[model_name] to run""")


def main():
    args = parser.parse_args()
    
    run_dir, params = create_run(args.model)
    
    if not params:
        raise SystemExit(f"Failed to load parameters from experiments/{args.model}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    app_main(params, run_dir, device)
    

if __name__ == "__main__":
    main()
    
    