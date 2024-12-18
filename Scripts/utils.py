import argparse
import numpy as np
import random
import transformers
import torch
import collections
import datasets
import os
import shutil
import sys
import warnings
import requests
import pkg_resources
import subprocess

def bool_type(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def disable_caching(args):
    if args.use_cache:
        sys.dont_write_bytecode = False
    else:
        datasets.disable_caching()
        sys.dont_write_bytecode = True
        for root, dirs, files in os.walk(".", topdown=False):
            for name in dirs:
                if name == "__pycache__":
                    shutil.rmtree(os.path.join(root, name))

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def print_parameters(args, accelerator):
    if args.display_parameters:
        accelerator.print("Parameters:")
        for arg in vars(args):
            accelerator.print(f"{arg}: {getattr(args, arg)}")

def disable_warnings(args):
    if not args.display_warnings:
        warnings.filterwarnings("ignore")
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

def control_verbosity(args):
    if args.verbose:
        datasets.enable_progress_bar()
    else:
        datasets.disable_progress_bar()
        
def print_fold_results(args, fold, results, accelerator):
    accelerator.print('\n' + '*'*20)
    accelerator.print(f'* Results - Fold {fold} *')
    accelerator.print('*'*20 + '\n')
    accelerator.print(f"Results for fold {fold}:")
    for key, value in results.items():
        accelerator.print(f">>> {key}: {value[-1]:.4f}")

def print_final_results(args, results, accelerator):
    accelerator.print('\n' + '*'*17)
    accelerator.print(f'* Final results *')
    accelerator.print('*'*17 + '\n')
    accelerator.print(f"Average results across {args.folds} folds:")
    for key, value in results.items():
        accelerator.print(f">>> {key}: {np.mean(value):.4f}")

def check_files(accelerator):
    github_api = 'https://api.github.com/repos/cesar-leblanc/plantbert/contents'
    stack = [(github_api, '')]
    missing_files = []
    while stack:
        current_api, current_path = stack.pop()
        try:
            response = requests.get(current_api) 
            if response.status_code != 200:
                accelerator.print(f"Failed to retrieve GitHub repository contents. Error code: {response.status_code}.")
            github_contents = response.json()
            for file_info in github_contents:
                file_name = file_info["name"]
                file_path = os.path.join(os.getcwd().replace('/Scripts', ''), current_path, file_name)
                if file_info["type"] == "file":
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)
                elif file_info["type"] == "dir":
                    stack.append((file_info["url"], os.path.join(current_path, file_name)))
        except requests.exceptions.RequestException as e:
            accelerator.print(f"An error occurred while accessing the GitHub API: {e}")
        except OSError as e:
            accelerator.print(f"An error occurred while accessing the local file system: {e}")
    return missing_files

def check_dependencies(REQUIRED_DEPENDENCIES):
    missing_dependencies = {}
    for dependency, required_version in REQUIRED_DEPENDENCIES.items():
        try:
            pkg_resources.require(dependency)
            installed_version = pkg_resources.get_distribution(dependency).version
            if installed_version != required_version:
                missing_dependencies[dependency] = required_version
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            missing_dependencies[dependency] = required_version
    return missing_dependencies

def check_environment():
    missing_environments = [False, False, False, False, False, False]
    required_python_version = (3, 8)
    python_version = sys.version_info[:2]
    if python_version < required_python_version:
        missing_environments[0] = True
    cuda_version = subprocess.getoutput("nvcc --version")
    if "command not found" in cuda_version:
        missing_environments[1] = True
    pip_version = subprocess.getoutput("pip --version")
    if "command not found" in pip_version:
        missing_environments[2] = True
    git_version = subprocess.getoutput("git --version")
    if "command not found" in git_version:
        missing_environments[3] = True
    git_lfs_version = subprocess.getoutput("git-lfs --version")
    if "command not found" in git_lfs_version:
        missing_environments[4] = True
    environment = sys.executable
    if "pl@ntbert" not in environment:
        missing_environments[5] = True
    return missing_environments

def print_datasets_info(accelerator, mask_dataset, classification_dataset):
    accelerator.print("Mask dataset (without habitat types):")
    accelerator.print(f"   - {len(mask_dataset['train'])} vegetation plots")
    accelerator.print(f"   - {len([species.strip() for row in mask_dataset['train']['text'] for species in row.split(',')])} species observations")
    accelerator.print(f"   - {len(set([species.strip() for row in mask_dataset['train']['text'] for species in row.split(',')]))} unique species")
    accelerator.print("Classification dataset (with habitat types):")
    accelerator.print(f"   - {len(classification_dataset)} folds")
    accelerator.print(f"   - {sum(dataset.num_rows for dataset in classification_dataset.values())} vegetation plots (an average of {round(sum(dataset.num_rows for dataset in classification_dataset.values()) / len(classification_dataset))} per fold)")
    accelerator.print(f"   - {sum(len(species.split(', ')) for fold in classification_dataset.values() for species in fold['text'])} species observations")
    accelerator.print(f"   - {len(set(species for fold in classification_dataset.values() for sample in fold['text'] for species in sample.split(', ')))} unique species")
    accelerator.print(f"   - {len(set(label for fold in classification_dataset.values() for label in fold['label']))} unique habitats")