# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import OmegaConf
import math
import os


def add_args(*args):
    return sum(float(x) for x in args)


def add_args_int(*args):
    return int(sum(float(x) for x in args))

def subtract_args_int(*args):
    sub = float(args[0])
    for i in range(1, len(args)):
        sub -= float(args[i])
    return int(sub)


def multiply_args(*args):
    return math.prod(float(x) for x in args)


def multiply_args_int(*args):
    return int(math.prod(float(x) for x in args))


def floor_division(dividend, divisor):
    return dividend // divisor

def float_division(dividend, divisor):
    return dividend / divisor

def num_files_in_directory(path):
    isExist = os.path.exists(path)
    if not isExist:
        return 0

    isDir = os.path.isdir(path)
    if not isDir:
        raise Exception(f"Path `{path}` does not correspond to a directory!")

    ls_dir = os.listdir(path)
    return len(ls_dir)


def path_to_python_executable():
    import sys

    return sys.executable


def best_ckpt_path_retrieve(path):
    isExist = os.path.exists(path)
    if not isExist:
        raise Exception(f"Path `{path}` does not exist!")

    isDir = os.path.isdir(path)
    if not isDir:
        raise Exception(f"Path `{path}` does not correspond to a directory!")

    best_ckpt_file_path = os.path.join(path, "best_ckpt_path.txt")
    best_ckpt_exists = os.path.isfile(best_ckpt_file_path)
    if not best_ckpt_exists:
        raise Exception(f"File `{best_ckpt_file_path}` does not exist. trainer.fit() might have crashed before saving the path to the best ckpt!")

    with open(best_ckpt_file_path, "r") as f:
        best_ckpt_name = f.readlines()[0]

    return os.path.join(path, "checkpoints", best_ckpt_name)

def extract_dim(ckpt_path, dim_name):
    # ckpt_path is like: "/path/to/your/home/dir/scratch/logs/training/multiruns/deepset/train-10-val-10-xdim-20-phi_dim-20-ydim-20/2024-02-24_07-54-16/seed=1234/checkpoints/last.ckpt"
    # dim_name is like: "xdim"
    import re
    pattern = f"{dim_name}-([0-9]+)"
    match = re.search(pattern, ckpt_path)
    if match:
        return int(match.group(1))
    else:
        raise Exception(f"Dimension `{dim_name}` not found in the checkpoint path `{ckpt_path}`")
    
def phi_path(ckpt_path, phi_name):
    # ckpt_path is like: "/path/to/your/home/dir/scratch/logs/training/multiruns/deepset/train-10-val-10-xdim-20-phi_dim-20-ydim-20/2024-02-24_07-54-16/seed=1234/checkpoints/last.ckpt"
    # phi_name is like: "phi_individual.pt"
    # the above file is located under # ckpt_path is like: "/path/to/your/home/dir/scratch/logs/training/multiruns/deepset/train-10-val-10-xdim-20-phi_dim-20-ydim-20/2024-02-24_07-54-16/seed=1234/"
    # go two levels up from the ckpt_path
    phi_dir = os.path.dirname(os.path.dirname(ckpt_path))

    return os.path.join(phi_dir, phi_name)

def count_char(string):
    return len(string)


OmegaConf.register_new_resolver("add", add_args)
OmegaConf.register_new_resolver("mult", multiply_args)

OmegaConf.register_new_resolver("add_int", add_args_int)
OmegaConf.register_new_resolver("sub_int", subtract_args_int)
OmegaConf.register_new_resolver("mult_int", multiply_args_int)

OmegaConf.register_new_resolver("floor_div", floor_division)
OmegaConf.register_new_resolver("float_div", float_division)

OmegaConf.register_new_resolver("num_files", num_files_in_directory)

OmegaConf.register_new_resolver("path_to_python_executable", path_to_python_executable)

OmegaConf.register_new_resolver("path_to_best_ckpt", best_ckpt_path_retrieve)
OmegaConf.register_new_resolver("extract_dim_from_ckpt", extract_dim)
OmegaConf.register_new_resolver("phi_path", phi_path)

OmegaConf.register_new_resolver("count_char", count_char)