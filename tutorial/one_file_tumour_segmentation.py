
from collections import OrderedDict
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import itertools as it
import math
import pprint
import yaml
import os
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from random import randint, random, sample, uniform
import pathlib
from sklearn.model_selection import KFold
import SimpleITK as sitk
import time
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from types import SimpleNamespace
from itertools import combinations, product

NULL_IMAGE = "data/MRI_data/999_im.nii.gz"
DATA_FOLDER = "data/MRI_data/BraTS2020"













def generate_combinations(strings):
    combinations_list = []
    for r in range(1, len(strings) + 1):
        combinations_list.extend([list(comb) for comb in combinations(strings, r)])
    return combinations_list

input_pattern = ["_t1", "_t1ce", "_t2", "_flair"]
combinations = generate_combinations(input_pattern)
combinations = combinations[8:]
combinations.reverse()

run_train = True
if run_train:
    print("Start random")
    tms = TumorSegmentation()
    tms.train_param.save_folder = "full_rand_neural_net"
    tms.train_param.rand_blank = True
    tms.train_param.data_folder = DATA_FOLDER
    tms.train_param.input_patterns = ["_t1", "_t1ce", "_t2", "_flair"]
    tms.run_training()
    print("Finished random")

run_train2 = False
if run_train2:
    for pattern in combinations:
        print("Start ", pattern)
        tms = TumorSegmentation()
        tms.train_param.save_folder = "".join(pattern)
        tms.train_param.data_folder = DATA_FOLDER
        tms.train_param.input_patterns = pattern
        tms.run_training()
        print("Finished ", pattern)

