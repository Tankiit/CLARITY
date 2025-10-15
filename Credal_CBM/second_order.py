import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, RationalQuadratic
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
import time
import logging
import os

class NestedCBM:
    def __init__(self, embedding: KernelEmbedding, concept_space: np.ndarray):