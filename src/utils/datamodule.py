import torch
import random
from typing import Optional, Iterator, List, Callable, Dict

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

import math

def select_distribution(distribution_config) -> Callable:
    # distribution config contains the name of the distribution and its parameters
    # for instance, for a uniform distribution we have the following config:
    # distribution_config: {name: uniform, parameters: {a: 0, b: 1}}
    if distribution_config.name == 'uniform':
        a = distribution_config['parameters']['a']
        b = distribution_config['parameters']['b']
        def random_uniform_function():
            return random.uniform(a, b)
        return random_uniform_function

    elif distribution_config.name == 'normal':
        mu = distribution_config['parameters']['mu']
        sigma = distribution_config['parameters']['sigma']
        def random_normal_function():
            return random.normalvariate(mu, sigma)
        return random_normal_function

    elif distribution_config.name == 'randint':
        a = distribution_config['parameters']['a']
        b = distribution_config['parameters']['b']
        def random_randint_function():
            return random.randint(a, b)
        return random_randint_function

    elif distribution_config.name == 'choice':
        seq = distribution_config['parameters']['seq']
        def random_choice_function():
            return random.choice(seq)
        return random_choice_function

    elif distribution_config.name == 'choices':
        seq = distribution_config['parameters']['seq']
        weights = distribution_config['parameters']['weights']
        def random_choices_function():
            return random.choices(seq, weights)
        return random_choices_function
