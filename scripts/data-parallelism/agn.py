"""PyTorch Boilerplate Code for Accumulated Gradient Normalization.

Author:    Joeri R. Hermans
Date:      2 November 2017
"""

from torch.autograd import *
from torch.optim import *

import numpy as os
import os
import pickle
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    """Main entry point of the distributed optimization mechanism."""
    # Fetch the settings from the arguments.
    settings = parse_arguments()
    # Check if the provided settings are valid.
    if settings['valid']:
        # TODO Implement.
        pass
    else :
        # Display the usage information.
        usage()


def set_parameterization(model, parameters):
    """Sets the tensors of the model to the specified set of parameters."""
    i = 0
    for p in model.parameters():
        p.data.copy_(parameters[i], async=True)
        i += i


def parse_arguments():
    """Parses the provided program arguments, and validates the types."""
    settings = {}
    valid = True
    # Obtain and store the arguments.
    store_argument_key(settings, key='--rank', store_in='rank', default=None)
    store_argument_key(settings, key='--world-size', store_in='world_size', default=None)
    store_argument_key(settings, key='--annouce-port', store_in='announce_port', default=5001)
    store_argument_key(settings, key='--communication-frequency', store_in='communication_frequency', default=15)
    store_argument_key(settings, key='--backend', store_in='backend', default='tcp')
    store_argument_key(settings, key='--master', store_in='master', default='127.0.0.1')
    store_argument_key(settings, key='--master-port', store_in='master_port', default=5000)
    # Validate and convert the type of the arguments.
    valid &= validate_argument_key(settings, 'rank', type='int')
    valid &= validate_argument_key(settings, 'world_size', type='int')
    valid &= validate_argument_key(settings, 'announce_port', type='int')
    valid &= validate_argument_key(settings, 'communication_frequency', type='int')
    valid &= validate_argument_key(settings, 'backend', type='string')
    valid &= validate_argument_key(settings, 'master', type='string')
    valid &= validate_argument_key(settings, 'master_port', type='int')
    # Set the validation flag of the settings.
    settings['valid'] = valid

    return settings


def store_argument_key(settings, key, store_in, default=None):
    """Stores the value of the specfied key in the settings map under the 'store_in' key.
    Sets the default value if it is not present.
    """
    if key in sys.argv and sys.argv.index(key) + 1 < len(sys.argv):
        settings[store_in] = sys.argv[sys.argv.index(key) + 1]
    else:
        settings[store_in] = default


def validate_argument_key(settings, key, type=None):
    """Validates the type of the specified key, and converts it if necessary."""
    valid = False
    if key in settings.keys():
        try:
            # Check if any conversion needs to be done.
            if type == 'int':
                settings[key] = int(settings[key])
            elif type == 'float':
                settings[key] = float(settings[key])
            valid = True
        except:
            pass

    return valid


def usage():
    """Displays the usage message of this script."""
    options = '''\033[1mAccumulated Gradient Normalization\033[0m

\033[1mRequired Arguments:\033[0m
    --rank [int] Rank-identifier of the local optimization process.
    --world-size [int] Total number of workers involved in the optimization process.

\033[1mOptional Arguments:\033[0m
    --backend [string] PyTorch Distributed backend ('tcp', 'mpi', or 'gloo'). Default 'tcp'.
    --annouce-port [int] Port responsible for handling broadcast requests. Default 5001.
    --communication-frequency [int] Number of local iterations before announcing ready state. Default 15 Hz.
    --master-port [int] Port on which the master orchestrator will run. Default 5000.
    --master [string] IP address of the master process. Default '127.0.0.1'.
'''
    print(options)


if __name__ == '__main__':
    main()
