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
        # Initialize the distributed backend.
        distributed_backend = settings['backend']
        master_orchestrator = settings['master']
        master_orchestrator_port = settings['master_port']
        # Obtain additional settings required for the distributed initialization.
        rank = settings['rank']
        world_size = settings['world_size']
        # Initialize the distributed process group.
        method = distributed_backend + '://' + master_orchestrator + ':' + str(master_orchestrator_port)
        dist.init_process_group(init_method=method, rank=rank, world_size=world_size, backend=distributed_backend)
        # Call the optimization procedure under the specified settings.
        optimize(settings)
    else:
        # Display the usage information.
        usage()


class Model(torch.nn.Module):
    """YOUR MODEL HERE."""

    def __init__(self, num_features, num_hidden):
        super(Model, self).__init__()
        self.fc_1 = torch.nn.Linear(num_features, num_hidden)
        self.fc_2 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc_3 = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.sigmoid(self.fc_3(x))

        return x


def build_model(settings):
    """Constructs the model under the specified settings."""
    num_features = 10
    num_hidden = 10
    model = Model(num_features, num_hidden)

    return model


def optimize(settings):
    """Distributed Optimization Procedure under the specified settings."""
    # Fetch the required settings.
    rank = settings['rank']
    world_size = settings['world_size']
    num_iterations = settings['num_iterations']
    master_rank = settings['master_rank']
    # Construst the model from the specified settings.
    model = build_model(settings)
    # Define the default next and previous rank.
    previous_rank = (rank - 1) % world_size
    next_rank = (rank + 1) % world_size
    # Allocate the data buffers.
    network_buffer = make_buffer(model)
    delta_buffer = make_buffer(model)
    # Clean the buffers.
    zero_buffer(network_buffer)
    zero_buffer(delta_buffer)
    if rank == master_rank:
        # Send the model parameterization to the next worker.
        send_model(model, next_rank)
    # Start the distributed training.
    for i in range(0, num_iterations):
        # TODO Training.
        delta_buffer[0][0][0] = 1.
        receive_parameters(network_buffer, previous_rank)
        add_buffer(delta_buffer, network_buffer)
        set_parameterization(network_buffer, model)
        send_model(model, next_rank)
        print(list(model.parameters())[0][0][0])
    # Finally, the master needs to collect the final result.
    if rank == master_rank:
        save_model(model)
    # Wait for all processes to complete.
    synchronize_workers()


def save_model(model):
    """Saves the parameterization of the model to some persistent medium."""
    # TODO Implement.
    pass


def synchronize_workers(group=None):
    """Starts a barrier (blocking) procedure to synchronize all workers of a specific group."""
    if group:
        dist.barrier(group)
    else:
        dist.barrier()


def broadcast_model(model, source=0):
    """Broadcasts the parameterization of the model to all workers."""
    parameters = [p.data for p in model.parameters()]
    for p in parameters:
        dist.broadcast(p, source)


def isend_model(model, destination):
    """Sends the parameterization of the model asynchronously."""
    parameters = [p.data for p in model.parameters()]
    isend_parameters(parameters, destination)


def send_model(model, destination):
    """Sends the parameterization of the model to the specified rank."""
    parameters = [p.data for p in model.parameters()]
    send_parameters(parameters, destination)


def send_parameters(parameters, destination):
    """Sends the parameterization to the specified rank."""
    for p in parameters:
        dist.send(p, destination)


def isend_parameters(parameters, destination):
    """Sends the specified parameters asynchronously to the destination."""
    for p in parameters:
        dist.isend(p, destination)


def receive_model(model, source):
    """Receives the parameterization from the specified rank."""
    parameters = [p.data for p in model.parameters()]
    receive_parameters(parameters, source)


def receive_parameters(parameters, source):
    """Receives the parameterization from the specified source rank."""
    for p in parameters:
        dist.recv(p, source)


def set_parameterization(parameters, model):
    """Sets the tensors of the model to the specified set of parameters."""
    tensors = list(model.parameters())
    num_tensors = len(tensors)
    for i in range(0, num_tensors):
        tensors[i].data.copy_(parameters[i], async=True)


def add_buffer(source, destination):
    """Adds the source parameter buffer to the specified destination."""
    num_parameters = len(source)
    for i in range(0, num_parameters):
        destination[i] += source[i]


def copy_buffer(source, destination):
    """Copies the specified buffer from source to destination."""
    num_parameters = len(source)
    for i in range(0, num_parameters):
        destination[i].copy_(source[i], async=True)


def make_buffer(model):
    """Copies the trainable tensors of the model to create a list of buffer tensors."""
    parameter_buffer = []
    for p in list(model.parameters()):
        parameter_buffer.append(p.clone().data)

    return parameter_buffer


def zero_buffer(parameters):
    """Zeros all the tensors in the parameter list."""
    for p in parameters:
        p.zero_()


def parse_arguments():
    """Parses the provided program arguments, and validates the types."""
    settings = {}
    valid = True
    # Obtain and store the arguments.
    store_argument_key(settings, key='--rank', store_in='rank', default=None)
    store_argument_key(settings, key='--world-size', store_in='world_size', default=None)
    store_argument_key(settings, key='--annouce-port', store_in='announce_port', default=5001)
    store_argument_key(settings, key=['--communication-frequency', '--lambda'], store_in='communication_frequency', default=15)
    store_argument_key(settings, key='--backend', store_in='backend', default='tcp')
    store_argument_key(settings, key='--master', store_in='master', default='127.0.0.1')
    store_argument_key(settings, key='--master-port', store_in='master_port', default=5000)
    store_argument_key(settings, key='--iterations', store_in='num_iterations', default=1000)
    store_argument_key(settings, key=['--batch-size', '--m'], store_in='batch_size', default=128)
    store_argument_key(settings, key='--master-rank', store_in='master_rank', default=0)
    # Validate and convert the type of the arguments.
    valid &= validate_argument_key(settings, 'rank', type='int')
    valid &= validate_argument_key(settings, 'world_size', type='int')
    valid &= validate_argument_key(settings, 'announce_port', type='int')
    valid &= validate_argument_key(settings, 'communication_frequency', type='int')
    valid &= validate_argument_key(settings, 'backend', type='string')
    valid &= validate_argument_key(settings, 'master', type='string')
    valid &= validate_argument_key(settings, 'master_port', type='int')
    valid &= validate_argument_key(settings, 'num_iterations', type='int')
    # Set the validation flag of the settings.
    settings['valid'] = valid

    return settings


def store_argument_key(settings, key, store_in, default=None):
    """Stores the value of the specfied key in the settings map under the 'store_in' key.
    Sets the default value if it is not present.
    """
    # TODO Fix situation when key is an array.
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
    --annouce-port [int] Port responsible for handling broadcast requests. Default 5001.
    --backend [string] PyTorch Distributed backend ('tcp', 'mpi', or 'gloo'). Default 'tcp'.
    --batch-size [int] Size of the mini-batch. Default 128.
    --communication-frequency [int] Number of local iterations before announcing ready state. Default 15 Hz.
    --iterations [int] Number of local mini-batches that have to be evaluated. Default 1000.
    --lambda [int] Equivalent to the `--communication-frequency` option.
    --m [int] Equivalent to the `--batch-size` option.
    --master [string] IP address of the master process. Default '127.0.0.1'.
    --master-port [int] Port on which the master orchestrator will run. Default 5000.
    --master-rank [int] Rank of the master process. Default 0.
'''
    print(options)


if __name__ == '__main__':
    main()
