"""PyTorch script which is intended to show the performance of
Asynchronous Data Parallelism using DOWNPOUR. Parameter sharing
is done using MPI.

Author:    Joeri R. Hermans
Date:      4 July, 2017"""


from torch.autograd import *
from torch.optim import *

from mpi4py import MPI as mpi
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pickle


def get_numpy_parameters(model):
    """Returns the parameters of the model in a Numpy format."""
    parameters = []
    for p in model.parameters():
        parameters.append(p.data.numpy())
    parameters = np.asarray(parameters)

    return parameters


def run_parameter_server(comm, model):
    """Runs the parameter server procedure."""
    central_variable = get_numpy_parameters(model)
    num_tensors = len(central_variable)
    comm.send(central_variable, dest=1)


def run_worker(comm, model, rank):
    """Runs the worker procedure."""
    b = bytearray(248631)
    parameters = get_numpy_parameters(model)
    num_tensors = len(parameters)
    comm.recv(b, source=0)
    parameters = pickle.loads(b)
    print(parameters)


def main():
    """Entry point of the training script."""
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    model = build_model()
    if rank == 0:
        run_parameter_server(comm, model)
    else:
        run_worker(comm, model, rank)

    return None


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def build_model():
    """Returns the Torch model."""
    return Model()


def optimize():
    """Runs the optimization procedure."""
    # Build the Torch model.
    model = build_model()
    model.zero_grad()
    parameters = model.parameters()
    # Get the data.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Specify the optimizer parameters.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Training.
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')


def help():
    """Displays the help message of this script."""
    logo = '''

     DOWNPOUR                 `-::/:/+osoosoo+:::-.`
                         `-:/soosooooso+oossso+oooso+/.
                      . +ssoso++ooo+so+++soo+osoooososso-
                     ./oso+o+++s+++s++oos+oooo++oo++ooooso`
                    +oos+++osoooooos++oy++++s+++soooo+soosy+.
                   +o++o+++oso+oo++s+++oo+++o+ooo+++os++o+oss`
                  .yoooo+ooo++++s++s++s++++s+++++oo++ss+oo+so+-
                .oso+ooooo++++s++++++soossoooso+++++++so+oooo-
                 +so+oo+++oo+oosoooossoo++++++oo+++++sooo+s+ss-
                  /osyoo++ooo+++++++o+++oo+++++++++oo+++ooo+ss+
                   :+++sys++oo++++oo+++osooosso+ooo+++oo+s+++ss
                     .-:+o+++ssoossoooo+o+++o+oooo+++oo++osos+s
                         /++oso+++so+++++oo+++++++++o++oooo+ss+
                          :+ssoo++oooooooo+oooooooosoooo+++sos.
                            .:++ooo+++ooso+oossooooooooosooo/`
                                      .++o////++/--////++`
                                        :oo+/:://////:-+.
                                         -oooo++++++::-`
                                          .o+o`
                                           .//.


'''
    print(logo)


if __name__ == '__main__':
    main()
