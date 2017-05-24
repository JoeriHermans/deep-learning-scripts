"""This TensorFlow script is intended to show the performance of
Asynchronous Data Parallelism using Accumulated Gradient Normalization.

In this script we train a simple model in an asynchronous manner. In an other
script, we provide a baseline using Google's DOWNPOUR to compare against.

Author:    Joeri R. Hermans
Date:      21 May, 2017

Program arguments:
-c [int]       --communication-frequency [int]         Number of local steps before communicating
                                                       the AGN gradient to the parameter server.
"""


import argparse
import numpy as np
import os
import tensorflow as tf


def build_argument_parser():
    parser = argparse.ArgumentParser()


def process_arguments():
    # Construct the argument parser.
    parser = build_argument_parser()
    raise NotImplementedError


def main():
    # Fetch the settings from the specified arguments.
    settings = process_arguments()


if __name__ == '__main__':
    main()
