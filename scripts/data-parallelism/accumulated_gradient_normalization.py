"""This TensorFlow script is intended to show the performance of
Asynchronous Data Parallelism using Accumulated Gradient Normalization.

In this script we train a simple model in an asynchronous manner. In an other
script, we provide a baseline using Google's DOWNPOUR to compare against.

Author:    Joeri R. Hermans
Date:      21 May, 2017
"""


import numpy as np
import os
import tensorflow as tf
import sys


def main():
    # Fetch the settings from the specified arguments.
    settings = process_arguments()
    print(settings)


def help():
    """Displays the help message."""
    logo = '''

     Accumulated              `-::/:/+osoosoo+:::-.`
     Gradient             `-:/soosooooso+oossso+oooso+/.
     Normalization     . +ssoso++ooo+so+++soo+osoooososso-
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
    print("Basic usage:\n    python agn.py -w\t\tRun as a worker.\n    python agn.py -ps\t\tRun as a parameter server.\n")
    print("Optional arguments:")
    print("    --lambda [int]\t\tCommunication frequency\n\t\t\t\tExploration steps before communicating with the parameter server.")
    print("    --epochs [int]\t\tNumber of epochs.\n\t\t\t\tNumber of full data iterations.")
    print("    --mini-batch [int]\t\tMini-batch size.")
    print("\n\n")
    exit(1)


def argument_value(key):
    """Obtains the value of the specified program argument."""
    index = sys.argv.index(key)

    return int(sys.argv[index + 1])


def validate_settings(settings):
    """Validates the program settings, and exits on failure."""

    if not settings['worker'] and not settings['ps']:
        help()


def process_arguments():
    """Processes the program arguments and returns a dictionary of the program
    configuration.
    """
    # Check if the help message needs to be displayed.
    if '-h' in sys.argv or '--help' in sys.argv:
        help()
    else:
        settings = {}
        # Set the default values.
        settings['lambda'] = 15 # Communication frequency.
        settings['epochs'] = 1
        settings['mini-batch'] = 32
        # Obtain the variables from the program arguments.
        settings['worker'] = '-w' in sys.argv
        settings['ps'] = '-ps' in sys.argv
        # Check for a different communication frequency.
        if '--lambda' in sys.argv: settings['communication-frequency'] = argument_value('--lambda')
        if '--epochs' in sys.argv: settings['epochs'] = argument_value('--epochs')
        if '--mini-batch' in sys.argv: settings['mini-batch'] = argument_value('--mini-batch')
        # Validate the settings.
        validate_settings(settings)

    return settings


if __name__ == '__main__':
    main()
