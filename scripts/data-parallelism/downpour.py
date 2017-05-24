"""This TensorFlow script is intended to show the performance of
Asynchronous Data Parallelism using DOWNPOUR.

Author:    Joeri R. Hermans
Date:      21 May, 2017
"""


import numpy as np
import os
import tensorflow as tf
import sys


def execute_worker(settings):
    """Executes the worker procedure."""
    # Obtain the task variables from the settings.
    task_index = settings['task-index']
    cluster_specification = settings['cluster-specification']
    server = settings['server']
    communication_frequency = settings['lambda']
    # Build the computation graph.
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,
                                                  cluster=cluster_specification)):

        # Initialize the global step counter.
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0, dtype=tf.int64), trainable=False)
        # Network structure will be as follows:
        # 784 (input)
        # 2000 (hidden-1, relu activations)
        # 2000 (hidden-2, relu activations)
        # 10 (output, softmax)

        # Construct the input of the network.
        with tf.name_scope("input"):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
            y_target = tf.placeholder(tf.float32, shape=[None, 10], name='target')

        # Construct the parameters.
        with tf.name_scope("parameters"):
            W_1 = tf.Variable(tf.random_normal([784, 2000]))
            b_1 = tf.Variable(tf.zeros([2000]))
            W_2 = tf.Variable(tf.random_normal([2000, 2000]))
            b_2 = tf.Variable(tf.zeros([2000]))
            W_3 = tf.Variable(tf.random_normal([2000, 10]))
            b_3 = tf.Variable(tf.zeros([10]))

        # Define the training procedure.
        with tf.name_scope("train"):
            # Compute activations of the first hidden layer.
            z_1 = tf.add(tf.matmul(x, W_1), b_1)
            a_1 = tf.nn.relu(z_1)
            # Compute activations of the second hidden layer.
            z_2 = tf.add(tf.matmul(a_1, W_2), b_2)
            a_2 = tf.nn.relu(z_2)
            # Compute the predictions of the neural network.
            z_3 = tf.add(tf.matmul(a_2, W_3), b_3)
            y = tf.nn.softmax(z_3)
            # Define the cross entropy.
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
            # Define the optimizer, and its operations.
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            gradients = optimizer.compute_gradients(cross_entropy)
            apply_gradient_op = optimizer.apply_gradients(gradients, global_step=global_step)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("loss", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

    supervisor = tf.train.Supervisor(is_chief=(task_index == 0), global_step=global_step, init_op=init_op)
    with supervisor.prepare_or_wait_for_session(server.target) as sess:
        raise NotImplementedError


def execute_server(settings):
    """Executes the server procedure."""
    server = settings['server']
    server.join()


def construct_cluster_specification(settings):
    """Constructs the cluster specification from the validated settings."""
    workers = settings['worker-hosts']
    parameter_servers = settings['ps-hosts']
    cluster_specification = tf.train.ClusterSpec({"worker": workers, "ps": parameter_servers})

    return cluster_specification


def construct_server(settings):
    """Starts a server which will handle the local task."""
    # Obtain the required parameters from the settings.
    cluster_specification = settings['cluster-specification']
    job_name = settings['job-name']
    task_index = settings['task-index']
    # Allocate the server, with the required arguments.
    server = tf.train.Server(cluster_specification, job_name, task_index)

    return server


def running_worker(settings):
    """Check if the user initiated the worker script, given the settings."""

    return settings['worker']


def main():
    """Main entry point of the distributed training script."""
    # Fetch the settings from the specified arguments.
    settings = process_arguments()
    # Construct the cluster specification, and server.
    cluster_specification = construct_cluster_specification(settings)
    settings['cluster-specification'] = cluster_specification
    server = construct_server(settings)
    settings['server'] = server
    # Check if the user initiated the parameter server, or worker procedure.
    if running_worker(settings):
        # Run the worker procedure.
        execute_worker(settings)
    else:
        # Run the server procedure.
        execute_server(settings)


def help():
    """Displays the help message."""
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
    print("Basic usage:\n    python downpour.py -w\tRun as a worker.\n    python downpour.py -ps\tRun as a parameter server.\n")
    print("Required arguments:")
    print("    --worker-hosts [string]\tComma-separated list defining the worker hosts.")
    print("    --ps-hosts [string]\t\tComma-separated list defining the parameter server hosts.")
    print("    --task-index [int]\tTask index associated with the worker or PS.\n")
    print("Optional arguments:")
    print("    --lambda [int]\t\tCommunication frequency\n\t\t\t\tExploration steps before communicating with the parameter server.")
    print("    --epochs [int]\t\tNumber of epochs.\n\t\t\t\tNumber of full data iterations.")
    print("    --mini-batch [int]\t\tMini-batch size.")
    print("\n\n")
    exit(1)


def argument_value(key):
    """Obtains the value of the specified program argument."""
    index = sys.argv.index(key)

    return sys.argv[index + 1]


def validate_settings(settings):
    """Validates the program settings, and exits on failure."""

    # Check if the worker or parameter server flag has been specified.
    if not settings['worker'] and not settings['ps']:
        print("Please specify a worker '-w' or a parameter server '-ps' flag.")
        help()
    # Check if the host / ps hosts have been specified.
    if 'worker-hosts' not in settings or 'ps-hosts' not in settings:
        print("Please specify the worker and parameter server hosts.")
        help()
    # Check if a task index has been specified.
    if 'task-index' not in settings:
        print("Please specify a task index.")
        help()


def format_settings(settings):
    """Formats the settings to the expected structure, and converts the
    program arguments to the correct types.
    """
    # Format the provided arguments.
    settings['lambda'] = int(settings['lambda'])
    settings['epochs'] = int(settings['epochs'])
    settings['mini-batch'] = int(settings['mini-batch'])
    settings['worker-hosts'] = settings['worker-hosts'].split(",")
    settings['ps-hosts'] = settings['ps-hosts'].split(",")
    # Check if task-index has been specified.
    settings['task-index'] = int(settings['task-index'])
    # Set the job name, given the formatted settings.
    if running_worker(settings):
        job_name = "worker"
    else:
        job_name = "ps"
    settings['job-name'] = job_name


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
        if '--worker-hosts' in sys.argv: settings['worker-hosts'] = argument_value('--worker-hosts')
        if '--ps-hosts' in sys.argv: settings['ps-hosts'] = argument_value('--ps-hosts')
        if '--task-index' in sys.argv: settings['task-index'] = argument_value('--task-index')
        validate_settings(settings)
        format_settings(settings)


    return settings


if __name__ == '__main__':
    main()
