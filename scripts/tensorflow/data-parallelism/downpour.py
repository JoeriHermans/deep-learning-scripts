"""This TensorFlow script is intended to show the performance of
Asynchronous Data Parallelism using DOWNPOUR.

Author:    Joeri R. Hermans
Date:      21 May, 2017
"""


import numpy as np
import os
import sys
import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data


def execute_worker(settings):
    """Executes the worker procedure."""
    # Obtain the task variables from the settings.
    task_index = settings['task-index']
    cluster_specification = settings['cluster-specification']
    server = settings['server']
    num_epochs = settings['epochs']
    batch_size = settings['mini-batch']
    log_path = settings['log-path']

    # Assign the variables to the PS job.
    # TODO Check if already initialized.
    with tf.device("/job:ps/task:0"):
        tf.set_random_seed(1)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

    # Build the computation graph.
    print(task_index)
    with tf.device("/job:worker/task:" + str(task_index)):
        # count the number of updates
        global_step = tf.get_variable('global_step', [], initializer = tf.constant_initializer(0, dtype=tf.int64),  trainable = False)

        # input images
        with tf.name_scope('input'):
            # None -> batch size can be any size, 784 -> flattened mnist image
            x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
            # target 10 output classes
            y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")

        # Copy PS variables.
        with tf.name_scope("pull"):
            W1_w = tf.identity(W1)
            W2_w = tf.identity(W2)
            b1_w = tf.identity(b1)
            b2_w = tf.identity(b2)

        # implement model
        with tf.name_scope("softmax"):
            # y is our prediction
            z2 = tf.add(tf.matmul(x,W1_w),b1_w)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2,W2_w),b2_w)
            y  = tf.nn.softmax(z3)

        # specify cost function
        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # specify optimizer
        with tf.name_scope('train'):
            # optimizer is an "operation" which we can execute in a session
            grad_op = tf.train.AdamOptimizer(0.0001)
            gradients = grad_op.compute_gradients(cross_entropy)
            train_op = grad_op.apply_gradients(gradients, global_step=global_step)

        # Store local variables in PS.
        with tf.name_scope('commit'):
            W1 = W1.assign(W1_w)
            W2 = W2.assign(W2_w)
            b1 = b1.assign(b1_w)
            b2 = b2.assign(b2_w)

        with tf.name_scope('Accuracy'):
            # accuracy
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # create a summary for our cost and accuracy
            tf.summary.scalar("cost", cross_entropy)
            tf.summary.scalar("accuracy", accuracy)

        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        supervisor = tf.train.Supervisor(is_chief=(task_index == 0), global_step=global_step, init_op=init_op, logdir=log_path)

        # Read / obtain the MNIST dataset.
        mnist = input_data.read_data_sets('mnist_data', one_hot=True)

    frequency = 100
    tf_config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)
    with supervisor.prepare_or_wait_for_session(server.target, config=tf_config) as sess:
        # create log writer object (this will log on every machine)
        writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        # perform training cycles
        start_time = time.time()
        for epoch in range(num_epochs):
            # number of batches in one epoch
            batch_count = int(mnist.train.num_examples/batch_size)

            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                # perform the operations we defined earlier on batch
                _, cost, summary, step = sess.run([train_op, cross_entropy, summary_op, global_step], feed_dict={x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i+1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step+1),
                          " Epoch: %2d," % (epoch+1),
                          " Batch: %3d of %3d," % (i+1, batch_count),
                          " Cost: %.4f," % cost,
                          " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
                    count = 0


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
    print("    --epochs [int]\t\tNumber of epochs.\n\t\t\t\tNumber of full data iterations.")
    print("    --mini-batch [int]\t\tMini-batch size.")
    print("    --log-path [string]\t\tSpecifies the log-file on every machine.\n\t\t\t\tDefault: /tmp/tensorflow-logs/downpour")
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
        settings['epochs'] = 1
        settings['mini-batch'] = 32
        settings['log-path'] = '/tmp/tensorflow-logs/downpour'
        # Obtain the variables from the program arguments.
        settings['worker'] = '-w' in sys.argv
        settings['ps'] = '-ps' in sys.argv
        # Check for a different communication frequency.
        if '--epochs' in sys.argv: settings['epochs'] = argument_value('--epochs')
        if '--mini-batch' in sys.argv: settings['mini-batch'] = argument_value('--mini-batch')
        if '--worker-hosts' in sys.argv: settings['worker-hosts'] = argument_value('--worker-hosts')
        if '--ps-hosts' in sys.argv: settings['ps-hosts'] = argument_value('--ps-hosts')
        if '--task-index' in sys.argv: settings['task-index'] = argument_value('--task-index')
        if '--log-path' in sys.argv: settings['log-path'] = argument_value('--log-path')
        validate_settings(settings)
        format_settings(settings)


    return settings


if __name__ == '__main__':
    main()
