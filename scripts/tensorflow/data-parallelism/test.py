import numpy as np
import os
import sys
import tensorflow as tf
import time

task_index = int(sys.argv[2])
is_parameter_server = bool(sys.argv[1] == 'ps')
if is_parameter_server:
    job_name = 'ps'
else:
    job_name = 'worker'

cluster_specification = tf.train.ClusterSpec({"worker": ["192.168.1.12:3000", "192.168.1.12:3001"], "ps": ["192.168.1.12:5000"]})
server = tf.train.Server(cluster_specification, job_name=job_name, task_index=task_index)

with tf.device("/job:ps/task:0"):
    weights = tf.get_variable("weights", [10], initializer=tf.zeros([10]))

with tf.variable_scope("worker_0"):
    with tf.device("/job:worker/task:0"):
        x = tf.get_variable("worker_0_constant", [1], initializer=tf.constant(1.0))
        weights = tf.get_variable("weights")

with tf.device("/job:worker/task:0"):
    x_0 = tf.constant(1.0)
    W_0 = tf.identify(W)
    W_0 = tf.add(W_0, x_0)
    W += x_0

with tf.device("/job:worker/task:1"):
    x_1 = tf.constant(2.0)
    W += x_1

#with tf.device("/job:worker/task:1"):
#    x_1 = tf.constant(1.0)
#    W += x_1

init_op = tf.global_variables_initializer()

with tf.Session(server.target) as session:
    # Initialize the variables
    session.run(init_op)
    # Execute the graph.
    for i in range(0, 10):
        print("Running session.")
        W_value = session.run(W)
        print("Result done.")
        print(W_value)

server.join()
