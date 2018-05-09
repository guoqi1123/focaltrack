import tensorflow as tf
import numpy as np
import pdb

# this code double checks whether derivatives of sorting in
# Tensorflow works correctly
# it seems it did
graph = tf.Graph()
session = tf.Session(graph = graph)

a = np.array([1,2,5,4])
c = np.array([10,11,12,13])
d = np.array([14,15,16,17])
with graph.as_default():
	a_tf = tf.Variable(a, dtype = tf.float32)
	c_tf = tf.Variable(c, dtype = tf.float32)
	d_tf = tf.Variable(d, dtype = tf.float32)
	val = c_tf * d_tf
	b_tf = tf.gather(
		val,
		tf.nn.top_k(a_tf, k=len(a), sorted=True).indices,
	)
	db_tfda_tf = []
	for i in range(len(a)):
		tmp, = tf.gradients(b_tf[i],[c_tf])
		db_tfda_tf.append(tmp)
	init_op = tf.initialize_all_variables()
	session.run(init_op)

for i in range(len(a)):
	print(session.run(db_tfda_tf[i]))

# the derivative changes dynamically
session.run(a_tf.assign([5,4,2,1]))
for i in range(len(a)):
	print(session.run(db_tfda_tf[i]))

