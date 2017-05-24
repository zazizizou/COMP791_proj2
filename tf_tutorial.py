import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly 
print(node1, node2)

# The following code creates a Session object and then invokes its run
# method to run enough of the computational graph to evaluate node1
# and node 2. 
sess = tf.Session()
print(sess.run([node1, node2]))

# Operations are also nodes
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

# using placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1,3], b: 2}))

# variables
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x+b

# initialize variables
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

# Loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y: [0, -1, -2, -3]}))

# tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
for i in range(1000):
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print(sess.run([W, b]))



#####################
# simpler version with tf.contrib.learn
####################
print("starting simpler program")
import numpy as np

# declare list of features.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# estimator
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
x = np.array([1.,2.,3.,4.])
y = np.array([0.,-1,-2,-3])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                                num_epochs=1000)
# training
estimator.fit(input_fn=input_fn, steps=1000)
# evaluation
print(estimator.evaluate(input_fn=input_fn))


