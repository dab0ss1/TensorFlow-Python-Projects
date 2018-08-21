# dependencies
import tensorflow as tf

# constant: can't change value, can be numbers or strings
one = tf.constant(1, name="one")
hello = tf.constant("hello", name="hello")
space = tf.constant(" ", name="space")
world = tf.constant("world", name="world")

# variables: the value can be changed
zero = tf.Variable(0, name="zero")

# placeholder: save/hold resources to be used later
ph = tf.placeholder(tf.float32, name="ph")

# operations: summation, subtract, multiply, divide
new_value = tf.add(zero, one, name="new_value")
hello_space_world = hello + space + world
value = 2 * ph

# update variable value
update = tf.assign(zero, new_value)

''' Need to do if you are using TF variables'''
# initialize global variables
init_op = tf.global_variables_initializer()
# create a session
sess = tf.Session()
''' Need to do if you are using TF variables'''
# initialize variables
sess.run(init_op)


# run the session (examples)
print("zero: " + str(sess.run(zero)))
print("one: " + str(sess.run(one)))
print("zero + one: " + str(sess.run(new_value)))
print("hello_space_world: " + str(sess.run(hello_space_world)))
result1 = sess.run(value, feed_dict={ph:3})
dict = {ph:[1, 2, 3]}
result2 = sess.run(value, feed_dict=dict)
print("result1: " + str(result1))
print("result2: " + str(result2))

# close session when done
sess.close()

# open, run, and close session all in one go
with tf.Session() as sess:
    result = sess.run(hello + space + world)
    print("hello world; " + str(result))

# get default graph
graph = tf.get_default_graph()

# displays all nodes in the graph
gop = graph.get_operations()
print("gop: " + str(gop))
