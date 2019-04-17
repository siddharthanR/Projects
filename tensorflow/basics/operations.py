import tensorflow as tf

#constant values 
a = tf.constant(10, tf.int32)
b = tf.constant(20, tf.int32)

#performing addition operation
add = tf.add(a, b)

#product of two constants
product = tf.multiply(a, b)

#creating session for every invocation
print("\na:{0}".format(tf.Session().run(a)))
print("b:{0}".format(tf.Session().run(b)))
print("sum:{0}".format(tf.Session().run(add)))
print("product:{0}".format(tf.Session().run(product)))

#seperate session
with tf.Session() as session:

	#one session for all invocation
	print("\na:{0}".format(session.run(a)))
	print("b:{0}".format(session.run(b)))
	print("sum:{0}".format(tf.Session().run(add)))
	print("product:{0}".format(tf.Session().run(product)))
session.close()