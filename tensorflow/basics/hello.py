import tensorflow as tf

h = tf.constant("hello world")

print("hello:{0}".format(tf.Session().run(h)))

with tf.Session() as session:

	print("hello:{0}".format(session.run(h)))