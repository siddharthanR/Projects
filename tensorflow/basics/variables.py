import tensorflow as tf

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

print("\na:{0}".format(tf.Session().run(a, feed_dict = {
											a : 100})))
print("b:{0}".format(tf.Session().run(b, feed_dict = {
											b : 200})))
add = tf.add(a, b)

print("add:{0}".format(tf.Session().run(add, feed_dict = {
							a : 250, b : 305})))

with tf.Session() as session:

	print("\na:{0}".format(tf.Session().run(a, feed_dict = {
											a : 100})))
	print("b:{0}".format(tf.Session().run(b, feed_dict = {
											b : 200})))

	add = tf.add(a, b)

	print("add:{0}".format(tf.Session().run(add, feed_dict = {
							a : 250, b : 305})))