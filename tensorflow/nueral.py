import input_data as dt
import tensorflow as tf

mnist = dt.read_data_sets('data/', one_hot = True)

#hyper paramters
learning_rate = 0.0001
epoch = 5
display_step = 10
batch_size = 100
ip_nuerons = 784
op_nuerons = 10
h1_nuerons = 256
h2_nuerons = 256

#graph input 
x = tf.placeholder(tf.float32, [None, ip_nuerons])
y = tf.placeholder(tf.float32, [None, op_nuerons])

#defining weights
weights = {
	'w1' : tf.Variable(tf.random_normal([ip_nuerons, h1_nuerons])),
	'w2' : tf.Variable(tf.random_normal([h1_nuerons, h2_nuerons])),
	'out' : tf.Variable(tf.random_normal([h2_nuerons, op_nuerons]))

}

#defining bias
bias = {
	'b1' : tf.Variable(tf.random_normal([h1_nuerons])),
	'b2' : tf.Variable(tf.random_normal([h2_nuerons])),
	'out' : tf.Variable(tf.random_normal([op_nuerons]))
}

#2 layer feed forward network
def nueral_nets(x, weights, bias):
	l1 = tf.add(tf.matmul(x, weights['w1']), bias['b1'])
	l2 = tf.add(tf.matmul(l1, weights['w2']), bias['b2'])
	out = tf.add(tf.matmul(l2, weights['out']), bias['out'])
	return out

logits = nueral_nets(x, weights, bias)

#loss function
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		logits = logits, labels = y))

#adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as session:

	session.run(init)

	for i in range(epoch):
		average_cost = 0
		batch = int(mnist.train.num_examples / batch_size)
		for j in range(batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			c = session.run([error, optimizer], feed_dict = {
				x : batch_x,
				y : batch_y
			})

			average_cost += c[0] / batch

		if epoch % display_step == 0:
			print("Epoch:{0}".format(i+1),"cost={0}".format(average_cost))