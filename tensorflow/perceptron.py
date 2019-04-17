import input_data as dt
import tensorflow as tf
mnist = dt.read_data_sets('data/', one_hot = True)

epoch = 100
batch_size = 100
display_step = 1
learning_rate = 0.0001

ip_nuerons = 784
op_nuerons = 10
nuerons_hidden1 = 256
nuerons_hidden2 = 256

X = tf.placeholder(tf.float32, [None, ip_nuerons])
Y = tf.placeholder(tf.float32, [None, op_nuerons])

weights = {
	'w1' : tf.Variable(tf.random_normal( [ip_nuerons, nuerons_hidden1])),
	'w2' : tf.Variable(tf.random_normal([nuerons_hidden1, nuerons_hidden2])),
	'out' : tf.Variable(tf.random_normal([nuerons_hidden2, op_nuerons]))
}

bias = {
	'b1' : tf.Variable(tf.random_normal([nuerons_hidden1])),
	'b2' : tf.Variable(tf.random_normal([nuerons_hidden2])),
	'out' : tf.Variable(tf.random_normal([op_nuerons]))
}

def multilayer_perceptron(x, weights, bias):
	layer1 = tf.add(tf.matmul(x, weights['w1']), bias['b1'])
	layer2 = tf.add(tf.matmul(layer1, weights['w2']), bias['b2'])
	out = tf.add(tf.matmul(layer2, weights['out']), bias['out'])
	return out

#
logits = multilayer_perceptron(X, weights, bias)

#loss function
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
		logits = logits, labels = Y))

#adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as session:

	session.run(init)

	for i in range(epoch):
		#batch wise error
		average_cost = 0
		batch = int(mnist.train.num_examples/batch_size)
		for j in range(batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			c = session.run([optimizer, error], feed_dict = {
				X : batch_x,
				Y : batch_y
				})

			#cost for every batch
			average_cost += c[1] / batch

		if epoch % display_step == 0:
			print("Epoch:{0}".format(i+1),"cost={0}".format(average_cost))