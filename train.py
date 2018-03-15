import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

onehot = preprocessing.OneHotEncoder(categorical_features=[0])

n_epochs = 1000
learning_rate = 0.0001
num_batches = 32

file = np.load('small_images.npy')
np.random.shuffle(file)
train = file[:16384]
x = train[:, 0]
y = train[:, 1]
y = np.reshape(y, (len(y), 1))
onehot.fit(y)
y = onehot.transform(y).toarray()
l = []
for i in x:
	l.append(i)
x = np.array(l)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

X_ = tf.reshape(X, (-1, 28, 28, 1))
filter1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.1))
filter2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
filter3 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.1))
filter4 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1))
weights1 = tf.Variable(tf.random_normal([28 * 28 * 64//16 , 1000], stddev=0.1))
biases1 = tf.Variable(tf.random_normal([1000], stddev=0.1))
weights2 = tf.Variable(tf.random_normal([1000, 10], stddev=0.1))
biases2 = tf.Variable(tf.random_normal([10], stddev=0.1))
conv1 = tf.nn.conv2d(X_, filter=filter1, strides=[1, 1, 1, 1], padding="SAME")
conv1 = tf.nn.relu(conv1)
conv1 = tf.nn.max_pool(conv1, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

conv2 = tf.nn.conv2d(conv1, filter=filter2, strides=[1, 1, 1, 1], padding="SAME")
conv2 = tf.nn.relu(conv2)
#conv2 = tf.nn.max_pool(conv2, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

conv3 = tf.nn.conv2d(conv2, filter=filter3, strides=[1, 1, 1, 1], padding="SAME")
conv3 = tf.nn.relu(conv3)
conv3 = tf.add(conv1, conv3)
#conv3 = tf.nn.max_pool(conv3, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")

conv4 = tf.nn.conv2d(conv3, filter=filter4, strides=[1, 1, 1, 1], padding="SAME")
conv4 = tf.nn.relu(conv4)
conv4 = tf.nn.max_pool(conv4, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding="SAME")
conv4 = tf.reshape(conv4, (-1, 28 * 28 * 64// 16))

output = tf.add(tf.matmul(conv4, weights1), biases1)
output = tf.nn.relu(output)
output = tf.add(tf.matmul(output, weights2), biases2)
output = tf.nn.softmax(output)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(n_epochs):
	for j in range(num_batches):
		x_batch = x[j * int(len(x) / num_batches): (j+1) * int(len(x) / num_batches)]
		y_batch = y[j * int(len(x) / num_batches): (j+1) * int(len(x) / num_batches)]
		#print(x_batch.shape, y_batch.shape)
		_, c = sess.run([optimizer, loss], feed_dict={X: np.reshape(x_batch, (-1, 784)), Y: np.reshape(y_batch, (len(y_batch), 10))})
	print(c)