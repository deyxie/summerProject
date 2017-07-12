from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

# 这里的784 是输入的mnist 28x28的像素图片
# none 表示 输入的数量 可以是任何大小
# shape 是可选项 但是可以帮助找bug  例如图片尺寸不符合
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# 模型参数 一般用variable表示 可以在计算图中存在 变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 变量 需要在Session 中初始化 才能在session 中使用
# 批量化初始化variables
sess.run(tf.global_variables_initializer())

# 回归模型
y = tf.nn.softmax(tf.matmul(x,W)+b)

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# 定义训练方式和目标
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
