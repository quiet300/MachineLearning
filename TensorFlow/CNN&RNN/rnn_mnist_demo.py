# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import  sys

mnist = input_data.read_data_sets('data/', one_hot=True)

train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels

# 哑编码后分多少类
n_class = train_label.shape[1]
# 样本维度
feature_num = train_image.shape[1]

# 学习率
learning_rate = 0.001
# 每一时刻输入的数据维度大小(这里设置为1次输入1行)
input_size = 28
# 时刻数目，总共输入多少次
timestep_size = 28
# 细胞中一个神经网络的层次中的神经元的数目
hidden_size = 128
# RNN中的隐层的数目
layer_num = 2

_X = tf.placeholder(tf.float32, shape=[None, feature_num])
y = tf.placeholder(tf.float32, shape=[None, n_class])

# 一次输入的样本数
batch_size = tf.placeholder(tf.int32, shape=[])
# dropout的时候，保留率多少
keep_prob = tf.placeholder(tf.float32, [])

# 1. 输入数据格式转换
X = tf.reshape(_X, shape=[-1, timestep_size, input_size])

# 2.多层LSTM RNN
def lstm_cell():
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

# cells参数给定的是每一层的cell，有多少层就给多少个cell，但是cell的类型不做要求
mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell() for i in range(layer_num)])

# 3. 给定初始化状态信息
init_state = mlstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

# 4. 构建可以运行的网络结构(加入时间)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state)
# 得到最后一个时刻对应的输出值
output = outputs[:, -1, :]

# 将输出值(最后一个时刻对应的输出值构建加下来的全连接)
w = tf.Variable(initial_value=tf.truncated_normal(shape=[hidden_size, n_class], mean=0.0, stddev=0.1), dtype=tf.float32, name='w')
b = tf.Variable(initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[n_class], name='b'))
y_pre = tf.nn.softmax(tf.add(tf.matmul(output, w), b))

# 定义损失函数
loss = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_pre), 1))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 准确率
cp = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(cp, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)

        # 训练模型
        feed_dict = {_X: batch[0], y: batch[1], keep_prob: 0.7, batch_size: _batch_size}
        sess.run(train, feed_dict=feed_dict)

        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("批次:{}, 步骤:{}, 训练集准确率:{}".format(mnist.train.epochs_completed, (i + 1), acc))

            if acc > 0.8:
                test_acc = sess.run(accuracy, feed_dict={_X: test_image, y: test_label, keep_prob: 0.7, batch_size: test_image.shape[0]})
                print("测试集准确率:{}".format(test_acc))

            if(acc > 0.95 and test_acc > 0.95):
                # 模型保存、持久化
                saver = tf.train.Saver()
                saver.save(sess, './mnist/RNN_model')
                break



