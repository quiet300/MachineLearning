# coding:utf-8

import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import train_test_split

np.random.seed(18)
# 1. 做数据
n = 1000
def create_data(n):
    # x是n行，3列的矩阵
    x = np.random.normal(loc=0, scale=1, size=(n, 3))
    # y = 3x1 - 5x2 + x3
    y = np.dot(x, np.array([[3.0], [-5.0], [1.0]])) + np.random.random(1)
    return x, y

def print_info(r_w, r_b, r_loss):
    print("w={},b={},loss={}".format(r_w, r_b, r_loss))

def train_module(x_train, y_train, xlenth):

    x_t = tf.placeholder(tf.float32, shape=[xlenth, 3])
    # 定义的两个变量
    w = tf.Variable(initial_value=tf.random_uniform(shape=([3, 1]), minval=-1.0, maxval=1.0), name='w', dtype=tf.float32)
    b = tf.Variable(tf.ones([1]), dtype=tf.float32)

    y_hat = tf.matmul(x_t, w) + b

    # 损失函数
    loss = tf.reduce_mean(tf.square(y_hat - y_train), name='loss')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

    train = optimizer.minimize(loss, name='train')

    # 初始化所有变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        r_w, r_b, r_loss = sess.run([w, b, loss], feed_dict={x_t: x_train})
        print_info(r_w, r_b, r_loss)

        i = 0
        while True:
            i += 1
            sess.run(train, feed_dict={x_t: x_train})
            # 输出训练后的w、b、loss
            r_w, r_b, r_loss = sess.run([w, b, loss], feed_dict={x_t: x_train})

            # print('第', i, '次', print_info(r_w, r_b, r_loss))

            if r_loss < 1e-4:
                break

        # 模型可视化输出
        writer = tf.summary.FileWriter('./module/graph', tf.get_default_graph())
        writer.close()
        return r_w, r_b, r_loss

def test_result(x_test, y_test, w, b):
    W = tf.placeholder(dtype=tf.float32)
    B = tf.placeholder(dtype=tf.float32)
    X = tf.placeholder(dtype=tf.float32)
    Y = tf.placeholder(dtype=tf.float32)

    pred = tf.add(tf.matmul(X, W), B)
    cost = tf.reduce_mean(tf.square(pred-Y))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        cost = sess.run(cost, feed_dict={W: w, B:b, X:x_test, Y:y_test})

    return cost


if __name__ == '__main__':
    # 造随机数据
    x, y = create_data(n)

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # 训练模型，找到w,b的值
    w, b, loss = train_module(x_train, y_train, len(x_train))

    print_info(w, b, loss)
    cost = test_result(x_test, y_test, w, b)
    print(cost)


