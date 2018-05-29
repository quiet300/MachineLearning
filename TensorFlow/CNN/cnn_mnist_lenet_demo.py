# coding:utf-8

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys

# 数据加载
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

# 手写数字识别的数据集主要包含三个部分：训练集(5.5w, mnist.train)、测试集(1w, mnist.test)、验证集(0.5w, mnist.validation)
# 手写数字图片大小是28*28*1像素的图片(黑白)，也就是每个图片由784维的特征描述
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
train_sample_number = mnist.train.num_examples

# 学习率(不能过大，否则可能不收敛，)
# GradientDescentOptimizer(随机梯度下降时)之前试过1.0和0.5都不收敛,0.05收敛到0.891,0.02收敛到0.893,0.001收敛到0.8045
# 用AdamOptimizer做梯度下降的时候学习率不能设置太大，不然不收敛。0.0001的时候9次收敛到0.9010;21次测试集准确率0.9858
learn_rate_set = 0.0001
# 迭代训练样本数量
batch_size = 100

feature_num = train_img.shape[1]
n_class = train_label.shape[1]

x = tf.placeholder(tf.float32, shape=[None, feature_num], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_class], name='y')
learn_rate = tf.placeholder(tf.float32, name='learn_rate')

def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    """
    返回一个对应的变量
    :param name:
    :param shape:
    :param dtype:
    :param initializer:
    :return:
    """
    return tf.get_variable(name, shape, dtype, initializer)

def learn_rate_func(epoch):
    """
    根据给定的迭代批次，更新产生一个学习率的值
    :param epoch:
    :return:
    """
    if (epoch % 10 == 0):
        return learn_rate_set * (0.9 ** int(epoch / 10))
    else:
        return learn_rate_set

def le_net(x, y):
    '''
    循环网络
    :param x:
    :param y:
    :return:
    '''
    # 第1层 输入层
    with tf.variable_scope('input1'):
        # 将输入的x的格式转换为规定的格式
        # [None, input_dim] -> [None, height, weight, channels](黑白图片，channel=1)
        net = tf.reshape(x, shape=[-1, 28, 28, 1])

    # 第2层 卷积层
    with tf.variable_scope('conv1'):
        # conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", name=None) => 卷积的API
        # data_format: 表示的是输入的数据格式，两种：NHWC和NCHW，N=>样本数目，H=>Height, W=>Weight, C=>Channels
        # input：输入数据，必须是一个4维格式的图像数据，具体格式和data_format有关，如果data_format是NHWC的时候，
        # input的格式为: [batch_size, height, weight, channels] => [批次中的图片数目，图片的高度，图片的宽度，图片的通道数]；
        # 如果data_format是NCHW的时候，input的格式为: [batch_size, channels, height, weight] => [批次中的图片数目，图片的通道数，图片的高度，图片的宽度]

        # filter: 卷积核，是一个4维格式的数据，shape: [height, weight, in_channels, out_channels] => [窗口的高度，窗口的宽度，输入的channel通道数(上一层图片的深度)，输出的通道数(卷积核数目)]

        # strides：步长，是一个4维的数据，每一维数据必须和data_format格式匹配，表示的是在data_format每一维上的移动步长，
        # 当格式为NHWC的时候，strides的格式为: [batch, in_height, in_weight, in_channels] => [样本上的移动大小，高度的移动大小，宽度的移动大小，深度的移动大小],要求在样本上和在深度通道上的移动必须是1
        # 当格式为NCHW的时候，strides的格式为: [batch,in_channels, in_height, in_weight]

        # padding: 只支持两个参数"SAME", "VALID"，当取值为SAME的时候，表示进行填充，
        #          在TensorFlow中，如果步长为1，并且padding为SAME的时候，经过卷积之后的图像大小是不变的
        #          当VALID的时候，表示多余的特征会丢弃
        net = tf.nn.conv2d(net, filter=get_variable('w', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [20]))
        net = tf.nn.relu(net)
    # 第3层 池化层
    with tf.variable_scope('pool1'):
        # 和conv2一样，需要给定窗口大小和步长
        # max_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
        # avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None)
        # 默认格式下：NHWC，value：输入的数据，必须是[batch_size, height, weight, channels]格式
        # 默认格式下：NHWC，ksize：指定窗口大小，必须是[batch, in_height, in_weight, in_channels]， 其中batch和in_channels必须为1
        # 默认格式下：NHWC，strides：指定步长大小，必须是[batch, in_height, in_weight, in_channels],其中batch和in_channels必须为1
        # padding： 只支持两个参数"SAME", "VALID"，当取值为SAME的时候，表示进行填充，；当VALID的时候，表示多余的特征会丢弃；
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第4层 卷积层
    with tf.variable_scope('conv2'):
        net = tf.nn.conv2d(net, filter=get_variable('w', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, bias=get_variable('b', [50]))
        net = tf.nn.relu(net)
    # 第5层 池化层
    with tf.variable_scope('pool2'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # 第6层 全连接
    with tf.variable_scope('fc1'):
        net = tf.reshape(net, shape=[-1, 7 * 7 * 50])
        net = tf.add(tf.matmul(net, get_variable('w', [7 * 7 * 50, 500])), get_variable('b', [500]))
        net = tf.nn.relu(net)
    # 第7层 全连接
    with tf.variable_scope('fc2'):
        net = tf.add(tf.matmul(net, get_variable('w', [500, n_class])), get_variable('b', [n_class]))
        act = tf.nn.softmax(net)

    return act

def test_mnist(validation_feature, validation_label):
    '''
    预测
    :param validation_feature:
    :param validation_label:
    :return:
    '''
    tmp_i = 4999
    output = le_net(X, Y)
    saver = tf.train.Saver()
    img = validation_feature[tmp_i]
    label = validation_label[tmp_i]

    print('validation_label[tmp_i]==', label)
    with tf.Session() as sess:
        # 预测的时候不能初始化，或者说，载入模型之后不能再初始化，因为w和b已经存在了
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./mnist/model")

        predict = tf.argmax(output, axis=1)
        text_list = sess.run(predict, feed_dict={X: [img], Y: [label]})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    train = 1
    if train == 0:

        act = le_net(x, y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=act, labels=y))

        train = tf.train.AdamOptimizer(learn_rate).minimize(cost)

        pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))

        acc = tf.reduce_mean(tf.cast(pred, tf.float32))

        # 初始化
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # 模型保存、持久化
            saver = tf.train.Saver()

            epoch = 0

            while True:
                avg_cost = 0
                # 计算出总的批次
                total_batch = int(train_sample_number / batch_size)

                # 迭代更新
                for i in range(total_batch):
                    # 获取x和y
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    feeds = {x: batch_xs, y: batch_ys, learn_rate: learn_rate_func(epoch)}
                    # 模型训练
                    sess.run(train, feed_dict=feeds)
                    # 获取损失函数值
                    avg_cost += sess.run(cost, feed_dict=feeds)

                # 重新计算平均损失(相当于计算每个样本的损失值)
                avg_cost = avg_cost / total_batch

                # DISPLAY  显示误差率和训练集的正确率以此测试集的正确率
                if (epoch + 1) % 1 == 0:
                    print("批次: %03d 损失函数值: %.9f" % (epoch, avg_cost))
                    # 这里之所以使用batch_xs和batch_ys，是因为我使用train_img会出现内存不够的情况，直接就会退出
                    feeds = {x: batch_xs, y: batch_ys, learn_rate: learn_rate_func(epoch)}
                    train_acc = sess.run(acc, feed_dict=feeds)
                    print("训练集准确率: %.4f" % train_acc)
                    feeds = {x: test_img, y: test_label, learn_rate: learn_rate_func(epoch)}
                    test_acc = sess.run(acc, feed_dict=feeds)
                    print("测试准确率: %.4f" % test_acc)

                    if train_acc > 0.95 and test_acc > 0.95:
                        saver.save(sess, './mnist/model')
                        break
                epoch += 1

            # 模型可视化输出
            writer = tf.summary.FileWriter('./mnist/graph', tf.get_default_graph())
            writer.close()
    if train == 1:
        # 预测
        validation_feature = mnist.validation.images
        validation_label = mnist.validation.labels

        X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
        Y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])

        v_y = test_mnist(validation_feature, validation_label)
        print(v_y)

