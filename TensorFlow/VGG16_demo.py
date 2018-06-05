# coding:utf-8
'''
VGG16模型实现
VGG16结构图：http://ethereon.github.io/netscope/#/gist/dc5003de6943ea5a6b8b
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 数据加载
mnist = input_data.read_data_sets('data/mnist', one_hot=True)

train_image = mnist.train.images
train_label = mnist.train.labels
test_image = mnist.test.images
test_label = mnist.test.labels

train_sample_number = mnist.train.num_examples

feature_num = train_image.shape[1]
n_class = train_label.shape[1]
print(n_class)

# 学习率
learn_rate_set = 0.0001
# 迭代训练样本数量
batch_size = 100

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

def vgg16_network(x, y):

    # 由于标准的VGG16参数过多，计算量过大，对下列参数进行调整
    net1_kernel_size = 64/2
    net2_kernel_size = 128/2
    net3_kernel_size = 256/2
    net4_kernel_size = 512/2
    net5_kernel_size = 512/2
    fc6_size = 4096/4
    fc7_size = 4096/4
    fc8_size = 10

    # input数组变换维度
    net = tf.reshape(x, shape=[-1, 28, 28, 1])
    # conv1:两个卷积
    with tf.variable_scope('conv1'):
        # 第一个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, 1, net1_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv1_1')
        net = tf.nn.bias_add(net, get_variable('b', [net1_kernel_size]))
        net = tf.nn.relu(net)

        # 第二个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net1_kernel_size, net1_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv1_2')
        net = tf.nn.bias_add(net, get_variable('b1', [net1_kernel_size]))
        net = tf.nn.relu(net)

    # pool1:池化
    with tf.variable_scope('pool1'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2:两个卷积
    with tf.variable_scope('conv2'):
        # 第一个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net1_kernel_size, net2_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv2_1')
        net = tf.nn.bias_add(net, get_variable('b', [net2_kernel_size]))
        net = tf.nn.relu(net)

        # 第二个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net2_kernel_size, net2_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv2_2')
        net = tf.nn.bias_add(net, get_variable('b1', [net2_kernel_size]))
        net = tf.nn.relu(net)

    # pool2:池化
    with tf.variable_scope('pool2'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3:三个卷积
    with tf.variable_scope('conv3'):
        # 第一个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net2_kernel_size, net3_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv3_1')
        net = tf.nn.bias_add(net, get_variable('b', [net3_kernel_size]))
        net = tf.nn.relu(net)

        # 第二个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net3_kernel_size, net3_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv3_2')
        net = tf.nn.bias_add(net, get_variable('b1', [net3_kernel_size]))
        net = tf.nn.relu(net)

        # 第三个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net3_kernel_size, net3_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv3_3')
        net = tf.nn.bias_add(net, get_variable('b2', [net3_kernel_size]))
        net = tf.nn.relu(net)

    # pool3:池化
    with tf.variable_scope('pool3'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    # conv4:三个卷积
    with tf.variable_scope('conv4'):
        # 第一个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net3_kernel_size, net4_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv4_1')
        net = tf.nn.bias_add(net, get_variable('b', [net4_kernel_size]))
        net = tf.nn.relu(net)

        # 第二个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net4_kernel_size, net4_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv4_2')
        net = tf.nn.bias_add(net, get_variable('b1', [net4_kernel_size]))
        net = tf.nn.relu(net)

        # 第三个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net4_kernel_size, net4_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv4_3')
        net = tf.nn.bias_add(net, get_variable('b2', [net4_kernel_size]))
        net = tf.nn.relu(net)

    # pool4:池化
    with tf.variable_scope('pool4'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5:三个卷积
    with tf.variable_scope('conv5'):
        # 第一个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w', [3, 3, net4_kernel_size, net5_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv5_1')
        net = tf.nn.bias_add(net, get_variable('b', [net5_kernel_size]))
        net = tf.nn.relu(net)

        # 第二个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w1', [3, 3, net5_kernel_size, net5_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv5_2')
        net = tf.nn.bias_add(net, get_variable('b1', [net5_kernel_size]))
        net = tf.nn.relu(net)

        # 第三个卷积
        net = tf.nn.conv2d(net, filter=get_variable('w2', [3, 3, net5_kernel_size, net5_kernel_size]), strides=[1, 1, 1, 1], padding='SAME', name='conv5_3')
        net = tf.nn.bias_add(net, get_variable('b2', [net5_kernel_size]))
        net = tf.nn.relu(net)

    # pool5:池化
    with tf.variable_scope('pool5'):
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    # fc6:全连接+relu+dropout
    with tf.variable_scope('fc6'):
        # 将四维的数据转换为两维的数据
        shape = net.get_shape()
        feature_number = shape[1] * shape[2] * shape[3]
        net = tf.reshape(net, shape=[-1, feature_number])
        # 全连接
        net = tf.add(tf.matmul(net, get_variable('w', [feature_number, fc6_size])), get_variable('b', [fc6_size]))
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, keep_prob=0.5)

    # fc7:全连接+relu+dropout
    with tf.variable_scope('fc7'):
        net = tf.add(tf.matmul(net, get_variable('w', [fc6_size, fc7_size])), get_variable('b', [fc7_size]))
        net = tf.nn.relu(net)
        net = tf.nn.dropout(net, keep_prob=0.5)

    # fc8:全连接+prob
    with tf.variable_scope('fc8'):
        net = tf.add(tf.matmul(net, get_variable('w', [fc7_size, fc8_size])), get_variable('b', [fc8_size]))
        act = tf.nn.softmax(net)

    return act

if __name__ == '__main__':
    train = 0
    if train == 0:
        act = vgg16_network(x, y)

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
                    feeds = {x: test_image, y: test_label, learn_rate: learn_rate_func(epoch)}
                    test_acc = sess.run(acc, feed_dict=feeds)
                    print("测试准确率: %.4f" % test_acc)

                    if train_acc > 0.95 and test_acc > 0.95:
                        saver.save(sess, './mnist/vgg_model{}'.format(epoch))
                        break
                epoch += 1

            # 模型可视化输出
            writer = tf.summary.FileWriter('./mnist/vgg_graph', tf.get_default_graph())
            writer.close()