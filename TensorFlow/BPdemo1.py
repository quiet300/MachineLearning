# coding:utf-8
# BP神经网络，有2个输入特征（l1, l2），隐层有3个节点(h1, h2, h3)，输出有2个(o1, o2)
# 其中：
#    l1->h1 是w1
#    l2->h1 是w2
#    l1->h2 是w3
#    l2->h2 是w4
#    l1->h3 是w5
#    l2->h3 是w6
#    h1->o1 是w7
#    h1->o2 是w8
#    h2->o1 是w9
#    h2->o2 是w10
#    h3->o1 是w11
#    h3->o2 是w12
# l1,l2为[5, 10] o1,o2为[0.01, 0.99]
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

w = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
b = [0.35, 0.65]

l = [5, 10]
learn_rate = 0.05

def step1(w, b, l, learn_rate=0.5):
    # 前向传播
    h1 = sigmoid(w[0] * l[0] + w[1] * l[1] + b[0])
    h2 = sigmoid(w[2] * l[0] + w[3] * l[1] + b[0])
    h3 = sigmoid(w[4] * l[0] + w[5] * l[1] + b[0])

    o1 = sigmoid(w[6] * h1 + w[8] * h2 + w[10] * h3 + b[1])
    o2 = sigmoid(w[7] * h1 + w[8] * h2 + w[11] * h3 + b[1])
    # print('h1={},h2={},h3={},o1={},o2={}'.format(h1, h2, h3, o1, o2))

    # 反向传播
    # 隐层的损失函数 Eo = 1/2(o - o_hat)^2 -> 求导=-(o - o_hat)*o_hat*(1-o_hat)
    detlah1 = -(0.01-o1)*o1*(1-o1)
    detlah2 = -(0.99-o2)*o2*(1-o2)

    w[6] = w[6] - learn_rate * (detlah1 * h1)
    w[8] = w[8] - learn_rate * (detlah1 * h2)
    w[10] = w[10] - learn_rate * (detlah1 * h3)
    w[7] = w[7] - learn_rate * (detlah2 * h1)
    w[9] = w[9] - learn_rate * (detlah2 * h2)
    w[11] = w[11] - learn_rate * (detlah2 * h3)

    # 输入层损失函数
    w[0] = w[0] - learn_rate * (detlah1 * w[6] + detlah2 * w[7]) * h1 * (1 - h1) * l[0]
    w[1] = w[1] - learn_rate * (detlah1 * w[6] + detlah2 * w[7]) * h1 * (1 - h1) * l[1]
    w[2] = w[2] - 0.5 * (detlah1 * w[8] + detlah2 * w[9]) * h2 * (1 - h2) * l[0]
    w[3] = w[3] - 0.5 * (detlah1 * w[8] + detlah2 * w[9]) * h2 * (1 - h2) * l[1]
    w[4] = w[4] - 0.5 * (detlah1 * w[10] + detlah2 * w[11]) * h3 * (1 - h3) * l[0]
    w[5] = w[5] - 0.5 * (detlah1 * w[10] + detlah2 * w[11]) * h3 * (1 - h3) * l[1]

    return o1, o2, w


for i in range(1000001):
    r1, r2, w = step1(w, b, l, learn_rate)
    print("第{}次迭代后，结果值为:({},{}),权重更新为:{}".format(i, r1, r2, w))

# 第1000000次迭代后，结果值为:(0.010000208234153936,0.9899997925329185),百万次已经解决真实值
# 权重更新为:[0.26462116918241213, 0.47924233836483698, -0.19645944000335569, -0.54291888000669186, 0.43810447245120376, 0.62620894490240586, -2.6558277638937922, 1.8636109574755999, 0.36903410046082735, 0.5652358920215127, -2.5946457582684568, 2.0838385646011188]