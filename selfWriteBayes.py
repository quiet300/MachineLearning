#coding:utf-8

import numpy as np
import sys

'''
分别用贝努利，多项式，混合型对数据进行分类
'''

def createWordList(data):
    '''
    去掉重复项，取得样本种类
    :param data:样本
    :return: 样本的种类
    '''
    wordSet = set([])
    for item in data:
        wordSet = wordSet | set(item)

    return list(wordSet)

# 将多维数据转化为一维向量，方便计算
def word2Vec(wordList,inputWord):
    returnVec = []

    for i in range(len(inputWord)):
        tempval = [0]*len(wordList)

        for word in inputWord[i]:
            if word in wordList:
                tempval[wordList.index(word)] = 1
        returnVec.append(tempval)

    return returnVec

def train_fit(train_data,train_label,feature,type=0):
    '''
    :param train_data:训练模型用数据
    :param train_label: 训练模型用标签
    :param feature: 待测试样本
    :param type: 模型类型：0 贝努利型、1 多项式型、2 混合型、3 朴素贝叶斯
    :return:
    '''
    # 取得总标签个数
    y_len = float(len(train_label))
    # 取得标签分类
    y_class = np.unique(train_label)

    # 计算p(y=1)和P(y=-1)的概率
    class_prior = []

    if type == 0:
        # 贝努利型
        print('贝努利型')
        # 取得所有样本种类
        xlist = createWordList(train_data)
        classBianhuang = word2Vec(xlist, train_data)

        # 得到标签为1和1以外的样本个数
        p1Denom = np.sum(np.equal(1, train_label))
        p0Denom = len(train_label) - p1Denom

        # 得到样本的个数
        classnum = len(classBianhuang)

        # 得到样本中1的概率
        p1per = p1Denom / float(classnum)

        # 避免一个概率值为0,最后的乘积也为0
        p0Num = np.ones(len(xlist))
        p1Num = np.ones(len(xlist))

        for i in range(classnum):
            if train_label[i] == 1:
                p1Num += classBianhuang[i]
            else:
                p0Num += classBianhuang[i]

        p0Vect = np.log(p0Num / p0Denom)
        p1Vect = np.log(p1Num / p1Denom)

        tempval = [0] * len(xlist)

        for word in feature:
            if word in xlist:
                tempval[xlist.index(word)] = 1

        p1 = sum(tempval * p1Vect) + np.log(p1per)
        p0 = sum(tempval * p0Vect) + np.log(1.0 - p1per)

        print('1的概率：', p1)
        print('-1的概率：', p0)
        if p1 > p0:
            return 1
        else:
            return -1

    elif type == 1:
        # 多项式型
        print('多项式型')
        for c in y_class:
            num = np.sum(np.equal(c, train_label))
            k = ((num + 1) / (y_len + len(y_class)))
            class_prior.append(k)

        data_prior = {}
        for c in y_class:
            data_prior[c] = {}
            for i in range(len(train_data[0])):
                xfeature = train_data[np.equal(train_label, c)][:, i]
                data_prior[c][i] = {}

                # 取得所有属于c标签的样本
                tmp_feature = np.unique(xfeature)
                for m in tmp_feature:
                    # 计算不同种类的样本在该标签下的概率值(拉普拉斯平滑)
                    cPrior = (np.sum(np.equal(m, xfeature)) + 1) / (float(len(xfeature)) + len(tmp_feature))
                    data_prior[c][i][m] = cPrior

        # 取得了所有概率，开始计算待测试样本概率
        # 初始化一个类别
        label = -1
        # 存计算出的分类概率
        maxPri = 0

        for c in range(len(y_class)):

            # 取得标签概率
            label_prior = class_prior[c]

            # 取计算出的样本分类的概率
            c_pri = data_prior[y_class[c]]

            # 计算概率
            jisuangailv = label_prior

            i = 0
            for item in c_pri.keys():
                fe = feature[i]
                if fe in c_pri[item]:
                    jisuangailv *= c_pri[item][fe]
                else:
                    jisuangailv *= 1
                i += 1

            if maxPri < jisuangailv:
                maxPri = jisuangailv
                label = y_class[c]
            print('分类：', y_class[c], '  概率：', jisuangailv)

        return label

    elif type == 2:
        # 混合型
        print('混合型')

        # 用贝努利训练模型
        # 取得所有样本种类
        xlist = createWordList(train_data)
        classBianhuang = word2Vec(xlist, train_data)

        # 得到标签为1和1以外的样本个数
        p1Denom = np.sum(np.equal(1, train_label))
        p0Denom = len(train_label) - p1Denom

        # 得到样本的个数
        classnum = len(classBianhuang)

        # 得到样本中1的概率
        p1per = p1Denom / float(classnum)

        # 避免一个概率值为0,最后的乘积也为0
        p0Num = np.ones(len(xlist))
        p1Num = np.ones(len(xlist))

        for i in range(classnum):
            if train_label[i] == 1:
                p1Num += classBianhuang[i]
            else:
                p0Num += classBianhuang[i]

        p0Vect = p0Num / p0Denom
        p1Vect = p1Num / p1Denom

        tempval = [0] * len(xlist)

        # 把[X1, X2]类型的样本转化成[0, 1, 0, 1, 0, 1]类型
        for word in feature:
            if word in xlist:
                tempval[xlist.index(word)] = 1

        # 用多项式预测

        # 计算样本属于1的概率
        px1per = p1per
        for i in range(len(tempval)):
            if tempval[i] == 1:
                px1per *= p1Vect[i]
        print('类别是1的概率：', px1per)

        # 计算样本属于-1的概率
        px0per = 1 - p1per
        for i in range(len(tempval)):
            if tempval[i] == 1:
                px0per *= p0Vect[i]
        print('类别是-1的概率：', px0per)

        if px1per > px0per:
            return 1
        else:
            return -1

    elif type == 3:
        # 朴素贝叶斯
        print('朴素贝叶斯')
        for c in y_class:
            num = np.sum(np.equal(c, train_label))
            class_prior.append(num / y_len)

        data_prior = {}
        for c in y_class:
            data_prior[c] = {}
            for i in range(len(train_data[0])):
                xfeature = train_data[np.equal(train_label, c)][:, i]
                data_prior[c][i] = {}

                # 取得所有属于c标签的样本
                tmp_feature = np.unique(xfeature)
                for m in tmp_feature:
                    # 计算不同种类的样本在该标签下的概率值
                    cPrior = np.sum(np.equal(m, xfeature)) / float(len(xfeature))
                    data_prior[c][i][m] = cPrior

        # 取得了所有概率，开始计算待测试样本概率
        # 初始化一个类别
        label = -1
        # 存计算出的分类概率
        maxPri = 0

        for c in range(len(y_class)):
            # 取得标签概率
            label_prior = class_prior[c]

            # 取计算出的样本分类的概率
            c_pri = data_prior[y_class[c]]

            # 计算概率
            jisuangailv = label_prior

            i = 0
            for item in c_pri.keys():
                fe = feature[i]
                if fe in c_pri[item]:
                    jisuangailv *= c_pri[item][fe]
                else:
                    jisuangailv *= 1
                i += 1
            print('分类：', y_class[c], '  概率：', jisuangailv)
            if maxPri < jisuangailv:
                maxPri = jisuangailv
                label = y_class[c]

        return label



if __name__ == '__main__':
    # S=4 M=5 L=6
    train_data = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ])
    # 转置是为了上面样本与标签对应上
    train_data = train_data.T
    train_label = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    feature = np.array([2,4])

    # 贝努利朴素贝叶斯
    print(train_fit(train_data, train_label, feature, type=0))
    print('*'*30)
    # 多项式朴素贝叶斯
    print(train_fit(train_data, train_label, feature,type=1))
    print('*'*30)
    # 混合朴素贝叶斯
    print(train_fit(train_data, train_label, feature,type=2))
    print('*'*30)
    # 朴素贝叶斯
    print(train_fit(train_data, train_label, feature, type=3))
