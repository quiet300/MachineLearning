#coding:utf-8

import numpy as np

def createWordList(data):
    wordSet = set([])
    for item in data:
        wordSet = wordSet | set(item)

    return list(wordSet)

# 将多维数据转化为一维向量，方便计算
def word2Vec(wordList,inputWord):
    returnVec = [0]*len(wordList)
    for word in inputWord:
        if word in wordList:
            returnVec[wordList.index(word)] = 1
    return returnVec

def train_fit(train_data,train_label,feature,type=0):
    '''
    :param train_data:训练模型用数据
    :param train_label: 训练模型用标签
    :param feature: 待测试样本
    :param type: 模型类型：0 贝努利型、1 多项式型、2 混合型、3 朴素贝叶斯、4 高斯分布
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
        # 取得所有样本种类
        xlist = createWordList(train_data)
        print(word2Vec(xlist, train_data))



        pass
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
            print('分类是：',label, '  概率：',jisuangailv)

        return label

    elif type == 2:
        # 混合型






        pass
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

            if maxPri < jisuangailv:
                maxPri = jisuangailv
                label = y_class[c]
            print('分类是：',label, '  概率：',jisuangailv)
        return label

    elif type == 4:
        # 高斯贝叶斯
        pass


if __name__ == '__main__':
    # S=4 M=5 L=6
    train_data = np.array([
        [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        [4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6]
    ])
    # 转置是为了上面样本与标签对应上
    train_data = train_data.T
    train_label = np.array([-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1])

    feature = np.array([2,7])
    # 贝努利方法
    # print(train_fit(train_data, train_label, feature, type=0))
    # 多项式方法
    print(train_fit(train_data, train_label,feature,type=1))
    print('*'*30)
    # 朴素贝叶斯
    print(train_fit(train_data, train_label, feature, type=3))
