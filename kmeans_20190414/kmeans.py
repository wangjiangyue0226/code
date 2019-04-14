#!/usr/bin/env python
#_*_encoding:utf-8_*_

'''
典型的基于距离的聚类算法，采用距离作为相似性的评价指标，即认为两个对象的距离越近，其相似度越大。
第一步是随机选取任意k个对象作为初始聚类中心，初始地代表一个簇。
    在每次迭代中对数据集中剩余的每个对象，根据其与各个簇中心的距离将每个对象重新赋给最近的簇。当考察完所有数据对象后，依次迭代运算完成后，新的聚类中心将被计算出来。
    如果在一次迭代前后，J的值没有发生变化，说明算法已经收敛。
第二步是：当任意一个簇的分配结果发生改变的时候
第三步是：对数据集的每个数据点。
    对每个质心：
        计算质心与数据点之间的距离；
    将数据分配到距离其最近的簇
第四步是：对每一个簇，计算簇中所有点的均值，并将其均值作为质心。

'''

import numpy as np

class KMeansClassifier():
    '''
    this is a k-means classifier
    '''

    def __init__(self, k=3, initCent='random', max_iter=500):
        self._k = k
        self._initCent = initCent
        self._max_iter = max_iter
        self._clusterAssment = None
        self._labels = None
        self._sse = None

    def _calEDist(self, arrA, arrB):
        '''
        :param arrA: 数组A
        :param arrB: 数组B
        :return: 两个一维数组的欧式距离
        '''
        return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

    def _calMDist(self, arrA, arrB):
        '''
        :param arrA: 数组A
        :param arrB: 数组B
        :return: 返回两个一维数组的曼哈顿距离
        '''
        return sum(np.abs(arrA-arrB))

    def _randCent(self, data_X, k):
        '''
        :param data_X: 输入数据
        :param k: 质心个数
        :return: 返回一个k*n的质心矩阵
        '''
        n = data_X.shape[1]    # 获取特征的维数
        centroids = np.empty((k, n))    # 使用numpy生成一个k*n的矩阵，用于存储质心
        for j in range(n):
            minJ = min(data_X[:, j])    # j=0时，是第一列，第一个维度的最小值
            rangeJ = float(max(data_X[:, j] - minJ))    # j=0时，表示第一个维度的范围（最大值-最小值）
            # 使用flatten拉平嵌套列表
            centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()    # j=0时，生成质心第一个维度向量。
        return centroids

    def fit(self, data_X):
        '''
        :param data_X: 输入数据
        :return: 一个m*n的矩阵
        '''
        if not isinstance(data_X, np.ndarray) or isinstance(data_X, np.matrixlib.defmatrix.matrix):
            try:
                data_X = np.asarray(data_X)
            except:
                raise TypeError("numpy.ndarray resuired for data_X")

        m = data_X.shape[0]    # 获取样本个数
        # 一个 m * 2 的二维矩阵，矩阵的第一列存储样本点所属的簇的索引值，
        # 第二列存储该点与所属簇的质心的平方误差。
        self._clusterAssment = np.zeros((m, 2))

        if self._initCent == 'random':
            self._centroids = self._randCent(data_X, self._k)

        clusterChanged = True
        for _ in range(self._max_iter):    # 使用"_"是因为后面没有用到这个值
            clusterChanged = False
            # 对第 i 个样本寻找与其最近的簇，并更新其聚类簇及误差。
            for i in range(m):    # 将每个样本点分配到离它最近的质心所属的簇
                minDist = np.inf    # 首先将minDist置为一个无穷大的数
                minIndex = -1    # 将最近质心的下标置为-1
                for j in range(self._k):    # 次迭代用于寻找最近的质心
                    arrA = self._centroids[j, :]    # 第j个质心点
                    arrB = data_X[i, :]    # 第i个数据样本
                    distJI = self._calEDist(arrA,arrB)    # 计算误差
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j

                if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist**2:
                    clusterChanged = True
                    self._clusterAssment[i, :] = minIndex, minDist**2

            if not clusterChanged:    # 若所有样本点所属的簇都不变，则已收敛，结束迭代
                break

            for i in range(self._k):    # 更新质心，将每个簇中的点的均值作为质心
                index_all = self._clusterAssment[:,0]    # 取出样本所属簇的索引值
                value = np.nonzero(index_all == i)    # 取出所有属于第i个簇的索引值，索引值对应 data_X 中的样本。
                ptsInClust = data_X[value[0]]    # 取出属于第i个簇的所有样本点
                self._centroids[i, :] = np.mean(ptsInClust, axis=0)    # 计算均值

        self._labels = self._clusterAssment[:, 0]    # 获取第j列，也就是第0列
        self._sse = sum(self._clusterAssment[:, 1])    # 获取第1列，为误差

    def predict(self, X):    # 根据聚类结果，预测新输入数据所属的簇
        # 类型检查
        if not isinstance(X, np.ndarry):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarry required for X")

        m = X.shape[0]    # m表示样本数量
        preds = np.empty((m,))
        for i in range(m):    # 将每个样本点分配到离它最近的质心所属的簇
            minDist = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._clusterAssment[j,:],X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds



