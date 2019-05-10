#!/usr/bin/env python
#_*_encoding:utf-8_*_
import numpy as np
from kmeans import KMeansClassifier
class biKMeansClassifier():
    '''
    标准KMeans算法的簇数K是提前给定的，但是实际中K值是非常难以估计的，提出二分K-means算法。
    该算法首先将所有点看作一个簇，然后一分为二， 找到最小sse的聚类中心。接着选择其中一个簇继续一分为二，此处哪一个簇需要根据划分后的SSE值来判断。
    但是：
    该算法对离群点敏感；
    结果不稳定（受输入顺序影响）
    时间复杂度高O（nkt），其中n是对象总数，k是簇数,t是迭代次数。
    '''
    def __init__(self, k=3):
        self._k = k
        self._centroids = None
        self._clusterAssment = None
        self._labels = None
        self._sse = None

    def _calEDist(self, arrA, arrB):
        return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

    def fit(self, X):
        m = X.shape[0]
        self._clusterAssment = np.zeros((m, 2))
        centroid0 = np.mean(X, axis=0).tolist()
        cenList = [centroid0]
        for j in range(m):    # 计算每个样本点与质心之间初始的平方误差
            self._clusterAssment[j, 1] = self._calEDist(np.asarray(centroid0), X[j, :])**2

        while(len(cenList) < self._k):
            lowestSSE = np.inf
            for i in range(len(cenList)):
                index_all = self._clusterAssment[:, 0]    # 取出样本所属簇的索引值
                value = np.nonzero(index_all == i)    # 取出所有属于第i个簇的索引点
                pstInCurrCluster = X[value[0], :]    # 取出属于第i个簇的所有样本点
                clf = KMeansClassifier(k=2)
                clf.fit(pstInCurrCluster)
                centroidMat, splitClustAss = clf._centroids, clf._clusterAssment
                sseSplit = sum(splitClustAss[:, 1])
                index_all = self._clusterAssment[:,0]
                value = np.nonzero(index_all == i)
                sseNotSplit = sum(self._clusterAssment[value[0], 1])
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit

            # 该簇被划分为两个子簇后，其中一个子簇的索引变为原簇的索引
            # 另一个子簇的索引变为len(cenList)，然后存入cenList。
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(cenList)
            bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
            cenList[bestCentToSplit] = bestNewCents[0, :].tolist()
            cenList.append(bestNewCents[1, :]).tolist()
            self._clusterAssment[np.nonzero(self._clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss

        self._labels = self._clusterAssment[:, 0]
        self._sse = sum(self._clusterAssment[:, 1])
        self._centroids = np.asarray(cenList)

    def predict(self, X):    # 根据聚类结果，预测新输入数据所属的簇
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise ValueError("numpy.ndarray required for X")

        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):
            minDst = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._centroids[j, :], X[i, :])
                if distJI < minDst:
                    minDst = distJI
                    preds[i] = j
        return preds
