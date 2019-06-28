#-*-coding:utf-8-*-

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)

def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf):
    n_train, n_test = len(X_train), len(X_test)
    # 初始化权重
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        miss = [int(x) for x in (pred_train_i != Y_train)]

        miss2 = [x if x == 1 else -1 for x in miss]

        err_m = np.dot(w, miss) / sum(w)

        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # 预测
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    # 返回训练集和测试集的错误率
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


def plot_error_rate(er_train, er_test):

    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')
    plt.show()

if __name__ == '__main__':

    # 读数据
    x, y = make_hastie_10_2()
    df = pd.DataFrame(x)
    df['Y'] = y

    # 划分训练集和测试集
    train, test = train_test_split(df, test_size=0.2)
    X_train, Y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, Y_test = test.iloc[:, :-1], test.iloc[:, -1]

    # 首先拟合一个简单的决策树
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)

    # 基于决策树的自适应Adaboost分类器
    # 不同迭代次数的测试
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    x_range = range(10, 410, 10)
    for i in x_range:
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])
        print_error_rate(er_i)
    # 错误率与迭代次数
    plot_error_rate(er_train, er_test)
