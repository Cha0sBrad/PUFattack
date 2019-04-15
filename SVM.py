import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd


def svm_c(x_train, x_test, y_train, y_test):
    # rbf核函数，设置数据权重
    svc = SVC(kernel='rbf', class_weight='balanced', degree=3.0)
    c_range = np.logspace(-5, 15, 11, base=2)
    gamma_range = np.logspace(-9, 3, 13, base=2)# 网格搜索交叉验证的参数范围，cv=3,3折交叉
    param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]
    grid = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1)
    # 训练模型
    clf = grid.fit(x_train, y_train.values.ravel())
    # 计算测试集精度
    score = grid.score(x_test, y_test.values.ravel())
    print("此次模型精度为 %s"%score)


if __name__ == '__main__':
    x_test = pd.read_csv("testdatasetin.csv", header=None)
    x_train = pd.read_csv("traindatasetin.csv", header=None)
    y_test = pd.read_csv("testdatasetout.csv", header=None)
    y_train = pd.read_csv("traindatasetout.csv", header=None)
    svm_c(x_train, x_test, y_train, y_test)