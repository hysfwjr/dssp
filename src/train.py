# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08
# Load libraries
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.svm import SVC
import numpy as np
import xgboost as xgb
# 特征选择，卡方检验
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Load dataset
testset = pd.read_csv('/Users/wenjurong/github/dssp/data/data_test_B.sample', quoting=3, error_bad_lines=False)
dataset = pd.read_csv('/Users/wenjurong/github/dssp/data/data_train.sample', quoting=3, error_bad_lines=False)

Y = dataset.target
X = dataset.as_matrix()[:, 1:-1]
test_X = testset.as_matrix()[:, 1: ]


# 选择最好的2500个特征
feat_sel = SelectKBest(chi2, k=3000)
X_new = feat_sel.fit_transform(X, Y)
test_X_new = feat_sel.transform(test_X)

#X_new = X

clf = MultinomialNB(alpha=0.4, class_prior=None, fit_prior=True).fit(X_new, Y) # multinomially分布数据的贝叶斯
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X_new.shape[0])

#from sklearn.linear_model import SGDClassifier, Perceptron
#clf = Perceptron()

scores = cross_val_score(clf, X_new, Y, cv=5) # 5折交叉验证

print 'multinomially分布数据的贝叶斯: {}, avg: {}'.format(scores, np.mean(scores))

# stacking
ntrain = X_new.shape[0]
ntest = test_X_new.shape[0]
kf = KFold(n_splits=5, shuffle=False, random_state=2017)


def get_oof(clf, X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]
        kf_y_train = y_train[train_index]
        kf_X_test = X_train[test_index]
        
        clf.fit(kf_X_train, kf_y_train)
        
        oof_train[test_index] = clf.predict(kf_X_test)
        oof_test_skf[i, :] = clf.predict(X_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
# model1
clf = MultinomialNB()
m1_train, m1_test = get_oof(clf, X_new, Y, test_X_new)
print('stacking MultinomialNB finish')
# model2
clf = Perceptron()
m2_train, m2_test = get_oof(clf, X_new, Y, test_X_new)
print('stacking Perceptron finish')

clf = AdaBoostClassifier()
m3_train, m3_test = get_oof(clf, X_new, Y, test_X_new)

print('stacking finish')

# 独立使用fisher特征
dataset_fisher = pd.read_csv('data/data_train_fisher.sample')
testset_fisher = pd.read_csv('data/data_test_fisher_B.sample')
Y_fisher = dataset_fisher.target
X_fisher = dataset_fisher.as_matrix()[:, : -1]
test_X_fisher = testset_fisher.as_matrix()[:, :]

#clf = svm.SVC(kernel='linear', gamma=2).fit(X_fisher, Y_fisher) # svm
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6).fit(X, Y)
#clf = MultinomialNB().fit(X_fisher, Y_fisher) # multinomially分布数据的贝叶斯

scores = cross_val_score(clf, X_fisher, Y_fisher, cv=5) # 5折交叉验证
print 'lr: {}, avg: {}'.format(scores, np.mean(scores))

m4_train, m4_test = get_oof(clf, X_fisher, Y, test_X_fisher)

# xgboost 融合
ntrain = X_new.shape[0]
ntest = test_X_new.shape[0]
kf = KFold(n_splits=5, shuffle=False, random_state=2017)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.4
param['max_depth'] = 1
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 3
param['booster'] = 'gblinear' # gbtree->gblinear, 训练从0.67 -> 0.83
num_round = 10

def get_oof2(X_train, y_train, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    scores_cross = []
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        kf_X_train = X_train[train_index]
        kf_y_train = y_train[train_index]
        kf_X_test = X_train[test_index]
        
        dtrain = xgb.DMatrix(kf_X_train, kf_y_train)
        dtest = xgb.DMatrix(X_test)
        bst = xgb.train(param, dtrain, num_round)       
        
        oof_train[test_index] = bst.predict(xgb.DMatrix(kf_X_test))
        oof_test_skf[i, :] = bst.predict(dtest)
        scores = f1_score(y_train[test_index], oof_train[test_index], average='micro')
        scores_cross.append(scores)        
        
    oof_test[:] = oof_test_skf.mean(axis=0)
    print('cross f1: {}, mean: {}'.format(scores_cross, np.mean(scores_cross)))
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

m5_train, m5_test = get_oof2(X_new, Y, test_X_new)

# 亚光结果
m6_train = pd.read_csv('~/Downloads/data_train_result_SVM_200_cx_1').as_matrix()[:, 1]
m6_test = pd.read_csv('~/Downloads/data_test_B_SVM_200_result_average.csv').as_matrix()[:, 1]
#print predict
scores = f1_score(Y, m6_train, average='micro')
print scores
#print m6_train

#s_train = pd.DataFrame({'m1': m1_train[:, 0], 'm2': m2_train[:, 0], 'm3': m3_train[:, 0],'target': Y})
s_train = pd.DataFrame({'m1': m1_train[:, 0], 'm2': m2_train[:, 0], 'm3': m3_train[:, 0], 'm4': m4_train[:, 0],'m5': m5_train[:, 0], 'm6': m6_train})
s_test = pd.DataFrame({'m1': m1_test[:, 0], 'm2': m2_test[:, 0], 'm3': m3_test[:, 0], 'm4': m4_test[:, 0], 'm5': m5_test[:, 0], 'm6': m6_test})
clf = AdaBoostClassifier()

# 两两查看
for i in range(len(s_train.columns)):
    columni = s_train.columns[i]
    s_train_i = s_train[[columni]]
    s_test_i = s_test[[columni]]
    clf = clf.fit(s_train_i, Y)
    predict_i = clf.predict(s_train_i)
    scores_i = f1_score(Y, predict_i, average='micro')
    print('stacking train result {}: {}'.format(columni, scores_i))
    
    for j in range(i + 1, len(s_train.columns)):        
        cloumnj = s_train.columns[j]
        #if columni == 'm5' or cloumnj == 'm5':
        #    continue
        
        s_train_ij = s_train[[columni, cloumnj]]
        s_test_ij = s_test[[columni, cloumnj]]
        clf = clf.fit(s_train_ij, Y) # 次级分类器使用adboost
        
        predict_ij = clf.predict(s_train_ij)
        scores_ij = f1_score(Y, predict_ij, average='micro')
        
        #print(pearsonr(s_train[columni], s_train[cloumnj])[0])
        print('stacking train result {}&{}: {}'.format(columni, cloumnj, scores_ij))

s_train = s_train[['m1','m4','m5','m6']]
s_test = s_train[['m1','m4','m5','m6']]


clf = AdaBoostClassifier().fit(s_train, Y) # 次级分类器使用adboost 效果最好
#clf = Perceptron().fit(s_train, Y)
#clf = LogisticRegression().fit(s_train, Y)

# 训练样本上的F1值
predict = clf.predict(s_train)
scores = f1_score(Y, predict, average='micro')
print 'train f1: {}'.format(scores)

# 训练样本上的cross F1值
scores = cross_val_score(clf, s_train, Y, cv=5) # 5折交叉验证
print 'train cross f1: {}, avg: {}'.format(scores, np.mean(scores))

# 输出
predict_ret = clf.predict(s_test)

with open('data/test_v8_result.csv', 'w') as f:
    for i, flag in enumerate(predict_ret):
        print >> f, '{},{}'.format(i + 1, flag)

