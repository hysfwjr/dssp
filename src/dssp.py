# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08
""" Usage:
  dssp predict <conf_in> <predict_out>
  dssp model_eval <conf_in>

Options:
  -h --help     Show this screen.
"""

# Load libraries
from docopt import docopt
import ConfigParser
import time
import sys
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from mlxtend.classifier import StackingClassifier
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
from sklearn.multiclass import OneVsRestClassifier
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

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier():
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.4)
    #model.fit(train_x, train_y)
    return model

class XgbClassifier(xgb.XGBClassifier):
    def __init__(self):
        xgb_param = {}
        # use softmax multi-class classification
        xgb_param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        xgb_param['eta'] = 0.4
        xgb_param['max_depth'] = 1
        xgb_param['silent'] = 1
        xgb_param['nthread'] = 4
        xgb_param['num_class'] = 3
        xgb_param['booster'] = 'gblinear' # gbtree->gblinear, 训练从0.67 -> 0.83
        xgb_param['missing'] = 0
        num_round = 10
        #clf = xgb.XGBClassifier(n_estimators = 100,nthread=10,max_depth=5,objective= 'multi:softmax')
        xgb.XGBClassifier.__init__(self, **xgb_param)
        #print clf.attributes()
        #clf.fit(train_x, train_y)

class  Classifer():
    def __init__(self, conf_in):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(conf_in)

        self.train_path = str(self.cf.get('basic', 'train_path'))
        self.test_path = str(self.cf.get('basic', 'test_path'))
        self.fold_k = self.cf.getint('feature_select', 'fold_k')
        self.is_chi2 = self.cf.getint('feature_select', 'is_chi2')
        self.chi2_best_k = self.cf.getint('feature_select', 'chi2_best_k')

        # xgboost params
        xgb_param = {}
        # use softmax multi-class classification
        xgb_param['objective'] = 'multi:softmax'
        # scale weight of positive examples
        xgb_param['eta'] = 0.4
        xgb_param['max_depth'] = 1
        xgb_param['silent'] = 1
        xgb_param['nthread'] = 10 
        xgb_param['num_class'] = 3
        xgb_param['booster'] = 'gblinear' # gbtree->gblinear, 训练从0.67 -> 0.83
        #num_round = 10

        self.classifiers = {
                'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
                'NB': MultinomialNB(alpha=0.4),
                'Perceptron': Perceptron(alpha=0.1),
                'XGBOOST': xgb.XGBClassifier(**xgb_param),
        }
        self.test_classifiers = ['NB', 'Perceptron', 'RF', 'XGBOOST']
        self.ensemble_classifiers = ['NB', 'RF', 'XGBOOST']

    def stacking_base_train(self, train_X, train_y, test_X):
        """ stacking 基分类器分类

        Parameters
        ----------
        train_X: array-like, shape = (n_samples, n_features)
        train_y: array-like, shape = (n_samples)
        test_X: array-like, shape = (n_tests, n_features)

        Returns
        -------
        s_train_X: pd.DataFrame, shape = (n_samples, n_base_models)
        s_test_X: pd.DataFrame, shape = (n_tests, n_base_models)
        """
        s_trains = {}
        s_tests = {}
        for classifier in self.ensemble_classifiers:
            print >> sys.stderr, 'stacking base model: {} start'.format(classifier)
            a_train, a_test = self.get_oof(self.classifiers[classifier],
                    train_X, train_y, test_X, self.fold_k)
            a_scores = self.score(a_train, train_y)
            print >> sys.stderr, 'stacking base model: {} scores: {}, mean_score: {}'.format(
                    classifier, a_scores, np.mean(a_scores))
            print >> sys.stderr, 'stacking base model: {} finished'.format(classifier)
            s_trains[classifier] = a_train
            s_tests[classifier] = a_test
	
        # 次级分类器分类
        s_train_X = pd.DataFrame(s_trains)
        s_test_X = pd.DataFrame(s_tests)
        return s_train_X, s_test_X

    def score(self, y_true, y_pred):
        """ f1_score

        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
        y_pred : 1d array-like, or label indicator array / sparse matrix

        Returns
        -------
        f1_score : float or array of float, shape = [n_unique_labels]
        """
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average='macro') 

    def eval_ensemble_model(self, train_X, train_y, test_X):
        """ evaluate ensemble model
        """
        s_train_X, s_test_X = self.stacking_base_train(train_X, train_y, test_X)
	sec_clf = AdaBoostClassifier() # 次级分类器使用adboost 效果最好
	# 两两查看
	for i in range(len(s_train_X.columns)):
	    columni = s_train_X.columns[i]
	    s_train_i = s_train_X[[columni]]
	    s_test_i = s_test_X[[columni]]
            
	    clf = sec_clf.fit(s_train_i, train_y)
	    predict_i = clf.predict(s_train_i)
	    scores_i = f1_score(train_y, predict_i, average='micro')
	    print('stacking train result {}: {}'.format(columni, scores_i))
	    
	    for j in range(i + 1, len(s_train_X.columns)):        
		cloumnj = s_train_X.columns[j]
		s_train_ij = s_train_X[[columni, cloumnj]]
		s_test_ij = s_test_X[[columni, cloumnj]]
		clf = sec_clf.fit(s_train_ij, train_y) 
		
		predict_ij = clf.predict(s_train_ij)
		scores_ij = f1_score(train_y, predict_ij, average='micro')
		#print(pearsonr(s_train_X[columni], s_train_X[cloumnj])[0])
		print('stacking train result {}&{}: {}'.format(columni, cloumnj, scores_ij))

	# 全部模型
        sec_clf = sec_clf.fit(s_train_X, train_y)
        s_train_predict = sec_clf.predict(s_train_X)
        s_train_scores = f1_score(train_y, s_train_predict, average='micro')
        print('stacking train result {}: {}'.format('&'.join(self.ensemble_classifiers),
                s_train_scores))
	

    def fit_and_predict(self, clf_name, train_X, train_y, test_X):
        """ 使用单个模型训练评估，并返回测试的结果
        """
        if clf_name not in self.classifiers:
            print >> sys.stderr, '{} not in model dict'.format(clf_name)
        print >> sys.stdout, '******************* %s ********************' % clf_name
        clf = self.classifiers[clf_name]

        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import f1_score
        cross_scores = cross_val_score(clf, train_X, train_y, cv=self.fold_k)
        print >> sys.stdout, 'model:{} {}-fold cross cross_scores: {}, avg: {}'.format(
                clf_name, self.fold_k, cross_scores, np.mean(cross_scores))

        t_start = time.time()
        clf.fit(train_X, train_y)
        t_end = time.time()

        train_predict = clf.predict(train_X)
        test_predict = clf.predict(test_X)
        train_score = f1_score(train_y, train_predict, average='micro')
        print >> sys.stdout, 'model:{} train cost: {}s, predict f1_score: {}'.format(
                clf_name, t_end - t_start, train_score)
        print >> sys.stdout, '*******************************************'
        return test_predict

    def eval_single_model(self, train_X, train_y, test_X):
        """ evaluate single_model
        """
        for classifier in self.test_classifiers:
            if classifier not in self.classifiers:
                print >> sys.stderr, '{} is not classifiers'.format(classifier)
                return -1
            self.fit_and_predict(classifier, classifier, train_X, train_y, test_X)

    def load_data(self, train_path, test_path):
        """ load_data

        Parameters
        ----------
        train_path: str
        test_path: str

        Returns
        -------
        train_X: array-like, shape = (n_samples, n_features)
        train_y: array-like, shape = (n_samples)
        test_X: array-like, shape = (n_tests, n_features)
        """
        # Load dataset
        #dataset = pd.read_csv('/Users/wenjurong/github/dssp/data/data_train.sample')
        #testset = pd.read_csv('/Users/wenjurong/github/dssp/data/data_test_B.sample', quoting=3, error_bad_lines=False)
        dataset = pd.read_csv(self.train_path, quoting=3, error_bad_lines=False)
        testset = pd.read_csv(self.test_path, quoting=3, error_bad_lines=False)
        #train_X = dataset.as_matrix()[:, 1: -1] # 排除首列id/末列target
        train_X = dataset.ix[:, 1: -1] # 排除首列id/末列target
        train_y = dataset.target
        test_X = testset.ix[:, 1: ] # 排除首列id
        num_train, num_feat = train_X.shape
        num_test, num_feat_test = test_X.shape
        if num_feat != num_feat_test:
            print >> sys.stderr, ('loads data failed, num feats of train and test '
                    'is not equal: {} != {}').format(num_feat, num_feat_test)
        print >> sys.stderr, ('load train & test set finished, train\'shape: {},'
                'test\'shape: {}').format(train_X.shape, test_X.shape)
        return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X}

    def train_and_predict(self, test_pred_path, model_out_path):
        """ train
        """
        # loads data
        datas = self.load_data(self.train_path, self.test_path)
        train_X = datas.get('train_X', None)
        train_y = datas.get('train_y', None)
        test_X = datas.get('test_X', None)
        if train_X is None or train_y is None or test_X is None:
            print >> sys.stderr, 'loads data failed'
            return -1
	
	# noise erase
        train_y_ori = train_y.copy()
        ignore_features = ['COMMENTWEIBO', 'ATWEIBO', 'RETWEETWEIBO']
        train_y = Classifer.erase_noise(train_X, train_y, ignore_features, new_val=1)

        # feature norm
        train_X = Classifer.norm_feature(train_X)
        test_X = Classifer.norm_feature(test_X)
        print >> sys.stderr, 'feature norm finish'

        # feature select
        if self.is_chi2 == 1:
            train_X, test_X = self.feature_selection(train_X, train_y, test_X, self.chi2_best_k)
        print >> sys.stderr, ('feature select finish, new train\'shape: {}, new '
                'test\'shape: {}').format(train_X.shape, test_X.shape)

        # result 1: 单独xgboost
        pre_xgboost = self.fit_and_predict('XGBOOST', train_X, train_y, test_X)
	self.predict_emit('xgb_{}'.format(test_pred_path), pre_xgboost)

        # result 2: 自己实现的stacking
        s_train_X, s_test_X = self.stacking_base_train(train_X, train_y, test_X)
	sec_clf = AdaBoostClassifier().fit(s_train_X, train_y) # 次级分类器使用adboost 效果最好

        cross_scores = cross_val_score(sec_clf, s_train_X, train_y, cv=self.fold_k)
        print('my stacking model {}: cross_scores: {}, avg: {}'.format(
                '&'.join(self.ensemble_classifiers),
                cross_scores,
                np.mean(cross_scores)))
		
        s_train_predict = sec_clf.predict(s_train_X)
        s_train_scores = f1_score(train_y, s_train_predict, average='micro')
        print('my stacking train result {}: {}'.format('&'.join(self.ensemble_classifiers),
                s_train_scores))

        self.save_model(sec_clf, 'mystack_{}_{}'.format('&'.join(self.ensemble_classifiers),
                model_out_path))
        pre_mystack = sec_clf.predict(test_X)
        self.predict_emit('mystack_{}_{}'.format('&'.join(self.ensemble_classifiers),
            test_pred_path), pre_mystack)

        # result 3: 使用mlxtend 库的Stacking
        sclf = StackingClassifier(
                classifiers=[self.classifiers[i] for i in self.ensemble_classifiers], 
                use_probas=True, # 
                average_probas=False,
                meta_classifier=AdaBoostClassifier()) 
        sclf = sclf.fit(train_X, train_y)

        cross_scores = cross_val_score(sclf, train_X, train_y, cv=self.fold_k)
        print('stacking model {}: cross_scores: {}, avg: {}'.format(
                '&'.join(self.ensemble_classifiers),
                cross_scores,
                np.mean(cross_scores)))

        s_train_predic2 = sclf.predict(train_X)
        s_train_scores2 = f1_score(train_y, s_train_predic2, average='micro')
        print('stacking model {}: train predict result: {}'.format(
                '&'.join(self.ensemble_classifiers),
                s_train_scores2))

        self.save_model(sclf, 'stack_{}_{}'.format('&'.join(self.ensemble_classifiers),
                model_out_path))
        pre_stack = sclf.predict(test_X)
	self.predict_emit('stack_{}_{}'.format('&'.join(self.ensemble_classifiers),
                test_pred_path), pre_stack)

    @staticmethod
    def save_model(clf, out_path):
        """ save_model

        Parameters
        ----------
        clf: model
        out_path: str

        Returns
        -------
        None
        """
        from sklearn.externals import joblib
        joblib.dump(clf, out_path)

    @staticmethod
    def load_model(in_path):
        """ load model

        Parameters
        ----------
        in_path: str

        Returns
        -------
        clf: model
        """
        from sklearn.externals import joblib
        return joblib.load(in_path)

    def predict_emit(self, path, predict_result):
        """ predict emit
        """
        with open(path, 'w') as f:
	    for i, flag in enumerate(predict_result):
		print >> f, '{},{}'.format(i + 1, flag)

    def feature_selection(self, train_X, train_y, test_X, topk=10000):
        """ chi2 特征选择

        Parameters
        ----------
        train_X: ndarray
        train_y: 1 array
        test_X: ndarray
        topk: int

        Returns
        -------
        best_train_X: ndarray
        best_test_X: ndarray
        """
        feat_sel = SelectKBest(chi2, k=topk)
        best_train_X = feat_sel.fit_transform(train_X, train_y)
        best_test_X = feat_sel.transform(test_X)
        return best_train_X, best_test_X

    @staticmethod
    def norm_feature(data):
        """ 将data 数据中>1的数值全部归一化1

        Parameters
        ----------
        data: ndarray

        Returns
        -------
        norm_data: ndarray
        """
        pd_data = pd.DataFrame(data)
        pd_data[pd_data > 1] = 1
        return pd_data.as_matrix()

    def get_oof(self, clf, train_X, train_y, test_X, fold_k=5):
        """ kfold stacking model assemb

        Parameters
        ----------
        clf: model
        train_X: ndarray
        train_y: 1 array
        test_X: ndarray
        fold_k: int

        Returns
        -------
        oof_train: ndarray
        oof_test: ndarray
        """
        ntrain = train_X.shape[0]
        ntest = test_X.shape[0]

	oof_train = np.zeros((ntrain,))
	oof_test = np.zeros((ntest,))
	oof_test_skf = np.empty((fold_k, ntest))

        kf = KFold(n_splits=fold_k, shuffle=False, random_state=2017)
	
	for i, (train_index, test_index) in enumerate(kf.split(train_X)):
	    kf_train_X = train_X[train_index]
	    kf_train_y = train_y[train_index]
	    kf_test_X = train_X[test_index]
	    
	    clf.fit(kf_train_X, kf_train_y)
	    
	    oof_train[test_index] = clf.predict(kf_test_X)
	    oof_test_skf[i, :] = clf.predict(test_X)
	oof_test[:] = oof_test_skf.mean(axis=0)
	return oof_train, oof_test

    def model_assember():
        """
        """
        pass

    @staticmethod
    def erase_noise(train_X, train_y, ignore_features, new_val=1):
        """ 将train_X 中一行元素特征全为0对应的行的targe置为1（中性）
        Parameters
        ----------
        train_X: array-like, shape = (n_samples, n_features)
        train_y: array-like, shape = (n_samples)
        ignore_features: array, ['COMMENTWEIBO_eng', 'ATWEIBO_eng', 'RETWEETWEIBO_eng']

        Returns
        -------
        n_train_y: array-like, shape = (n_samples)
        """
        # 排除weibo评论、at、转发
        train_X_temp = train_X.drop(ignore_features, axis=1)
        n_train, n_features = train_X_temp.shape
        n_train_y = train_y.copy()
        
        #print train_X_temp
        update_y = []
        for i in range(n_train):
            x = train_X_temp.loc[i]
            # 特征全为0，targe修正
            if np.sum(x) == 0 and \
                    train_y[i] != new_val:
                #print >> sys.stderr, 'sample: {}\'s features is all 0'.format(i)
                update_y.append(str(i))
                n_train_y[i] = new_val
        print >> sys.stderr, 'total: {} features is all 0, detail: {}'.format(
                (n_train_y != train_y).tolist().count(True), update_y)
        return n_train_y

def model_eval(conf_path):
    classifier = Classifer(conf_path)
    # loads data
    datas = classifier.load_data(classifier.train_path, classifier.test_path)
    train_X = datas.get('train_X', None)
    train_y = datas.get('train_y', None)
    test_X = datas.get('test_X', None)
    if train_X is None or train_y is None or test_X is None:
        print >> sys.stderr, 'loads data failed'
        sys.exit(1)

    # noise erase
    train_y_ori = train_y.copy()
    ignore_features = ['COMMENTWEIBO', 'ATWEIBO', 'RETWEETWEIBO']
    train_y = Classifer.erase_noise(train_X, train_y, ignore_features, new_val=1)

    # feature norm
    train_X = Classifer.norm_feature(train_X)
    test_X = Classifer.norm_feature(test_X)
    print >> sys.stderr, 'feature norm finish'

    # feature select
    train_X, test_X = classifier.feature_selection(train_X, train_y, test_X, classifier.chi2_best_k)
    print >> sys.stderr, ('feature select finish, new train\'shape: {}, new '
            'test\'shape: {}').format(train_X.shape, test_X.shape)

    # 单分类器评估
    classifier.eval_single_model(train_X, train_y, test_X)

    # 多分类器评估
    classifier.eval_ensemble_model(train_X, train_y, test_X)

if __name__ == '__main__':
    args = docopt(__doc__, version='dssp')
    #print args
    conf_path = args['<conf_in>']
    if args['model_eval']:
        model_eval(conf_path)
    if args['predict']:
        classifier = Classifer(conf_path)
        test_pred_path = args['<predict_out>']
        # train and pred
        classifier.train_and_predict(test_pred_path, "assember_model.pkl")
    sys.exit(0)
