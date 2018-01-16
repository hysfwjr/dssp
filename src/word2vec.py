# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import common
import gensim

class MySentences(object):
    """ 文件的输入格式为id\tsenstence[\tflag]
    """
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r') as f:
            for line in f:
                fields = line.strip('\n').split('\t')
                if len(fields) < 2:
                    raise ValueError(
                            'input: {} is illegal, should be id\tsenstence[\tflag]'.format(line))
                sentences = fields[1]
                yield sentences.split('/')

# size是输出词向量的维数, min_count是对词进行过滤，频率小于min-count的单词则会被忽视
model = gensim.models.Word2Vec(MySentences('data/data_train.seg'),
        size=100, min_count=5, workers=4)
model.train(MySentences('data/data_test.seg'), total_examples=5000, epochs=1) # 进一步训练
model.wv.save_word2vec_format('data/data_word2vec.bin')
