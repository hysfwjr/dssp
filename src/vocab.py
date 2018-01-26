# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import numpy as np
import pandas as pd
import json
import common
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
stopwords_file = '/Users/wenjurong/github/dssp/data/stopwords.dat'
vocab_black_file = '/Users/wenjurong/github/dssp/data/vocab_black.txt'

vocab_dict = {} 
vocab_dict_detail = {} # {word: [0, 1, 2, 3]} 表示每个单词在0、1、2类样本中个数，3表示测试样本个数

def gen_vocab_black(path):
    """ vocab black
    """
    vocab_black = set()
    with open(path, 'r') as f:
        for line in f:
            vocab_black |= set(line.decode('utf-8').strip().split(' '))
    return vocab_black
            

stopwords = common.gen_stopwords(stopwords_file)
vocab_black = gen_vocab_black(vocab_black_file)
#print >> sys.stderr, vocab_black

for line in sys.stdin:
    fields = line.split('\t')
    segs = fields[1].decode('utf-8').split('/')
    flag = int(fields[2]) if len(fields) == 3 else 3 # 如果flag不存在，则为测试集数据，数组为下标为3
    # 过滤空串、1个单词、停用词
    words = []
    for word in segs:
        word = word.strip().decode('utf-8')
        if word == '':
            continue
        if len(word.split('_')) != 2:
            print >> sys.stderr, '{} is illegal'.format(line)
            sys.exit(1)
        true_word = word.split('_')[0]
        if true_word not in stopwords and true_word not in vocab_black:
        #    words.append(word)
            words.append(true_word) # 不带词性，发现有些词有多种词性, 如艹_yg,艹_y,艹_zg
        else:
            #print >> sys.stderr, 'word: {} is filter'.format(word)
            pass

    #for word in words:
    for word in set(words): # 一行中1个词只计1次
        if word not in vocab_dict_detail:
            vocab_dict_detail[word] = [0] * 4
        vocab_dict_detail[word][flag] += 1
        vocab_dict[word] = vocab_dict.get(word, 0) + 1

## 计算tf-idf
#tf_idf_data = common.tf_idf(vocab_dict_detail)
#tf_idf_data.to_csv('data/tf_idf.dat')
## 基于tf-idf 过滤

# 低频词(次数少于2次) 过滤, 如果词长度为1需要满足词频> 9
#words_nofreq = [k for k, v in vocab_dict.items() if v < 2 or (len(k.split('_')[0]) == 1 and v < 10)]
words_nofreq = [k for k, v in vocab_dict.items() if v < 2]
for word in words_nofreq:
    vocab_dict.pop(word)

# 按照发现次数从高到低排序
sorted_vocab = sorted(vocab_dict.iteritems(), lambda x, y: cmp(x[1], y[1]), reverse=True)
#for i in range(500):
#    print >> sys.stderr, 'words_top_freq: {}: {}'.format(sorted_vocab[i][0], sorted_vocab[i][1])
for x in sorted_vocab:
    print >> sys.stdout, '{}:{}'.format(x[0], x[1])
