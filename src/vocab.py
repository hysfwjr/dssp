# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
stopwords_file = '/Users/wenjurong/github/dssp/data/stopwords.dat'
vocab_black_file = '/Users/wenjurong/github/dssp/data/vocab_black.txt'

vocab_dict = {}

def gen_stopwords(path):
    """ stopwords
    Parameters
    ----------
    path: str, file path

    Returns
    -------
    stopwords: set
    """
    stopwords = set()
    with open(path, 'r') as f:
        for line in f:
            stopwords |= set(line.decode('utf-8').strip().split(' '))
    return stopwords

def gen_vocab_black(path):
    """ vocab black
    """
    vocab_black = set()
    with open(path, 'r') as f:
        for line in f:
            vocab_black |= set(line.decode('utf-8').strip().split(' '))
    return vocab_black
            

stopwords = gen_stopwords(stopwords_file)
vocab_black = gen_vocab_black(vocab_black_file)
#print >> sys.stderr, vocab_black

for line in sys.stdin:
    segs = line.split('\t')[1].decode('utf-8').split('/')
    # 过滤空串、1个单词、停用词
    words = []
    for word in segs:
        word = word.strip().decode('utf-8')
        if word not in stopwords and word not in vocab_black:
            words.append(word)
        else:
            print >> sys.stderr, 'word: {} is filter'.format(word)

    for word in words:
        vocab_dict[word] = vocab_dict.get(word, 0) + 1

# 低频词(次数少于4次) 过滤, 如果词长度为1需要满足词频> 9
words_nofreq = [k for k, v in vocab_dict.items() if v < 3 or (len(k) == 1 and v < 10)]
for word in words_nofreq:
    vocab_dict.pop(word)

# 按照发现次数从高到低排序
sorted_vocab = sorted(vocab_dict.iteritems(), lambda x, y: cmp(x[1], y[1]), reverse=True)
#for i in range(500):
#    print >> sys.stderr, 'words_top_freq: {}: {}'.format(sorted_vocab[i][0], sorted_vocab[i][1])
for x in sorted_vocab:
    print >> sys.stdout, '{}:{}'.format(x[0], x[1])
