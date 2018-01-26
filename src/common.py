# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import numpy as np
import pandas as pd
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
vocab_path = 'data/vocab.dict'
stopwords_file = '/Users/wenjurong/github/dssp/data/stopwords.dat'

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


def tf_idf(vocab_dict):
    """ tf_idf

    Parameters
    ----------
    vocab_dict: dict, {word: [0, 1, 2, 3]} 表示每个单词在0、1、2类样本中个数，3表示测试样本个数

    Returns
    -------
    tf_idf: pd.Series
    """
    vocab_datas = pd.DataFrame(vocab_dict).T
    vocab_datas.to_csv('data/vocab_dict_aux.dat')
    total_words = vocab_datas.sum(axis=0)
    print >> sys.stderr, total_words
    tf_data = vocab_datas / (total_words + 1) # 避免除0, 将total_words + 1
    def f(x):
	f_0 = 1 if x.iloc[0] > 0 else 0
	f_1 = 1 if x.iloc[1] > 0 else 0
	f_2 = 1 if x.iloc[2] > 0 else 0
	f_3 = 1 if x.iloc[3] > 0 else 0
	return np.log2(3.0 + 1 / (f_0 + f_1 + f_2 + 1))
    idf_data = vocab_datas.apply(f, axis = 1)
    tf_data['idf'] = idf_data
    tf_idf = tf_data.apply(lambda x: '{},{},{},{}'.format(
            x.iloc[0] * x['idf'], x.iloc[1] * x['idf'],
            x.iloc[2] * x['idf'], x.iloc[3] * x['idf']),
            axis = 1)
    return tf_idf

def gen_vocab(path):
    """ vocab
    Parameters
    ----------
    path: str, file path

    Returns
    -------
    vocab_vector: dict
    """
    vocab_vector = {}
    lino = 0
    with open(path, 'r') as f:
        for line in f:
            fields = line.split(':')
            if len(fields) != 2:
                print >> sys.stderr, 'line: {} is illegal'.format(line)
            word = fields[0].decode('utf-8')
            freq = int(fields[1])
            vocab_vector[word] = lino
            lino = lino + 1
    return vocab_vector


def gen_vocab_black(path):
    """ vocab black
    """
    vocab_black = set()
    with open(path, 'r') as f:
        for line in f:
            vocab_black |= set(line.decode('utf-8').strip().split(' '))
    return vocab_black


def gen_vocab_with_theme(path):
    """ vocab
    Parameters
    ----------
    path: str, file path, content: word:freq:theme_id

    Returns
    -------
    vocab_vector: dict
    """
    vocab_vector = {}
    with open(path, 'r') as f:
        for line in f:
            fields = line.split(':')
            if len(fields) != 3:
                print >> sys.stderr, 'line: {} is illegal'.format(line)
            word = fields[0].decode('utf-8')
            freq = int(fields[1])
            theme_id = int(fields[2])
            vocab_vector[word] = theme_id
    return vocab_vector


def clique_k_aux(mem_dict):
    """
    采用DFS遍历，每完成1次遍历即为一个社区
    Parameters
    ----------
    mem_dict: dict, {mema: [memb, mebc]}

    Returns
    -------
    ret: dict, 每个成员的社区信息
    """
    all_mem_set = set(reduce(lambda x, y: x if y in x else x + y, mem_dict.values(), [])) | set(mem_dict.keys())
    ret = {mem: -1 for mem in all_mem_set}
    clique = 0
    for mem in ret.keys():
        if ret[mem] != -1:
            continue
        ret[mem] = clique
        dfs(mem, mem_dict, ret, clique)
        clique = clique + 1
    return ret


def dfs(node, mem_dict, clique_dict, clique):
    """ dfs
    """
    if node not in mem_dict:
        return
    for adj in mem_dict[node]:
        if clique_dict[adj] != -1: # 防止死循环
            continue
        clique_dict[adj] = clique
        dfs(adj, mem_dict, clique_dict, clique)

