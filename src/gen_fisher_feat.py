# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
fisher_path = 'data/data_train_processed_fisher'

def gen_fisherdict(path):
    """ fisherdict
    Parameters
    ----------
    path: str, file path

    Returns
    -------
    fisherdict: dict
    """
    fisherdict = {}
    with open(path, 'r') as f:
        for line in f:
            fields = line.decode('utf-8').split('\t')
            if len(fields) != 4:
                print >> sys.stderr, 'line: {} is illegal'.format(line)
            word = fields[0]
            vec = map(float, fields[1: ])
            fisherdict[word] = vec
    return fisherdict

fisherdict = gen_fisherdict(fisher_path)
#print >> sys.stderr, fisherdict

title_vector = ['neg', 'neu', 'pos']
title_print = False
for line in sys.stdin:
    fields = line.strip('\n').split('\t')
    if len(fields) != 2 and len(fields) != 3:
        print >> sys.stderr, '{} is illegal'.format(line)
        continue
    s_id = fields[0]
    s_seg = fields[1].split('/')
    s_flag = fields[2] if len(fields) == 3 else ''

    vec = np.array([0.0] * 3)
    for word in s_seg:
        word = word.decode('utf-8')
        if word in fisherdict:
            vec = vec + fisherdict[word]

    vec = map(str, vec)
    if s_flag == '': # 测试样本
        if not title_print:
            title_print = True
            print >> sys.stdout, ','.join(title_vector)
        out_val = ','.join(vec)
    else:
        if not title_print:
            title_print = True
            print >> sys.stdout, '{},target'.format(','.join(title_vector))
        out_val = '{},{}'.format(','.join(vec), s_flag)
    print >> sys.stdout, out_val
