# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
vocab_path = 'data/vocab.dict'

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


def gen_titile(vocab):
    """ 根据vocab的value次序输出vocab word
    Parameters
    ----------
    vocab: dict

    Returns
    -------
    title: str
    """
    v_s = sorted(vocab.iteritems(), lambda x, y: cmp(x[1], y[1]))
    #print >> sys.stderr, v_s
    wds = [x[0] for x in v_s]
    #print >> sys.stderr, ','.join(wds)
    return ','.join(wds)


vocab_vector = gen_vocab(vocab_path)
vocab_vector_len = len(vocab_vector.keys())
title_print = False
for line in sys.stdin:
    fields = line.strip().split('\t')
    s_id = fields[0]
    s_seg = fields[1].split('/')
    s_flag = fields[2] if len(fields) == 3 else ''

    vec = [0] * vocab_vector_len
    for word in s_seg:
        word = word.decode('utf-8')
        if word in vocab_vector:
            vec[vocab_vector[word]] += 1
        else:
            #print >> sys.stderr, 'word: {} is not in vector'.format(word)
            pass
    vec = map(str, vec)
    if s_flag == '': # 测试样本
        if not title_print:
            title_print = True
            print >> sys.stdout, 'id,{}'.format(gen_titile(vocab_vector))
        out_val = '{},{}'.format(s_id, ','.join(vec))
    else:
        if not title_print:
            title_print = True
            print >> sys.stdout, 'id,{},target'.format(gen_titile(vocab_vector))
        out_val = '{},{},{}'.format(s_id, ','.join(vec), s_flag)
    print >> sys.stdout, out_val
