# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import common

def gen_titile(vocab):
    """ 根据vocab的value次序输出vocab word
    Parameters
    ----------
    vocab: dict

    Returns
    -------
    title: str
    """
    max_themeid = max(vocab.values())
    #v_s = sorted(vocab.iteritems(), lambda x, y: cmp(x[1], y[1]))
    #print >> sys.stderr, v_s
    wds = ['theme_{}'.format(x) for x in range(max_themeid + 1)]
    #print >> sys.stderr, ','.join(wds)
    return ','.join(wds)


vocab_path = 'data/vocab_2_theme.dict'
vocab_vector = common.gen_vocab_with_theme(vocab_path)
vocab_vector_len = len(set(vocab_vector.values()))
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
