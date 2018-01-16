# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

# 使用训练好的word2vec 模型找出本例中较为相近的『word』
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import json
import sys
import common
reload(sys)
sys.setdefaultencoding("utf-8")

def gen_simword_dict(vocab_path, simi_threshold=0.9):
    """ 从vocab 生成simword

    Parameters
    ----------
    vocab_path: str, path
    simi_threshold: float, choose min simi score 

    Returns
    -------
    simword_dict: dict, {word: [simili word]}
    """
    simword_dict = {}
    with open(vocab_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) != 3:
                print >> sys.stderr, '{} is illegal'.format(line)
            left = fields[0].decode('utf-8')
            right = fields[1].decode('utf-8')
            sim = float(fields[2])
            if sim >= simi_threshold:
                if left not in simword_dict:
                    simword_dict[left] = []
                if right not in simword_dict:
                    simword_dict[right] = []
                simword_dict[left].append(right)
                simword_dict[right].append(left)
    return simword_dict

simword_dict = gen_simword_dict('data/vocab_simi.dat', 0.9)
sys.setrecursionlimit(1000002) # 解决递归深度问题 
theme_dict = common.clique_k_aux(simword_dict)
vocab_path = 'data/vocab.dict'
vocab_vector = common.gen_vocab(vocab_path)
max_theme_id = max(theme_dict.values())
for word in vocab_vector.keys():
    if word in theme_dict:
        theme_id = theme_dict[word]
    else:
        max_theme_id += 1
        theme_id = max_theme_id

    print >> sys.stdout, '{}:{}:{}'.format(word, vocab_vector[word], theme_id)

#with open('ss', 'w') as f:
#    for clique in set(theme_dict.values()):
#        members_clique = [k for k, v in theme_dict.items() if v == clique]
#        print >> f, 'clique: {} => mem: {}'.format(clique, ','.join(members_clique))
            
