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

