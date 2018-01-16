# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

# 使用训练好的word2vec 模型找出本例中较为相近的『word』
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

model = KeyedVectors.load_word2vec_format('data/data_feed_in_word2vec.txt.50.bin', binary=True)
# 测试
#print model.similarity(u'快乐', u'快乐')

# vocab path
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

vocab_vector = gen_vocab(vocab_path)
threshold = 0.7
words = vocab_vector.keys()
words_len = len(words)
words_sim = []
for i in xrange(words_len):
    left = words[i].decode('utf-8')
    if left not in model:
        print >> sys.stderr, '{} not in word2ve mode'.format(left)
        continue
    for j in xrange(i + 1, words_len):
        try:
            right = words[j].decode('utf-8')
            if right not in model:
                print >> sys.stderr, '{} not in word2ve mode'.format(right)
	    simi = model.similarity(left, right)
            if simi > threshold:
		print >> sys.stdout, '{}\t{}\t{}'.format(left, right, simi)
            #if model.similarity(left, right) >= threshold:
            #    words_sim.append((left, right))
            #    print('{} == {}'.format(left, right))
        except KeyError as e:
            pass
#print('total sim: {}'.format(len(words_sim)))
#for pair in words_sim:
#    print('{} == {}'.format(pair[0], pair[1]))
