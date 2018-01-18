# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08
import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import jieba
import jieba.posseg
#jieba.enable_parallel(4)
#jieba.set_dictionary('../conf/dict.txt.big')
#jieba.initialize()
#in_file = '/Users/wenjurong/github/dssp/data/data_train.csv'
#out_file = '/Users/wenjurong/github/dssp/data/data_train.seg'
in_file = '/Users/wenjurong/github/dssp/data/data_test_B.csv'
out_file = '/Users/wenjurong/github/dssp/data/data_test_B.seg'
vocab_black_file = '/Users/wenjurong/github/dssp/data/vocab_black.txt'
#in_file = 'ss'
#out_file = 'tt'

replace_pattern = [
        ur'@\S\+ ',
        ur'回复@\S\+:',
        ]
def gen_vocab_black(path):
    """ vocab black
    """
    vocab_black = set()
    with open(path, 'r') as f:
        for line in f:
            vocab_black |= set(line.decode('utf-8').strip().split(' '))
    return vocab_black

vocab_black = gen_vocab_black(vocab_black_file)
neg_words = [u'不']

with open(in_file, 'r') as in_f:    
    with open(out_file, 'w') as out_f:
        for line in in_f:
            line = line.strip().decode('utf-8')
            # 正则过滤, 去除@、回复
            #for pattern in replace_pattern:
            line = re.subn(ur'回复@\S+:', ' ', line)[0]
            #line = re.subn(ur'@\S+', ' ', line)[0]
            #print >> sys.stderr, line
            fields = line.strip().split('\t')
            s_id = fields[0]
            s_text = fields[1]
            s_flag = fields[2] if len(fields) == 3 else ''
            seg_list = jieba.posseg.cut(s_text)  # 默认是精确模式
            seg_list = list(seg_list)
            # 否定词处理，将否定词到紧接后面的"," 前的词都加上否定词
            neg_word = ''
            segs = []
            for i, seg in enumerate(seg_list):
                word = seg.word
                flag = seg.flag # 词性：refer: http://blog.sina.com.cn/s/blog_628cc2b70102wb7z.html
                
                # word 为标点符号如, 空格等，将neg_word 赋值''
                #if word in [u'，', u', ',u': ', u'。', ':', '?']:
                #    neg_word = ''
                #    continue
                #print >> out_f, 'word: {} is {}'.format(word, flag)
                # word 为标点符号
                if flag in ['x']: 
                    neg_word = ''
                    continue
                # 去掉人称代词
                if flag in ['r']:
                    continue 
                # 去掉黑名单
                if word in vocab_black:
                    continue
                # 判断word 是否含neg_word, 将neg_word 赋值否定词
                contain_negs = [w for w in neg_words if word.find(w) != -1]
                if len(contain_negs) > 0:
                    neg_word = contain_negs[0] # 取第一个
                    segs.append(word)
                    continue                
                # 如果neg_word 不为''，将本word前添加否定次
                if neg_word != '' and flag in ['vg', 'v', 'vd', 'vn', 'a', 'ad']: # 动、形容词
                    segs.append('{}{}'.format(neg_word, word))
                elif word != '':
                    segs.append(word)
                else:
                    pass

            fields[1] = '/'.join(segs)
            print >> out_f, '\t'.join(fields)
