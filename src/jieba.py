# -*- coding: utf-8 -*-
# @author hysfwjr()
# date 2018-01-08

import re
import json
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import jieba
#jieba.enable_parallel(4)
#jieba.set_dictionary('../conf/dict.txt.big')
#jieba.initialize()
#in_file = '/Users/wenjurong/github/dssp/data/data_train.csv'
in_file = sys.argv[1]
#out_file = '/Users/wenjurong/github/dssp/data/data_train.seg'
out_file = sys.argv[2]

replace_pattern = [
        ur'@\S\+ ',
        ur'回复@\S\+:',
        ]

with open(in_file, 'r') as in_f:
    with open(out_file, 'w') as out_f:
        for line in in_f:
            line = line.strip().decode('utf-8')
            # 正则过滤, 去除@、回复
            #print >> sys.stderr, line
            for pattern in replace_pattern:
                line = re.subn(ur'回复@\S+:', ' ', line)[0]
                line = re.subn(ur'@\S+', ' ', line)[0]
            #print >> sys.stderr, line
            fields = line.strip().split('\t')
            s_id = fields[0]
            s_text = fields[1]
            s_flag = fields[2] if len(fields) == 3 else ''
            seg_list = jieba.cut(s_text)  # 默认是精确模式
            fields[1] = '/'.join(seg_list)
            print >> out_f, '\t'.join(fields)
