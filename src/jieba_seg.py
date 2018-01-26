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
jieba.load_userdict("data/jiba_seg.dict")

import common

neg_words = [u'不']
vocab_black_file = 'data/vocab_black.txt'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: python ' + sys.argv[0] + ' path_in' + ' path_out'
        sys.exit(1)
    in_file = sys.argv[1]
    out_file = sys.argv[2]
    vocab_black = common.gen_vocab_black(vocab_black_file)

    with open(in_file, 'r') as in_f:    
	with open(out_file, 'w') as out_f:
	    for line in in_f:
		line = line.strip().decode('utf-8')
		# 正则过滤, 去除@、回复
		#self.reply_pattern = re.compile(ur"回复@[^:]*:")
		#self.retweet_pattern = re.compile(ur"//@[^:]*:")
		#self.at_pattern = re.compile(ur"@[^\s]*[\s]|@[^\s]*$")
		# 考虑微博中回复有很大比例是『负向』的，替换为有意义的词『REPLACEWEIBO』
		# 转发微博中很大比例是「正向」的，替换为有意义的词「RETWEETWEIBO」
		line = re.subn(ur'回复@\S+:', ' COMMENTWEIBO ', line)[0] 
                # 非贪婪匹配
		line = re.subn(ur"//@\S+?:", ' RETWEETWEIBO ', line)[0]
		#line = re.subn(ur"//@\S+?:\s*转发微博", ' RETWEETWEIBO ', line)[0]
		line = re.subn(ur"@\S+ ", ' ATWEIBO ', line)[0]
		fields = line.split('\t')
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
		    
		    # 去掉标点符号
		    if flag in ['x']: 
			neg_word = ''
			continue
		#    # 去掉人称代词
		#    if flag in ['r']:
		#	continue 
                    # 去掉助词
                    if flag.find('u') == 0: # u*
                        continue
                    # 去掉f 方位词, '前面、左边'
                    if flag.find('f') == 0: 
                        continue
		    # 去掉黑名单
		    if word in vocab_black:
			continue
                    o_word = '{}_{}'.format(word, flag) # 添加词性
		    # 判断word 是否含neg_word, 将neg_word 赋值否定词
		    contain_negs = [w for w in neg_words if word.find(w) != -1]
		    if len(contain_negs) > 0:
			neg_word = contain_negs[0] # 取第一个
			segs.append(o_word)
			continue                
	            
		    # 如果neg_word 不为''，将本word前添加否定次
		    if neg_word != '' and flag in ['vg', 'v', 'vd', 'vn', 'a', 'ad']: # 动、形容词
			segs.append('{}{}'.format(neg_word, o_word))
		    elif word != '':
			segs.append(o_word)
		    else:
			pass

		fields[1] = '/'.join(segs)
		print >> out_f, '\t'.join(fields)
