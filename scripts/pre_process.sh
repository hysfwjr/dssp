#/bin/bash

# 采用结巴分词, 修改jieba.py 中in_file, out_file
#python src/jieba.py /Users/wenjurong/github/dssp/data/data_train.csv /Users/wenjurong/github/dssp/data/data_train.seg
#python src/jieba.py /Users/wenjurong/github/dssp/data/data_test.csv /Users/wenjurong/github/dssp/data/data_test.seg

# 过滤空串、1个单词(后面去掉，因为发现艹单词影响）、停用词、高频词、低频词(次数少于5次) - 词袋模型
# 停用词: data/stopwords.dat from https://github.com/dongxiexidian/Chinese/blob/master/stopwords.dat
cat data/data_train.seg data/data_test.seg | python src/vocab.py 1> data/vocab.dict 2> log/vocab.log

# 根据vocab 生成样本
cat data/data_train.seg | python src/gen_sample.py > data/data_train.sample
cat data/data_test.seg | python src/gen_sample.py > data/data_test.sample
