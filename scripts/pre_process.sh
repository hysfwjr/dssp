#/bin/bash

# 采用结巴分词, 修改jieba.py 中in_file, out_file
echo "[`date "+%Y-%m-%d %H:%M:%S"`] seg word start ..."
python src/jieba_seg.py data/data_train.csv data/data_train.seg
#python src/jieba_seg.py data/data_test.csv data/data_test.seg
python src/jieba_seg.py data/data_test_B.csv data/data_test_B.seg
echo "[`date "+%Y-%m-%d %H:%M:%S"`] seg word finish ..."

# 过滤空串、1个单词(后面去掉，因为发现艹单词影响）、停用词、高频词、低频词(次数少于3次)、人称代词 - 词袋模型
# 停用词: data/stopwords.dat from https://github.com/dongxiexidian/Chinese/blob/master/stopwords.dat
echo "[`date "+%Y-%m-%d %H:%M:%S"`] vocab start ..."
#cat data/data_train.seg data/data_test.seg | python src/vocab.py 1> data/vocab.dict 2> log/vocab.log
cat data/data_train.seg data/data_test_B.seg | python src/vocab.py 1> data/vocab.dict 2> log/vocab.log
echo "[`date "+%Y-%m-%d %H:%M:%S"`] vocab finish ..."

# 根据vocab 生成样本
echo "[`date "+%Y-%m-%d %H:%M:%S"`] sample generate start ..."
cat data/data_train.seg | python src/gen_sample.py > data/data_train.sample
#cat data/data_test.seg | python src/gen_sample.py > data/data_test.sample
cat data/data_test_B.seg | python src/gen_sample.py > data/data_test_B.sample
echo "[`date "+%Y-%m-%d %H:%M:%S"`] sample generate finish ..."

# 根据带有主题的vocab 生成样本
#cat data/data_train.seg | python src/gen_sample_with_theme.py > data/data_train_with_theme.sample &
