## 短文本情感极性分析
http://dianshi-internal.baidu.com/gemstone/competitions/detail?raceId=2

## usage
1. 预测

    	# edit config: conf/dssp.cfg
    	bash scripts/pre_process.sh  # 预处理
    	python src/dssp.py predict conf/dssp.cfg data/test_b_predict.csv # 训练and预测


2. 模型评估, 评估单个模型交叉验证结果，多个模型融合交叉验证结果
    
   		python src/dssp.py model_eval conf/dssp.cfg > eval_model.txt

## 说明

### 预处理

    sh scripts/pre_process.sh

生成4429个词向量，每个样本包含4429个特征

### 外部工具
1. word2vec 训练
2. stacking: refer: https://zhuanlan.zhihu.com/p/26890738

### 首先跑个基线
1. 使用高斯朴素贝叶斯5折结果0.55817775599999997([0.5515,0.5635,0.56775,0.553,0.55513878])
2. multinomially分布数据的贝叶斯,5折cv结果[ 0.75197563  0.78392127  0.78745997  0.7740566   0.76901726], avg: 0.773286144565, 将测试结果提交平台f1值0.7137

### 迭代

#### 迭代1
1. 特征选择
    1. 使用卡方检验,选择top k个特征,实验结果如下表，可以看到k=2500时效果最好

    k|f1
    ---|---
    100|0.747587259315
    1000|0.790439509877
    1500|0.792489584896
    2000|0.793739684921
    2300|0.793389422356
    2500|0.793789472368
    2800|0.791789297324
    3000|0.790139197299
    3500|0.790139197299
    4000|0.779638822206
    全部|0.773286144565

2. badcase 分析
    * 有些单个字有含义，如训练集中第2个样本含'艹'
    
    id|words|comment
    ---|---|---
    2|我来/艹|我来对结果产生影响
    5||含有否定词'不'
    22||含有否定词'不'


3. 亚光跑出根据类别2x2列联表的fisher精确检验pvalue的-log10转换后的score：data_train_processed_fisher , 
    * 每个样本计算出3个特征
    cat data/data_train.seg | python src/gen_fisher_feat.py > data/data_train_fisher.sample
    cat data/data_test.seg | python src/gen_fisher_feat.py > data/data_test_fisher.sample
    * 独立使用fisher特征，
        1. 使用multinomially分布数据的贝叶斯, 5折cv结果[ 0.71        0.7275      0.71975     0.723       0.71467867], avg: 0.718985733933;
        2. 使用LR模型， 5折cv结果, [ 0.743      0.74875    0.75125    0.74475    0.7391848], avg: 0.74538695924
        3. LR加上bagging，效果变差, [ 0.44616756  0.44661986  0.43141045  0.44339829  0.39516147], avg: 0.43255152404
    * 与原有特征融合，效果变差，有可能是过拟合 [ 0.6635      0.691       0.6775      0.68325     0.67391848], avg: 0.6778336959
    * 使用bagging/adboost 效果都很差, [ 0.10586422  0.16317525  0.12463332  0.09681973  0.09065407], avg: 0.116229319125
4. 线上评估结果：0.7393

#### 迭代2
1. 考虑使用训练好的word2vec 模型找出本例中较为相近的『word』, 然后将相近的word进行合并，形成新的主题特征, 利用亚光训练好的word2vector 模型：data/data_feed_in_word2vec.txt.50.bin
   python src/vocab_sim.py > data/vocab_simi.dat 2> log/vocab_sim.log 
   python src/vocab_2_theme.py > data/vocab_2_theme.dict 2> log/vocab_2_theme.log

2. 结果：提到线上f1值没有明显提升, 0.7382


#### 迭代3
1. 使用stacking 模型融合, ji
2. 训练集评估结果为0.822441122056， 线上测试评估结果为0.7489

#### 迭代4
1. xgboost
调参数

eta|max_depth|booster|num_round|f1
---|---|---|---|---|---
0.5|3|gblinear|3|0.831641435359
0.6|3|gblinear|3|0.830341310328
0.4|3|gblinear|3|0.832241347837
0.3|3|gblinear|3|0.830591260315
0.4|2|gblinear|3|0.832241347837
0.4|1|gblinear|3|0.832241347837
0.4|1|gblinear|10|0.834291385346

2. 使用stacking 模型融合亚光的结果

#### 迭代5
1. 切词字典添加2015/2016网络流行词,如"蓝瘦香菇/套路/撩妹"
2. 噪声过滤。将特征全0的训练样本的y值更新为1(中性)
3. 将特征归一化，即只要特征值>0更新为1，因为模型主要看某个词是否出现

model|cv-5
---|---
xgboost|0.845442623756
NB|0.835691597874
Perceptron|0.827242284778
RF|0.805240532921
NB&XGBOOST|0.849142457123
