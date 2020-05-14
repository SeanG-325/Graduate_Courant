import os
import time
import datetime
import random
import gensim
import tensorflow as tf
import numpy as np
import jieba
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
class PreConfig(object):
    Path="./Data/"
    Word2VecModelPath="./Data/model.bin"
    TrainContentsPath="./Data/"
    Labels=['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    word2idxPath="./data/word2idx.json"
    label2idxPath="./data/label2idx.json"
    modelPath_TCNN="./model/textCNN/"
    modelPath_BiLSTM="./model/BiLSTM/"
    modelPath_BiLSTM_Attention="./model/BiLSTM_Attention/"
    modelPath_GRU_Attention="./model/GRU_Attention/"
    modelPath_Transformer="./model/Transformer/"


class TrainingConfig(object):
    epoches=3
    evaluateEvery_BiLSTM=1562/2
    checkPointEvery_BiLSTM=1562/2
    evaluateEvery_TCNN = 390
    checkPointEvery_TCNN = 390
    evaluateEvery_Transformer = 3125
    checkPointEvery_Transformer = 3125
    LearningRate=0.001

class ModelConfig(object):
    embeddingSize=200
    numFilters=128
    filterSizes=[2,3,4,5]
    dropkeepProb=0.5
    l2regLambda=0.001
    hiddenSizes=[512, 512]

    #Transformer Hyperparameters
    numBlocks=2
    numHeads=8
    epsilon=1e-8
    keepProb=0.9
    filters=128

class Config(object):
    sequenceLength=400
    batchSize=16
    dataSource="./Data/"
    stopWordScource="./Data/ChineseStopwords_Baidu.txt"
    numclasses=10
    rate=0.7
    training=TrainingConfig()
    model=ModelConfig()
    preconfig=PreConfig()

config=Config()
