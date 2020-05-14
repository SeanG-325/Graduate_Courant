import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, config, wordEmbedding):
        self.inputX=tf.placeholder(tf.int32,shape=[None, config.sequenceLength],name="inputX")
        self.inputY=tf.placeholder(tf.int32,shape=[None], name="inputY")

        self.dropoutKeepProb=tf.placeholder(tf.float32,name="dropoutkeepProb")

        l2loss=tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W=tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            self.embeddedWords=tf.nn.embedding_lookup(self.W, self.inputX)
            self.embeddedWordsExpanded=tf.expand_dims(self.embeddedWords,-1)

        pooledOutputs=[]
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                filterShape=[filterSize, config.model.embeddingSize,1,config.model.numFilters]
                W=tf.Variable(tf.truncated_normal(filterShape,stddev=0.1), name="W")
                b=tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]),name="b")
                conv=tf.nn.conv2d(self.embeddedWordsExpanded,W,strides=[1,1,1,1],padding="VALID",name="conv")

                h=tf.nn.relu(tf.nn.bias_add(conv,b), name="relu")

                pooled=tf.nn.max_pool(h,ksize=[1,config.sequenceLength-filterSize+1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")

                pooledOutputs.append(pooled)

        numFiltersTotal=config.model.numFilters * len(config.model.filterSizes)

        self.hPool=tf.concat(pooledOutputs,3)

        self.hPoolFlat=tf.reshape(self.hPool,[-1,numFiltersTotal])

        with tf.name_scope("dropout"):
            self.hDrop=tf.nn.dropout(self.hPoolFlat,config.model.dropkeepProb)

        with tf.name_scope("output"):
            outputW=tf.get_variable("outputW",shape=[numFiltersTotal,config.numclasses],initializer=tf.contrib.layers.xavier_initializer())
            outputB=tf.Variable(tf.constant(0.1, shape=[config.numclasses]),name="outputB")
            l2loss+=tf.nn.l2_loss(outputW)
            l2loss+=tf.nn.l2_loss(outputB)
            self.logits=tf.nn.xw_plus_b(self.hDrop,outputW,outputB,name="logits")
            self.prediction=tf.argmax(self.logits,axis=-1,name="predictions")
            # print(self.prediction)

        with tf.name_scope("loss"):
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.inputY)
            self.loss=tf.reduce_mean(losses)+config.model.l2regLambda * l2loss


