import tensorflow as tf
import numpy as np
from Config import config

def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition=[]
    for batch in range(batchSize):
        x=[]
        for step in range(sequenceLen):
            a=np.zeros(sequenceLen)
            a[step]=1
            x.append(a)
        embeddedPosition.append(x)
    return np.array(embeddedPosition,dtype="float32")

class Transformer(object):

    def _layerNormalization(self, inputs, scope="layerNorm"):
        ep=self.config.model.epsilon
        inputShape=inputs.get_shape()
        paramsShape=inputShape[-1:]
        mean,variance=tf.nn.moments(inputs,[-1],keep_dims=True)
        beta=tf.Variable(tf.zeros(paramsShape))
        gamma=tf.Variable(tf.ones(paramsShape))
        normalized=(inputs-mean)/((variance+ep)**(0.5))
        outputs=gamma*normalized+beta
        return outputs

    def _multiHeadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiHeadAttention"):
        numHeads=self.config.model.numHeads
        keepProb=self.config.model.keepProb

        if numUnits is None:
            numUnits=queries.get_shape().as_list()[-1]

        Q=tf.layers.dense(queries,numUnits,activation=tf.nn.relu)
        K=tf.layers.dense(keys,numUnits,activation=tf.nn.relu)
        V=tf.layers.dense(keys,numUnits,activation=tf.nn.relu)

        Q_=tf.concat(tf.split(Q,numHeads,axis=-1),axis=0)
        K_=tf.concat(tf.split(K,numHeads,axis=-1),axis=0)
        V_=tf.concat(tf.split(V,numHeads,axis=-1),axis=0)

        similary=tf.matmul(Q_,tf.transpose(K_,[0,2,1]))

        scaledSimilary=similary/(K_.get_shape().as_list()[-1]**(0.5))

        keyMasks=tf.tile(rawKeys,[numHeads,1])

        keyMasks=tf.tile(tf.expand_dims(keyMasks,1),[1,tf.shape(queries)[1],1])

        paddings=tf.ones_like(scaledSimilary)*(-2**(32+1))

        maskedSimilary=tf.where(tf.equal(keyMasks,0),paddings,scaledSimilary)

        if causality:
            diagVals=tf.ones_like(maskedSimilary[0,:,:])
            tril=tf.linalg.LinearOperatorLowerTriangular(diagVals).to_dense()
            masks=tf.tile(tf.expand_dims(tril,0),[tf.shape(maskedSimilary)[0],1,1])
            paddings=tf.ones_like(masks)*(-2**(32+1))
            maskedSimilary=tf.where(tf.equal(masks,0),paddings,maskedSimilary)

        weights=tf.nn.softmax(maskedSimilary)

        outputs=tf.matmul(weights,V_)

        outputs=tf.concat(tf.split(outputs,numHeads,axis=0),axis=-1)

        outputs=tf.nn.dropout(outputs,keep_prob=keepProb)

        outputs+=queries

        outputs=self._layerNormalization(outputs)

        return outputs

    def _feedforward2(self,inputs,filters,scope="multiHeadAttention"):
        params1={"inputs":inputs,"filters":filters[0],"kernel_size":1,"activation":tf.nn.relu,"use_bias": True}
        outputs=tf.layers.conv1d(**params1)
        params2 = {"inputs": outputs, "filters": filters[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs=tf.layers.conv1d(**params2)
        outputs+=inputs
        outputs=self._layerNormalization(outputs)
        return outputs

    def _feedforward(self,inputs,filters,scope="multiHeadAttention"):
        #units=inputs.get_shape().as_list()
        params1 = {"inputs": inputs, "units": 400, "activation": tf.nn.relu}
        outputs = tf.layers.dense(**params1)
        #units = outputs.get_shape().as_list()
        params2 = {"inputs": outputs, "units": 400, "activation": tf.nn.relu}
        outputs = tf.layers.dense(**params2)
        params3 = {"inputs": outputs, "units": filters[1], "activation": None}
        outputs = tf.layers.dense(**params3)
        outputs += inputs
        outputs = self._layerNormalization(outputs)
        return outputs

    def _positionEmbedding(self,scope="positionEmbedding"):
        batchSize=self.config.batchSize
        sequenceLen=self.config.sequenceLength
        embeddingSize=self.config.model.embeddingSize

        positionIndex=tf.tile(tf.expand_dims(tf.range(sequenceLen),0),[batchSize,1])

        positionEmbedding=np.array([[pos/np.power(10000,(i-i%2)/embeddingSize) for i in range(embeddingSize)] for pos in range(sequenceLen)])

        positionEmbedding[:,0::2]=np.sin(positionEmbedding[:,0::2])
        positionEmbedding[:,1::2]=np.cos(positionEmbedding[:,1::2])

        positionEmbedding_=tf.cast(positionEmbedding,tf.float32)

        positionEmbedded=tf.nn.embedding_lookup(positionEmbedding_,positionIndex)

        return positionEmbedded


    def __init__(self,config,wordEmbedding):
        self.inputX=tf.placeholder(tf.int32,[None,config.sequenceLength],name="inputX")
        self.inputY=tf.placeholder(tf.int32,[None],name="inputY")
        self.dropKeepProb=tf.placeholder(tf.float32,name="dropKeepProb")
        self.embeddedPosition=tf.placeholder(tf.float32,[None,config.sequenceLength,config.sequenceLength],name="embeddedPosition")

        self.config=config

        l2loss=tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W=tf.Variable(tf.cast(wordEmbedding,tf.float32,name="word2vec"),name="W")
            self.embedded=tf.nn.embedding_lookup(self.W,self.inputX)
            self.embeddedWords=tf.concat([self.embedded,self.embeddedPosition],-1)

        with tf.name_scope("transformer"):
            for i in range(config.model.numBlocks):
                with tf.name_scope("transformer-{}".format(i+1)):
                    multiHeadAtt=self._multiHeadAttention(rawKeys=self.inputX,queries=self.embeddedWords,keys=self.embeddedWords)
                    self.embeddedWords=self._feedforward(multiHeadAtt,[config.model.filters,config.model.embeddingSize+config.sequenceLength])
                    outputs=tf.reshape(self.embeddedWords,[-1,config.sequenceLength*(config.model.embeddingSize+config.sequenceLength)])
                    outputSize=outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs=tf.nn.dropout(outputs,keep_prob=self.dropKeepProb)

        with tf.name_scope("output"):
            outputW=tf.get_variable(name="outputW",shape=[outputSize,config.numclasses],initializer=tf.contrib.layers.xavier_initializer())
            outputB=tf.Variable(tf.constant(0.1,shape=[config.numclasses]),name="outputB")
            l2loss+=tf.nn.l2_loss(outputW)
            l2loss+=tf.nn.l2_loss(outputB)
            self.logits=tf.nn.xw_plus_b(outputs,outputW,outputB,name="logits")
            self.predictions=tf.argmax(self.logits,axis=-1,name="predictions")

        with tf.name_scope("loss"):
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.inputY)

        self.loss=tf.reduce_mean(losses)+config.model.l2regLambda*l2loss

