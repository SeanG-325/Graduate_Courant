import tensorflow as tf

from Config import config


class BiLSTM_Attention(object):
    def __init__(self, config, wordEmbedding):
        self.inputX=tf.placeholder(tf.int32,[None, config.sequenceLength], name="inputX")
        self.inputY=tf.placeholder(tf.int32,[None], name="inputY")
        self.dropKeepprob=tf.placeholder(tf.float32,name="dropKeepprob")
        l2Loss=tf.constant(0.0)
        with tf.name_scope("embedding"):
            self.W=tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32,name="word2vec"),name="W")
            self.embeddedWords=tf.nn.embedding_lookup(self.W,self.inputX)
        with tf.name_scope("BiLSTM"):
            hiddenSize=config.model.hiddenSizes[0]
            fwCell=tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True)
            lstmFwCell=tf.nn.rnn_cell.DropoutWrapper(fwCell,output_keep_prob=self.dropKeepprob)
            bwCell=tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize,state_is_tuple=True)
            lstmBwCell=tf.nn.rnn_cell.DropoutWrapper(bwCell,output_keep_prob=self.dropKeepprob)
            outputs_, self.current_state=tf.nn.bidirectional_dynamic_rnn(lstmFwCell,lstmBwCell,self.embeddedWords,dtype=tf.float32,scope="bi-lstm")
            self.embeddedWords=tf.concat(outputs_,2)

        outputs=tf.split(self.embeddedWords,2,-1)

        with tf.name_scope("attention"):
            H=outputs[0]+outputs[1]

        output=self.Attention(H)
        outputSize=config.model.hiddenSizes[-1]

        with tf.name_scope("output"):
            outputW=tf.get_variable(name="outputW",shape=[outputSize,config.numclasses],initializer=tf.contrib.layers.xavier_initializer())
            outputB=tf.Variable(tf.constant(0.1,shape=[config.numclasses],name="outputB"))
            l2Loss+=tf.nn.l2_loss(outputW)
            l2Loss+=tf.nn.l2_loss(outputB)
            self.logits=tf.nn.xw_plus_b(output,outputW,outputB,name="logits")
            self.prediction=tf.argmax(self.logits,axis=-1,name="prediction")

        with tf.name_scope("loss"):
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.inputY)
            self.loss=tf.reduce_mean(losses)+config.model.l2regLambda*l2Loss

    def Attention(self,H):
        hiddenSize=config.model.hiddenSizes[-1]

        W=tf.Variable(tf.random_normal([hiddenSize],stddev=0.1))

        M=tf.tanh(H)

        newM=tf.matmul(tf.reshape(M,[-1,hiddenSize]),tf.reshape(W,[-1,1]))

        restoreM=tf.reshape(newM,[-1,config.sequenceLength])

        self.alpha=tf.nn.softmax(restoreM)
        r=tf.matmul(tf.transpose(H,[0,2,1]),tf.reshape(self.alpha,[-1,config.sequenceLength,1]))
        squeezeR=tf.reshape(r,[-1,hiddenSize])
        sentenceRepren=tf.tanh(squeezeR)

        output=tf.nn.dropout(sentenceRepren,self.dropKeepprob)

        return output

