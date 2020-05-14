import tensorflow as tf
class BiLSTM(object):
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
            outputs, self.current_state=tf.nn.bidirectional_dynamic_rnn(lstmFwCell,lstmBwCell,self.embeddedWords,dtype=tf.float32,scope="bi-lstm")
            self.embeddedWords=tf.concat(outputs,2)
        finalOutput=self.embeddedWords[:,-1,:]
        outputSize=config.model.hiddenSizes[-1]*2
        output=tf.reshape(finalOutput,[-1,outputSize])

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