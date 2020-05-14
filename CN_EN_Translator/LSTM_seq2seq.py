import os
import DataPreProcess
import tensorflow as tf
import numpy as np
import tensorflow.contrib as con

data=DataPreProcess.Dataset()
data.Datagen()
display=100
dropKeepprob=0.7
epochs=1
batch_size=16
class LSTM_seq2seq(object):
    def __init__(self,wordEmbedding_en,wordEmbedding_cn):
        self.seqLength = 50
        self.embeddingSize=512
        self.hiddenSize=512
        self.cnvocabSize=6900#
        self.envocabSize=6142#
        self.max_step=self.seqLength
        self.alpha=0.001
        self.epochs=10
        self.en_embed_id=tf.range(0,self.envocabSize)
        # self.dropKeepprob = 0.8
        self.input_en = tf.placeholder(tf.int32, [None, self.seqLength], name="input_en")
        self.input_en_len=tf.placeholder(tf.int32,[None],name="input_en_len")
        self.input_cn = tf.placeholder(tf.int32, [None, self.seqLength], name="input_cn")
        self.input_cn_len = tf.placeholder(tf.int32, [None], name="input_cn_len")
        self.dropKeepprob=tf.placeholder(tf.float32,name="dropKeepprob")
        self.de_lstm=tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)


        input_en=self.input_en
        decoder_inputs = tf.strided_slice(input_en, [0, 0], [batch_size, -1], [1, 1])
        decoder_inputs = tf.concat([tf.fill([batch_size, 1], data.word2idx_en["<GO>"]), decoder_inputs], 1)
        input_en=decoder_inputs

        with tf.name_scope("embedding_en"):
            self.W1=tf.Variable(tf.cast(wordEmbedding_en,tf.float32,name="word2vec_en"), name="W1")
            self.embeddedwords_en=tf.nn.embedding_lookup(self.W1,input_en)
            self.en_embed=tf.nn.embedding_lookup(self.W1,self.en_embed_id)

        with tf.name_scope("embedding_cn"):
            self.W2=tf.Variable(tf.cast(wordEmbedding_cn,tf.float32,name="word2vec_cn"), name="W2")
            self.embeddedwords_cn=tf.nn.embedding_lookup(self.W2,self.input_cn)

        with tf.name_scope("En"):
            fwcell=tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)
            LSTMfwcell=tf.nn.rnn_cell.DropoutWrapper(fwcell,output_keep_prob=self.dropKeepprob)
            #bwcell=tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)
            #LSTMbwcell=tf.nn.rnn_cell.DropoutWrapper(bwcell,output_keep_prob=self.dropKeepprob)
            Cn_outputs_, Cn_currentstate = tf.nn.dynamic_rnn(LSTMfwcell,
                                                             self.embeddedwords_cn,
                                                             self.input_cn_len,
                                                             dtype=tf.float32,
                                                             scope="LSTM_CN")
            self.state=tf.concat(Cn_outputs_,2)

        CnState_=self.state[:,-1,:]
        CnStateSize=self.hiddenSize
        CnState=tf.reshape(CnState_,[-1,CnStateSize])
        outputlayer = tf.layers.Dense(self.envocabSize)
        decoder_cell=tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)

        with tf.variable_scope("De"):
            train_helper=con.seq2seq.TrainingHelper(self.embeddedwords_en,self.input_en_len,name="TrainHelper")
            train_decoder=con.seq2seq.BasicDecoder(decoder_cell,train_helper,Cn_currentstate,outputlayer)
            self.train_decoder_ans,_,_=con.seq2seq.dynamic_decode(train_decoder,impute_finished=True,maximum_iterations=self.max_step)

        with tf.variable_scope("De",reuse=True):
            start_tokens = tf.tile(tf.constant([data.word2idx_en["<GO>"]], dtype=tf.int32), [batch_size], name="start_tokens")
            infer_helper=con.seq2seq.GreedyEmbeddingHelper(self.en_embed,start_tokens,data.word2idx_en["<EOS>"])
            infer_decoder=con.seq2seq.BasicDecoder(decoder_cell,infer_helper,Cn_currentstate,outputlayer)
            self.infer_decoder_ans, _, _ = con.seq2seq.dynamic_decode(infer_decoder, impute_finished=True,maximum_iterations=self.max_step)




def get_batches(cn,en,batch_size):
    for batch_i in range(len(cn)//batch_size):
        start_i=batch_i*batch_size

        cn_batch=np.array(cn[start_i:start_i+batch_size],dtype="int32")
        en_batch=np.array(en[start_i:start_i+batch_size],dtype="int32")

        cn_len = []
        en_len = []

        for seq in cn_batch:
            cn_len.append(len(seq))

        for seq in en_batch:
            en_len.append(len(seq))

        cn_len = np.array(cn_len,dtype="int32")
        en_len = np.array(en_len, dtype="int32")

        yield cn_batch, en_batch, cn_len, en_len




with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        seq2seq = LSTM_seq2seq(data.wordEmbedding_en, data.wordEmbedding_cn)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        train_logits = tf.identity(seq2seq.train_decoder_ans.rnn_output, name="logits")
        infer_logits = tf.identity(seq2seq.infer_decoder_ans.sample_id, name="predictions")
        masks=tf.sequence_mask(seq2seq.input_en_len,seq2seq.max_step,dtype=tf.float32,name="mask")
        with tf.name_scope("optimization"):
            cost = con.seq2seq.sequence_loss(train_logits, seq2seq.input_cn, masks)
            optimizer = tf.train.AdamOptimizer(seq2seq.alpha)
            gredient = optimizer.compute_gradients(cost)
            clipped_gredient = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gredient if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gredient,globalStep)


        writer = tf.summary.FileWriter(os.path.join("tmp", "logs"), sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(epochs):
            for batch_i, (cn_batch, en_batch, cn_len, en_len) in enumerate(
                get_batches(data.cn_con, data.en_con, batch_size)):

                _,loss =sess.run([train_op, cost],{seq2seq.input_cn: cn_batch,
                                                   seq2seq.input_en:en_batch,
                                                   seq2seq.input_cn_len: cn_len,
                                                   seq2seq.input_en_len:en_len,
                                                   seq2seq.dropKeepprob:0.7})
                                               
                if batch_i % display==0 and batch_i>0:
                    batch_train_logits=sess.run(infer_logits,{seq2seq.input_cn:cn_batch,
                                                              seq2seq.input_cn_len:cn_len,
                                                              seq2seq.input_en_len:en_len,
                                                              seq2seq.dropKeepprob:1.0})
                    print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'.format(epoch_i, batch_i, len(data.cn_con) // batch_size, loss))

        saver=tf.train.Saver()
        saver.save(sess, os.path.join("tmp", "checkpoints", "model.ckpt"))
        print('Model Trained and Save to {}'.format(os.path.join("tmp", "checkpoints", "model.ckpt")))
