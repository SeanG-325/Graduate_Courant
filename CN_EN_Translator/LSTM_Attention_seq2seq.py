import os
from random import shuffle

import DataPreProcess
import tensorflow as tf
import numpy as np
import tensorflow.contrib as con

data = DataPreProcess.Dataset()
data.Datagen()
display = 100
dropKeepprob = 0.8
epochs = 15
batch_size = 32


class LSTM_seq2seq(object):
    def __init__(self, wordEmbedding_en, wordEmbedding_cn):
        self.seqLength = 30
        self.embeddingSize = 512
        self.hiddenSize = 512
        self.hiddenSize_de = 512*2
        self.cnvocabSize = data.cn_vocablen  #
        self.envocabSize = data.en_vocablen  #
        self.max_step = self.seqLength
        self.alpha = 0.001
        self.epochs = 15
        self.LSTM_num=3
        # self.en_embed_id = tf.range(0, self.envocabSize)
        # self.dropKeepprob = 0.8
        self.input_en = tf.placeholder(tf.int32, [None, self.seqLength], name="input_en")
        self.input_en_len = tf.placeholder(tf.int32, (None,), name="input_en_len")
        self.input_cn = tf.placeholder(tf.int32, [None, self.seqLength], name="input_cn")
        self.input_cn_len = tf.placeholder(tf.int32, (None,), name="input_cn_len")
        self.dropKeepprob = tf.placeholder(tf.float32, name="dropKeepprob")
        #self.de_lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)

        input_en = self.input_en
        self.decoder_inputs = tf.strided_slice(input_en, [0, 0], [batch_size, -1], [1, 1])
        self.decoder_inputs1 = tf.concat([tf.fill([batch_size, 1], data.word2idx_en["<GO>"]), self.decoder_inputs], 1)
        self.input_en = self.decoder_inputs1

        with tf.name_scope("embedding_en"):
            #self.W1 = tf.Variable(tf.cast(wordEmbedding_en, tf.float32, name="word2vec_en"), name="W1")
            self.W1 = tf.Variable(tf.random_uniform([data.en_vocablen, self.embeddingSize]))
            #self.W1 = tf.contrib.layers.embed_sequence(self.input_en, data.en_vocablen, self.embeddingSize)
            self.embeddedwords_en = tf.nn.embedding_lookup(self.W1, self.input_en)
            #self.embeddedwords_en = self.W1

        with tf.name_scope("embedding_cn"):
            #self.W2 = tf.Variable(tf.cast(wordEmbedding_cn, tf.float32, name="word2vec_cn"), name="W2")
            #self.W2 = tf.contrib.layers.embed_sequence(self.input_cn, data.cn_vocablen, self.embeddingSize)
            self.embeddedwords_cn = tf.contrib.layers.embed_sequence(self.input_cn, data.cn_vocablen, self.embeddingSize)

            #self.embeddedwords_cn=con.layers.embed_sequence(self.input_cn,self.cnvocabSize,self.embeddingSize)

        def get_lstm(rnn_size):
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            lstm=tf.nn.rnn_cell.DropoutWrapper(lstm,self.dropKeepprob)
            return lstm

        with tf.name_scope("En"):
            #fwcell = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)
            LSTMfw = con.rnn.MultiRNNCell([get_lstm(self.hiddenSize) for _ in range(self.LSTM_num)])
            #LSTMfwcell = tf.nn.rnn_cell.DropoutWrapper(fwcell, output_keep_prob=self.dropKeepprob)
            LSTMbw = con.rnn.MultiRNNCell([get_lstm(self.hiddenSize) for _ in range(self.LSTM_num)])

            #bwcell=tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)
            #LSTMbwcell=tf.nn.rnn_cell.DropoutWrapper(bwcell,output_keep_prob=self.dropKeepprob)

            Cn_outputs_, currentstate = tf.nn.bidirectional_dynamic_rnn(LSTMfw,
                                                                        LSTMbw,
                                                                        self.embeddedwords_cn,
                                                                        self.input_cn_len,
                                                                        dtype=tf.float32,
                                                                        scope="BiLSTM_CN")
            #print(currentstate.count())
            fw = currentstate[0]
            bw = currentstate[1]
            Cn_currentstate = ()
            for i in range(self.LSTM_num):
                state_c = fw[i][0] + bw[i][0]
                state_h = fw[i][1] + bw[i][1]
                Cn_currentstate+=(tf.contrib.rnn.LSTMStateTuple(c=state_c,h=state_h),)


            #final_state_c = outputs_state_fw[0] + outputs_state_bw[0]
            #final_state_h = outputs_state_fw[1] + outputs_state_bw[1]
            #Cn_currentstate = tf.contrib.rnn.LSTMStateTuple(c=final_state_c,
            #                                                h=final_state_h)
            #self.state = tf.concat(Cn_currentstate, -1)
            #Cn_currentstate=self.state
            Cn_outputs_=tf.concat(Cn_outputs_, -1)
            '''
            Cn_outputs_, Cn_currentstate = tf.nn.dynamic_rnn(LSTMfw,
                                                             self.embeddedwords_cn,
                                                             self.input_cn_len,
                                                             dtype=tf.float32)
            self.state = tf.concat(Cn_currentstate, 2)
'''


        #CnState_ = self.state[:, -1, :]
        #CnStateSize = self.hiddenSize
        #CnState = tf.reshape(CnState_, [-1, CnStateSize])
        outputlayer = tf.layers.Dense(self.envocabSize)
        decoder_cell = con.rnn.MultiRNNCell([get_lstm(self.hiddenSize) for _ in range(self.LSTM_num)])
        #decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize_de)
        with tf.variable_scope("De"):
            train_helper = con.seq2seq.TrainingHelper(self.embeddedwords_en, self.input_en_len, name="TrainHelper")

            attention_m=con.seq2seq.BahdanauAttention(num_units=self.hiddenSize_de,memory=Cn_outputs_,memory_sequence_length=self.input_cn_len)

            train_decoder_cell=con.seq2seq.AttentionWrapper(decoder_cell,attention_m,attention_layer_size=self.hiddenSize_de)

            train_decoder = con.seq2seq.BasicDecoder(train_decoder_cell, train_helper,
                                                     train_decoder_cell.zero_state(batch_size=batch_size,
                                                                                   dtype=tf.float32).clone(
                                                         cell_state=Cn_currentstate), outputlayer)

            self.train_decoder_ans, _, _ = con.seq2seq.dynamic_decode(train_decoder, impute_finished=True,
                                                                      maximum_iterations=self.max_step)

        with tf.variable_scope("De", reuse=tf.AUTO_REUSE):
            start_tokens = tf.tile(tf.constant([data.word2idx_en["<GO>"]], dtype=tf.int32), [batch_size],
                                   name="start_tokens")

            attention_m = con.seq2seq.BahdanauAttention(num_units=self.hiddenSize_de, memory=Cn_outputs_,
                                                        memory_sequence_length=self.input_cn_len)
            infer_helper = con.seq2seq.GreedyEmbeddingHelper(self.W1, start_tokens, data.word2idx_en["<EOS>"])

            infer_decoder_cell = con.seq2seq.AttentionWrapper(decoder_cell,attention_m,attention_layer_size=self.hiddenSize_de)

            infer_decoder = con.seq2seq.BasicDecoder(infer_decoder_cell, infer_helper,
                                                     train_decoder_cell.zero_state(batch_size=batch_size,
                                                                                   dtype=tf.float32).clone(
                                                         cell_state=Cn_currentstate), outputlayer)

            self.infer_decoder_ans, _, _ = con.seq2seq.dynamic_decode(infer_decoder, impute_finished=True,
                                                                      maximum_iterations=self.max_step)


def get_batches(cn, en, cn_len, en_len, batch_size):
    '''
    z = list(zip(cn, en,cn_len,en_len))
    shuffle(z)
    cn[:], en[:], cn_len[:],en_len[:] = zip(*z)
    '''
    for batch_i in range(len(cn) // batch_size):
        start_i = batch_i * batch_size

        cn_batch = np.array(cn[start_i:start_i + batch_size], dtype="int32")
        en_batch = np.array(en[start_i:start_i + batch_size], dtype="int32")

        cn_len_batch=np.array(cn_len[start_i:start_i + batch_size],dtype="int32")
        en_len_batch=np.array(en_len[start_i:start_i + batch_size],dtype="int32")

        yield cn_batch, en_batch, cn_len_batch, en_len_batch


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
        masks = tf.sequence_mask(seq2seq.input_en_len, seq2seq.max_step, dtype=tf.float32, name="mask")
        with tf.name_scope("optimization"):
            cost = con.seq2seq.sequence_loss(train_logits, seq2seq.input_cn, masks)
            optimizer = tf.train.AdamOptimizer(seq2seq.alpha)
            gredient = optimizer.compute_gradients(cost)
            clipped_gredient = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gredient if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gredient, globalStep)

        writer = tf.summary.FileWriter(os.path.join("tmp", "logs"), sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(epochs):
            for batch_i, (cn_batch, en_batch, cn_len, en_len) in enumerate(
                    get_batches(data.cn_con, data.en_con, data.cn_con_len, data.en_con_len,batch_size)):

                _, loss = sess.run([train_op, cost], {seq2seq.input_cn: cn_batch,
                                                                           seq2seq.input_en: en_batch,
                                                                           seq2seq.input_cn_len: cn_len,
                                                                           seq2seq.input_en_len: en_len,
                                                                           seq2seq.dropKeepprob: 0.8})
                #print(temp)

                if batch_i % display == 0 and batch_i > 0:
                    batch_train_logits = sess.run(infer_logits, {seq2seq.input_cn: cn_batch,
                                                                 seq2seq.input_cn_len: cn_len,
                                                                 seq2seq.input_en_len: en_len,
                                                                 seq2seq.dropKeepprob: 1.0})
                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.4f}'.format(epoch_i, batch_i,
                                                                           len(data.cn_con) // batch_size, loss))

        saver = tf.train.Saver()
        saver.save(sess, os.path.join("tmp", "checkpoints", "model.ckpt"))
        print('Model Trained and Save to {}'.format(os.path.join("tmp", "checkpoints", "model.ckpt")))
