import os
from random import shuffle
from math import floor
import DataProcess
import tensorflow as tf
import numpy as np
import tensorflow.contrib as con

data = DataProcess.Dataset()
data.Datagen()
display = 100
dropKeepprob = 0.8
epochs = 7
batch_size = 64


class LSTM_seq2seq(object):
    def __init__(self, wordEmbedding_text):
        self.seqLength = 75
        self.embeddingSize = 512
        self.hiddenSize = 512
        self.hiddenSize_de = 512
        self.vocabSize = data.vocablen
        self.max_step = data.absmax
        self.alpha = 0.001
        self.epochs = 7
        self.LSTM_num=3
        # self.en_embed_id = tf.range(0, self.envocabSize)
        # self.dropKeepprob = 0.8
        self.input_text = tf.placeholder(tf.int32, [None, self.seqLength], name="input_text")
        self.input_text_len = tf.placeholder(tf.int32, (None,), name="input_text_len")
        self.input_abs = tf.placeholder(tf.int32, [None, self.max_step], name="input_abs")
        self.input_abs_len = tf.placeholder(tf.int32, (None,), name="input_abs_len")
        self.dropKeepprob = tf.placeholder(tf.float32, name="dropKeepprob")
        #self.de_lstm = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize)

        input_abs = self.input_abs
        self.decoder_inputs = tf.strided_slice(input_abs, [0, 0], [batch_size, -1], [1, 1])
        self.decoder_inputs1 = tf.concat([tf.fill([batch_size, 1], data.word2idx_text["<GO>"]), self.decoder_inputs], 1)
        self.input_abs1 = self.decoder_inputs1
        self.W = tf.Variable(tf.cast(wordEmbedding_text, tf.float32, name="word2vec"), name="W")

        with tf.name_scope("embedding_abs"):
            #self.W = tf.Variable(tf.cast(wordEmbedding_abs, tf.float32, name="word2vec_en"), name="W")
            #self.W = tf.Variable(tf.random_uniform([data.vocablen, self.embeddingSize]))
            #self.W = tf.contrib.layers.embed_sequence(self.input_abs, data.vocablen, self.embeddingSize)
            self.embeddedwords_abs = tf.nn.embedding_lookup(self.W, self.input_abs1)
            #self.embeddedwords_text = self.W

        with tf.name_scope("embedding_text"):
            #self.W = tf.Variable(tf.cast(wordEmbedding_text, tf.float32, name="word2vec_cn"), name="W")
            self.embeddedwords_text = tf.nn.embedding_lookup(self.W, self.input_text)
            #self.W = tf.Variable(tf.cast(wordEmbedding_cn, tf.float32, name="word2vec_cn"), name="W")
            #self.W = tf.contrib.layers.embed_sequence(self.input_cn, data.cn_vocablen, self.embeddingSize)
            #self.embeddedwords_text = tf.contrib.layers.embed_sequence(self.input_abs, data.vocablen, self.embeddingSize)

            #self.embeddedwords_cn=con.layers.embed_sequence(self.input_cn,self.cnvocabSize,self.embeddingSize)

        def get_lstm(rnn_size):
            lstm = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
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
                                                                        self.embeddedwords_text,
                                                                        self.input_text_len,
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
        outputlayer = tf.layers.Dense(self.vocabSize)
        decoder_cell = con.rnn.MultiRNNCell([get_lstm(self.hiddenSize) for _ in range(self.LSTM_num)])
        #decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hiddenSize_de)
        with tf.variable_scope("De"):
            train_helper = con.seq2seq.TrainingHelper(self.embeddedwords_abs, self.input_abs_len, name="TrainHelper")

            attention_m=con.seq2seq.BahdanauAttention(num_units=self.hiddenSize_de,memory=Cn_outputs_,memory_sequence_length=self.input_text_len)

            traininfer_decoder_cell=con.seq2seq.AttentionWrapper(decoder_cell,attention_m,attention_layer_size=self.hiddenSize_de)

            train_decoder = con.seq2seq.BasicDecoder(traininfer_decoder_cell, train_helper,
                                                     traininfer_decoder_cell.zero_state(batch_size=batch_size,
                                                                                   dtype=tf.float32).clone(
                                                         cell_state=Cn_currentstate), outputlayer)

            self.train_decoder_ans, _, _ = con.seq2seq.dynamic_decode(train_decoder, impute_finished=True,
                                                                      maximum_iterations=self.max_step)
        
        #with tf.variable_scope("De", reuse=tf.AUTO_REUSE):
            start_tokens = tf.tile(tf.constant([data.word2idx_text["<GO>"]], dtype=tf.int32), [batch_size],
                                   name="start_tokens")

            #attention_m = con.seq2seq.BahdanauAttention(num_units=self.hiddenSize_de, memory=Cn_outputs_,
            #                                            memory_sequence_length=self.input_text_len)
            infer_helper = con.seq2seq.GreedyEmbeddingHelper(self.W, start_tokens, data.word2idx_text["<EOS>"])

            #infer_decoder_cell = con.seq2seq.AttentionWrapper(decoder_cell,attention_m,attention_layer_size=self.hiddenSize_de)

            infer_decoder = con.seq2seq.BasicDecoder(traininfer_decoder_cell, infer_helper,
                                                     traininfer_decoder_cell.zero_state(batch_size=batch_size,
                                                                                   dtype=tf.float32).clone(
                                                         cell_state=Cn_currentstate), outputlayer)

            self.infer_decoder_ans, _, _ = con.seq2seq.dynamic_decode(infer_decoder, impute_finished=True,
                                                                      maximum_iterations=self.max_step)
        

def get_batches(text, abs, text_len, abs_len, batch_size):

    #z = list(zip(text, abs, text_len,abs_len))
    #shuffle(z)
    #text[:], abs[:],  text_len[:],abs_len[:] = zip(*z)
    for batch_i in range(floor(min(len(abs),len(text)) / batch_size)):
        start_i = batch_i * batch_size

        text_batch = np.array(text[start_i:start_i + batch_size], dtype="int32")
        abs_batch = np.array(abs[start_i:start_i + batch_size], dtype="int32")

        text_len_batch=np.array(text_len[start_i:start_i + batch_size],dtype="int32")
        abs_len_batch=np.array(abs_len[start_i:start_i + batch_size],dtype="int32")

        yield text_batch, abs_batch, text_len_batch, abs_len_batch


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        seq2seq = LSTM_seq2seq(data.wordEmbedding_text)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        train_logits = tf.identity(seq2seq.train_decoder_ans.rnn_output, name="logits")
        infer_logits = tf.identity(seq2seq.infer_decoder_ans.sample_id, name="predictions")
        masks = tf.sequence_mask(seq2seq.input_abs_len, data.maxLen_Abs, dtype=tf.float32, name="mask")
        with tf.name_scope("optimization"):
            cost = con.seq2seq.sequence_loss(train_logits, seq2seq.input_abs, masks)
            optimizer = tf.train.AdamOptimizer(seq2seq.alpha)
            gredient = optimizer.compute_gradients(cost)
            clipped_gredient = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gredient if grad is not None]
            train_op = optimizer.apply_gradients(clipped_gredient, globalStep)

        writer = tf.summary.FileWriter(os.path.join("tmp", "logs"), sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(epochs):
            for batch_i, (text_batch, abs_batch, text_len, abs_len) in enumerate(
                    get_batches(data.text_con, data.abs_con, data.text_con_len, data.abs_con_len,batch_size)):

                _, loss = sess.run([train_op, cost], {seq2seq.input_text: text_batch,
                                                                           seq2seq.input_abs: abs_batch,
                                                                           seq2seq.input_text_len: text_len,
                                                                           seq2seq.input_abs_len: abs_len,
                                                                           seq2seq.dropKeepprob: 0.9})
                #print(temp)
                
                if batch_i % display == 0 and batch_i > 0:
                    batch_train_logits = sess.run(infer_logits, {seq2seq.input_text: text_batch,
                                                                 seq2seq.input_text_len: text_len,
                                                                 seq2seq.input_abs_len: abs_len,    
                                                                 seq2seq.dropKeepprob: 1.0})
                print('Epoch {:>3} Batch {:>4}/{} - Loss: {:>6.6f}'.format(epoch_i, batch_i,
                                                                           len(data.text_con) // batch_size, loss))

            saver = tf.train.Saver()
            saver.save(sess, os.path.join("tmp", "checkpoints", "model.ckpt"),write_meta_graph=True)
            print('Model Trained and Save to {}'.format(os.path.join("tmp", "checkpoints", "model.ckpt")))

        saver = tf.train.Saver()
        saver.save(sess, os.path.join("tmp", "checkpoints", "model.ckpt"),write_meta_graph=True)
        print('Model Trained and Save to {}'.format(os.path.join("tmp", "checkpoints", "model.ckpt")))
