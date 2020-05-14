import tensorflow as tf
import os
import nltk
import numpy as np
import jieba
import json
import DataPreProcess

batchSize=32
cn="你是谁"
cn_seg=' '.join(jieba.cut(cn))
data=DataPreProcess.Dataset()
with open(data.w2idx_cn_name, "r", encoding="utf8") as fp:
    w2idx_cn=json.load(fp)
fp.close()

with open(data.w2idx_en_name, "r", encoding="utf8") as fp:
    w2idx_en=json.load(fp)
fp.close()

idx2w_cn={idx: word for word, idx in w2idx_cn.items()}
idx2w_en={idx: word for word, idx in w2idx_en.items()}

cn_ids=[w2idx_cn.get(item, w2idx_cn["<UNK>"]) for item in cn_seg.split()]

if len(cn_ids) >= data.seqLength:
    xIds = cn_ids[:data.seqLength]
else:
    xIds = cn_ids + [w2idx_cn["<PAD>"]] * (data.seqLength - len(cn_ids))

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    loader=tf.train.import_meta_graph("./tmp/checkpoints/model.ckpt.meta")
    loader.restore(sess,tf.train.latest_checkpoint("./tmp/checkpoints"))
    input_data=graph.get_tensor_by_name('input_cn:0')
    logits=graph.get_tensor_by_name('predictions:0')
    cn_seqLen=graph.get_tensor_by_name("input_cn_len:0")
    en_seqLen=graph.get_tensor_by_name("input_en_len:0")
    keep=graph.get_tensor_by_name("dropKeepprob:0")

    en_logits=sess.run(logits,{input_data:[xIds]*batchSize,
                               cn_seqLen:[len(cn_ids)]*batchSize,
                               en_seqLen:[len(xIds)]*batchSize,
                               keep:1.0})[0]

print("cn:{}".format(cn))
print("en:"+" ".join([idx2w_en[i] for i in en_logits]))