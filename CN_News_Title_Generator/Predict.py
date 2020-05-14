import tensorflow as tf
import os
import nltk
import numpy as np
import jieba
import json
import DataProcess
import re

batchSize=64
text="11日，在第十届海外高层次人才座谈会上，87岁的袁隆平院士在演讲开场时谦虚地说：\"I speak poor English\"。随后他全程用英语向世界介绍超级杂交水稻，赢得全场掌声！（每日经济新闻）人民日报的秒拍视频网友：中国的骄傲！ ​​​​"
text = re.sub(r'\[.*\]', "", text)
text = re.sub(r'\(.*\)', "", text)
text = re.sub(r'#.*#', "", text)
text = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', "", text)
text = re.findall('[\u4e00-\u9fa5a-zA-Z0-9，。《》“”：！？、]+', text, re.S)
text = "".join(text)
text_seg=' '.join(jieba.cut(text))
data=DataProcess.Dataset()
with open(data.w2idx_name, "r", encoding="utf8") as fp:
    w2idx=json.load(fp)
fp.close()

#with open(data.w2idx_en_name, "r", encoding="utf8") as fp:
#    w2idx_en=json.load(fp)
#fp.close()

idx2w={idx: word for word, idx in w2idx.items()}
#idx2w_en={idx: word for word, idx in w2idx_en.items()}

text_ids=[w2idx.get(item, w2idx["<UNK>"]) for item in text_seg.split()]

if len(text_ids) >= data.seqLength:
    xIds = text_ids[:data.seqLength]
else:
    xIds = text_ids + [w2idx["<EOS>"]] * (data.seqLength - len(text_ids))

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    loader=tf.train.import_meta_graph("./tmp/checkpoints/model.ckpt.meta")
    loader.restore(sess,tf.train.latest_checkpoint("./tmp/checkpoints"))
    input_data=graph.get_tensor_by_name('input_text:0')
    logits=graph.get_tensor_by_name('predictions:0')
    text_seqLen=graph.get_tensor_by_name("input_text_len:0")
    abs_seqLen=graph.get_tensor_by_name("input_abs_len:0")
    keep=graph.get_tensor_by_name("dropKeepprob:0")

    abs_logits=sess.run(logits,{input_data:[xIds]*batchSize,
                               text_seqLen:[len(text_ids)]*batchSize,
                               abs_seqLen:[len(xIds)]*batchSize,
                               keep:1.0})[0]

print("text:{}\n\n".format(text))
print("en:"+" ".join([idx2w[i] for i in abs_logits]))