import json
from string import punctuation
import tensorflow as tf
from Config import config
import jieba
from Transformer import fixedPositionEmbedding
embeddingPosition=fixedPositionEmbedding(1,config.sequenceLength)
cat=['Sports', 'Finance', 'Real estate', 'decoration', 	'Education', 'Technology', 'fashion', 'politics', 'games', 'entertainment']
x="近一个世纪以来，以黄柳霜、李小龙、成龙、刘玉玲为代表的一批华人演员闯荡好莱坞，逐渐在好莱坞电影中占据一席之地。美国何振岱文化研究基金会创始人何欣晏向记者表示：“希望通过呈现好莱坞华人在电影历史上的奋斗历程，展现华人在海外奋斗、拼搏的精神，并把他们对社会的回馈和爱分享给大家。”"
#x="而在电影开幕表演开始之前，明星艺人们的礼服造型相继曝光了，一众女明星同台比美让人看得眼花缭乱。首先来看看电影当家花旦周冬雨，裸色吊带纱裙腰间搭配一条亮眼的橙色腰带，裸背的设计尤为吸睛。之前因为拍摄《少年的你》头发剪成寸头如今已经长了不少不再戴帽子了，将头发全部挽在后面，妆容淡雅，“缥缈影众人，疑似海上花”的感觉清新脱俗。"
#x="作为本届时装周的重要版块之一——2019北京时装周光华新闻大奖更加注重与合作媒体的深度合作与价值共生，并增设合作媒体奖项。与此同时，今年组委会不仅将目光聚焦在电视、期刊等传统媒体上，还将评选范围扩大到影响力日益剧增的自媒体、以及拥有时尚话语权的KOL等，更与合作媒体联合颁发品牌创意奖，进一步彰显了北京时装周开放、包容、合作的核心发展理念。"
#x="自香港修风波发生以来，香港警察一直守护家园，却被乱港分子造谣攻击。MV呈现了香港警察们真实而不为人知的一面：不知疲惫地在深夜执法、不顾砖块袭击将被暴徒围攻的市民救出、面对暴徒的挑衅攻击却始终保持理性克制、不少警察身上早已伤痕累累却始终坚持在一线止暴制乱等。在一帧帧香港警察无惧暴力、守护香港的画面中，乱港分子的谣言不攻自破。"
x_seg=' '.join(jieba.cut(x))
add_punc = "\t\r\n，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
all_punc = punctuation + add_punc
testline=x_seg.split(' ')
te2=[]
for i in testline:
    if i not in all_punc:
        te2.append(i)
x_seg2=' '.join(jieba.cut(''.join(te2)))
with open(config.preconfig.word2idxPath, "r", encoding="utf-8") as f:
    word2idx = json.load(f)

with open(config.preconfig.label2idxPath, "r", encoding="utf-8") as f:
    label2idx = json.load(f)

idx2label = {value: key for key, value in label2idx.items()}

xIds = [word2idx.get(item, word2idx["UNK"]) for item in x_seg2.split(" ")]
if len(xIds) >= config.sequenceLength:
    xIds = xIds[:config.sequenceLength]
else:
    xIds = xIds + [word2idx["PAD"]] * (config.sequenceLength - len(xIds))

graph = tf.Graph()
with graph.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint("./model/Transformer/")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 获得需要喂给模型的参数，输出的结果依赖的输入值
        inputX = graph.get_operation_by_name("inputX").outputs[0]
        dropoutKeepProb = graph.get_operation_by_name("dropKeepProb").outputs[0]
        embeddedPosition = graph.get_operation_by_name("embeddedPosition").outputs[0]

        # 获得输出的结果
        predictions = graph.get_tensor_by_name("output/predictions:0")

        pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: 1.0, embeddedPosition: embeddingPosition})[0]

# pred = [idx2label[item] for item in pred]
print(idx2label[pred])#Chinese
print(cat[pred])#English