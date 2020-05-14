import jieba
import os
import codecs,sys
import gensim
from string import punctuation
from gensim.models import word2vec
import Config
from Config import PreConfig

def splitLabelContent(Filename, Type="Train"):
    contentsName=PreConfig.Path+Type+"Contents_nonwordsplit.txt"
    labelsName=PreConfig.Path+Type+"Labels.txt"
    contents=codecs.open(contentsName,"w",encoding="utf8",errors="replace")
    labels=codecs.open(labelsName,"w",encoding="utf8",errors="replace")
    with open(Filename,"r",encoding='utf-8',errors='replace') as fp:
        for line in fp:
            try:
                label, content=line.strip().split('\t')
                if content:
                    contents.writelines(content+"\n")
                    labels.writelines(label+"\n")
            except:
                pass
    print(Type+"Labels and Contents have been split.")
    return contentsName, labelsName

def splitwords(Filename, Type="Train"):
    add_punc = "\t\r\n，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    all_punc=punctuation+add_punc
    line_num=1
    well=PreConfig.Path+Type+"Contentswell.txt"
    if not os.path.exists(well):
        fp=codecs.open(Filename,"r",encoding="utf8",errors="replace")
        target=codecs.open(well,"w",encoding="utf8",errors="replace")
        print(Type+":Reading File for Words Spliting...")
        line=fp.readline()
        while line:
            line_seg=" ".join(jieba.cut(line))
            testline=line_seg.split(' ')
            te2=[]
            for i in testline:
                if i not in all_punc:
                    te2.append(i)
            line_seg2=" ".join(jieba.cut(''.join(te2)))
            target.writelines(line_seg2+"\n")
            line_num+=1
            line=fp.readline()
        fp.close()
        target.close()

def word2vecModelPretrain(TrainTextName, ModelName=PreConfig.Word2VecModelPath):
    if not os.path.exists(ModelName):
        sentences=word2vec.Text8Corpus(TrainTextName)
        print("Start training wordvector...")
        model=word2vec.Word2Vec(sentences,size=200,sg=1,min_count=10)
        model.wv.save_word2vec_format(PreConfig.Word2VecModelPath,binary=True)
    Model=gensim.models.KeyedVectors.load_word2vec_format(ModelName,binary=True)
    print("Word2vec Model has been trained, loading...")
    return Model


ctname,ltname=splitLabelContent("./Data/cnews.train.txt",Type="Train")
csname,lsname=splitLabelContent("./Data/cnews.val.txt",Type="Test")
splitwords(ctname,Type="Train")
splitwords(csname,Type="Test")

model=word2vecModelPretrain(PreConfig.Path+"TrainContentswell.txt")
'''
fp=codecs.open(PreConfig.Path+"TrainContentswell.txt","r",encoding="utf8")
TrainContentsPath="./Data/"+"TrainContentswell.txt"
trainContents=[]
with open(TrainContentsPath, "r", encoding="utf-8", errors="ignore") as fp:
    for line in fp:
        content = list(line.split('\n'))
        trainContents.append(content)
fp.close()
print(trainContents[50001])
'''

