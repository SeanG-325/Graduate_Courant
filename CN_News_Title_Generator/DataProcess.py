import json
import os
from collections import Counter
import re
import gensim
import string
from gensim.models import word2vec
import jieba
import numpy as np
import codecs

class Dataset(object):
    def __init__(self):
        self.wordEmbedding_cn = []
        self.wordEmbedding_en = []
        self.EmbeddingSize=256

        self.text_name="./dataset/train_text.txt"
        self.abs_name="./dataset/train_label.txt"
        self.Filename_text = "./text.dat"
        self.Filename_abs = "./abs.dat"
        self.text_contents = []
        self.abs_contents = []
        self.w2idx_name="./w2idx.json"

        self.Model="./model.model"
        self.minLen=30
        self.vocab = []
        self.embedding = []
        self.temp = ""
        self.seqLength = 75
        self.absmax=25
        self.Wmax=105



    def Q2B(self, uchar):
        """单个字符 全角转半角"""
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
            return uchar
        return chr(inside_code)
    def stringQ2B(self, ustring):
        return "".join([Dataset.Q2B(self,uchar) for uchar in ustring])



    def PreProcess(self):
        i=1
        temp1=""
        temp = str.maketrans(string.punctuation, " " * len(string.punctuation))
        add_punc = "\t\r\n，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
        if not os.path.exists(self.Filename_text):
            print("Spliting file...")
            self.text = codecs.open(self.Filename_text, "w", encoding="utf8", errors="replace")
            with open(self.text_name, "r", encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line.replace("\r\n", "")
                    line.replace("\n", "")
                    line = re.sub(r'\[.*\]', "", line)
                    line = re.sub(r'\(.*\)', "", line)
                    line = re.sub(r'#.*#', "", line)
                    line = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', "", line)
                    line = re.sub(r'[a-z]*[:.]+\S+', "", line)
                    line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9，。《》“”：！？、]+', line, re.S)
                    line = "".join(line)
                    line_seg = " ".join(jieba.cut(line))
                    #self.text_contents.append(self.stringQ2B(line_seg).strip().split())
                    self.text.writelines(self.stringQ2B(line_seg)+"\n")
                    
            fp.close()
        if not os.path.exists(self.Filename_abs):
            print("Spliting file...")
            self.abs = codecs.open(self.Filename_abs, "w", encoding="utf8", errors="replace")
            with open(self.abs_name, "r", encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line.replace("\r\n", "")
                    line.replace("\n", "")
                    line = re.sub(r'\[.*\]', "", line)
                    line = re.sub(r'\(.*\)', "", line)
                    line = re.sub(r'#.*#', "", line)
                    line = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', "", line)
                    line = re.sub(r'[a-z]*[:.]+\S+', "", line)
                    line = re.findall('[\u4e00-\u9fa5a-zA-Z0-9，。《》“”：！？、]+', line, re.S)
                    line = "".join(line)
                    line_seg = " ".join(jieba.cut(line))
                    #self.abs_contents.append(self.stringQ2B(line_seg).strip().split())
                    self.abs.writelines(self.stringQ2B(line_seg).strip()+" <EOS>\n")
            fp.close()
        print("loading...")
        with open(self.Filename_text, "r", encoding='utf-8', errors='replace') as fp:
            with open(self.Filename_abs, "r", encoding='utf-8', errors='replace') as fp2:
                for line in fp:
                    try:
                        line2=fp2.readline()
                        temp1=line.strip().split()
                        temp2=line2.strip().split()
                        if len(temp1) >= 20 and len(temp2) > 0:
                            self.text_contents.append(temp1)
                            self.abs_contents.append(temp2)
                    except:
                        break
            



    def word2vecPretrain(self, TrainTextName, ModelName):
        if not os.path.exists(ModelName):
            sentences = word2vec.Text8Corpus(TrainTextName)
            print("Start training wordvector...")
            model = word2vec.Word2Vec(sentences, size=self.EmbeddingSize, sg=1, min_count=5)
            model.wv.save_word2vec_format(ModelName, binary=True)
        Model = gensim.models.KeyedVectors.load_word2vec_format(ModelName, binary=True)
        print("Word2vec Model has been trained, loading...")
        return Model
    def w2vecPreTrain(self):
        self.word2vecPretrain(self.Filename_text, self.Model)

    def _getwordEmbedding(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(self.Model, binary=True)
        vocab = []
        wordEmbedding = []
        vocab.append("<EOS>")
        vocab.append("<UNK>")
        vocab.append("<GO>")
        wordEmbedding.append(np.zeros(self.EmbeddingSize))
        wordEmbedding.append(np.random.randn(self.EmbeddingSize))
        wordEmbedding.append(np.random.randn(self.EmbeddingSize))
        for word in words:
            try:
                if(word!="<EOS>"):
                    wordvector = wordVec.wv[word]
                    vocab.append(word)
                    wordEmbedding.append(wordvector)
            except:
                #print("No such word in dict.\n")
                pass
        return vocab, np.array(wordEmbedding)

    def _getVocab(self, text_contents, abs_contents):
        allwords_text = [word for content in text_contents for word in content]
        #allwords_abs = [word for content in abs_contents for word in content]
        wordCount_text = Counter(allwords_text)
        #wordCount_en = Counter(allwords_abs)
        sortWordCount_text = sorted(wordCount_text.items(), key=lambda x: x[1], reverse=True)
        #sortWordCount_abs = sorted(wordCount_abs.items(), key=lambda x: x[1], reverse=True)
        words_text = [item[0] for item in sortWordCount_text if item[1] >= self.Wmax]
        #words_en = [item[0] for item in sortWordCount_abs if item[1] >= self.Wmax]
        self.words_text = words_text
        #self.words_abs = words_abs
        vocab_text, wordEmbedding_text = self._getwordEmbedding(words_text)
        #vocab_abs, wordEmbedding_abs = self._getwordEmbedding_EN(words_abs)
        self.wordEmbedding_text = wordEmbedding_text
        #self.wordEmbedding_abs = wordEmbedding_abs
        word2idx_text = dict(zip(vocab_text, list(range(len(vocab_text)))))
        #word2idx_abs = dict(zip(vocab_abs, list(range(len(vocab_abs)))))
        self.word2idx_text = word2idx_text
        #self.word2idx_abs = word2idx_abs
        with open(self.w2idx_name, "w", encoding="utf8") as fp:
            json.dump(word2idx_text, fp, ensure_ascii=False)
        fp.close()

        #with open(self.w2idx_name, "w", encoding="utf8") as fp:
        #    json.dump(word2idx_abs, fp, ensure_ascii=False)
        #fp.close()
        self.text_contents_id=[[word2idx_text.get(item, word2idx_text["<UNK>"]) for item in content] for content in text_contents]
        self.abs_contents_id=[[word2idx_text.get(item, word2idx_text["<UNK>"]) for item in content] for content in abs_contents]
        #self.text_contents_id_a=self.text_contents_id
        #self.abs_contents_id_a=self.abs_contents_id
        return word2idx_text #, word2idx_en
    def Datagen(self):
        self.PreProcess()
        self.w2vecPreTrain()
        self.text_con=[]
        self.abs_con=[]
        self.text_con_len=[]
        self.abs_con_len=[]
        text_w2idx=self._getVocab(text_contents=self.text_contents,abs_contents=self.abs_contents)
        self.vocablen=len(text_w2idx)
        for content in self.text_contents_id:
            if len(content)>=self.seqLength:
                self.text_con_len.append(self.seqLength)
            else:
                self.text_con_len.append(len(content))
            if len(content)>self.seqLength:
                self.text_con.append(content[:self.seqLength])
            else:
                self.text_con.append(content+[text_w2idx["<EOS>"]]*(self.seqLength-len(content)))
        for content1 in self.abs_contents_id:
            self.abs_con_len.append(self.absmax)
            if len(content1)>self.absmax:
                self.abs_con.append(content1[:self.absmax-1]+[self.word2idx_text["<EOS>"]])
            else:
                self.abs_con.append(content1+[self.word2idx_text["<EOS>"]]*(self.absmax-len(content1)))
        self.maxLen_Text=max(self.text_con_len)
        self.maxLen_Abs=max(self.abs_con_len)
        #self.seqLength=self.maxLen_Text

#data=Dataset()
#data.Datagen()
'''
self=Dataset()
abs = codecs.open(self.Filename_abs, "w", encoding="utf8", errors="replace")
with open(self.abs_name, "r", encoding='utf-8', errors='replace') as fp:
    for line in fp:
        line.replace("\r\n", "")
        line.replace("\n", "")
        line_seg = " ".join(jieba.cut(line))
        self.abs_contents.append(self.stringQ2B(line_seg).strip().split())
        abs.writelines(self.stringQ2B(line_seg).strip() + " <EOS>\n")
'''
