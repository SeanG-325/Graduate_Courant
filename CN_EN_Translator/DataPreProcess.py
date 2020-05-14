import json
import os
from collections import Counter
import nltk
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
        self.EmbeddingSize=512
        self.MODEL_CN = "./CN.model"
        self.MODEL_EN = "./EN.model"
        self.Filename_cn = "./cn_raw.dat"
        self.Filename_en = "./en_raw.dat"
        self.Filename="./Bi-Spoken.txt"
        self.w2idx_cn_name = "./w2idx_cn.json"
        self.w2idx_en_name = "./w2idx_en.json"
        self.cn_contents = []
        self.en_contents = []
        self.cn_contents_id = []
        self.en_contents_id = []
        self.cn_contents_id_a = []
        self.en_contents_id_a = []
        self.vocab = []
        self.embedding = []
        self.temp = ""
        self.words_cn=[]
        self.words_en=[]
        self.seqLength = 30
        self.Wmax=105
        self.en_vocablen=0
        self.cn_vocablen=0
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
        if not os.path.exists(self.Filename_cn) or not os.path.exists(self.Filename_en):
            print("Spliting file...")
            self.cn = codecs.open(self.Filename_cn, "w", encoding="utf8", errors="replace")
            self.en = codecs.open(self.Filename_en, "w", encoding="utf8", errors="replace")
            with open(self.Filename, "r", encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line.replace("\r\n", "")
                    line.replace("\n", "")
                    if i == 0:
                        line_seg = " ".join(jieba.cut(line))
                        self.cn_contents.append(self.stringQ2B(line_seg).strip().split())
                        self.cn.writelines(self.stringQ2B(line_seg))
                        i = 1
                    elif i == 1:
                        line=line.lower()
                        self.stringQ2B(line)
                        line = nltk.word_tokenize(line)
                        line.append("<EOS>")
                        self.en_contents.append(line)
                        for key in line:
                            temp1 += key + " "
                        self.en.writelines(self.stringQ2B(temp1 + "\n"))
                        temp1 = ""
                        i = 0
        else:
            print("loading...")
            with open(self.Filename_cn, "r", encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line.replace("\r\n", "")
                    line.replace("\n", "")
                    self.cn_contents.append(line.strip().split())
            with open(self.Filename_en, "r", encoding='utf-8', errors='replace') as fp:
                for line in fp:
                    line.lower()
                    line.replace("\r\n", "")
                    line.replace("\n", "")
                    line = line.split()
                    line.append("<EOS>")
                    line = ["<GO>"] + line
                    print(line)
                    self.en_contents.append(line)
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
        self.word2vecPretrain(self.Filename_cn, self.MODEL_CN)
        self.word2vecPretrain(self.Filename_en, self.MODEL_EN)
    def _getwordEmbedding_CN(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(self.MODEL_CN, binary=True)
        vocab = []
        wordEmbedding = []
        vocab.append("<PAD>")
        vocab.append("<UNK>")
        wordEmbedding.append(np.zeros(self.EmbeddingSize))
        wordEmbedding.append(np.random.randn(self.EmbeddingSize))

        for word in words:
            try:
                wordvector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(wordvector)
            except:
                pass
        return vocab, np.array(wordEmbedding)
    def _getwordEmbedding_EN(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(self.MODEL_EN, binary=True)
        vocab = []
        wordEmbedding = []
        vocab.append("<PAD>")
        vocab.append("<UNK>")
        vocab.append("<GO>")
        vocab.append("<EOS>")
        wordEmbedding.append(np.zeros(self.EmbeddingSize))
        wordEmbedding.append(np.random.randn(self.EmbeddingSize))
        wordEmbedding.append(np.random.randn(self.EmbeddingSize))
        wordEmbedding.append(np.ones(self.EmbeddingSize))
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
    def _getVocab(self, cn_contents, en_contents):
        allwords_cn = [word for content in cn_contents for word in content]
        allwords_en = [word for content in en_contents for word in content]
        wordCount_cn = Counter(allwords_cn)
        wordCount_en = Counter(allwords_en)
        sortWordCount_cn = sorted(wordCount_cn.items(), key=lambda x: x[1], reverse=True)
        sortWordCount_en = sorted(wordCount_en.items(), key=lambda x: x[1], reverse=True)
        words_cn = [item[0] for item in sortWordCount_cn if item[1] >= self.Wmax]
        words_en = [item[0] for item in sortWordCount_en if item[1] >= self.Wmax]
        self.words_cn = words_cn
        self.words_en = words_en
        vocab_cn, wordEmbedding_cn = self._getwordEmbedding_CN(words_cn)
        vocab_en, wordEmbedding_en = self._getwordEmbedding_EN(words_en)
        self.wordEmbedding_cn = wordEmbedding_cn
        self.wordEmbedding_en = wordEmbedding_en
        word2idx_cn = dict(zip(vocab_cn, list(range(len(vocab_cn)))))
        word2idx_en = dict(zip(vocab_en, list(range(len(vocab_en)))))
        self.word2idx_cn = word2idx_cn
        self.word2idx_en = word2idx_en
        with open(self.w2idx_cn_name, "w", encoding="utf8") as fp:
            json.dump(word2idx_cn, fp, ensure_ascii=False)
        fp.close()
        with open(self.w2idx_en_name, "w", encoding="utf8") as fp:
            json.dump(word2idx_en, fp, ensure_ascii=False)
        fp.close()
        self.cn_contents_id=[[word2idx_cn.get(item, word2idx_cn["<UNK>"]) for item in content] for content in cn_contents]
        self.en_contents_id=[[word2idx_en.get(item, word2idx_cn["<UNK>"]) for item in content] for content in en_contents]
        self.cn_contents_id_a=self.cn_contents_id
        self.en_contents_id_a=self.en_contents_id
        return word2idx_cn, word2idx_en
    def Datagen(self):
        self.PreProcess()
        self.w2vecPreTrain()
        self.cn_con=[]
        self.en_con=[]
        self.cn_con_len=[]
        self.en_con_len=[]
        cn_w2idx,en_w2idx=self._getVocab(cn_contents=self.cn_contents, en_contents=self.en_contents)
        self.en_vocablen=len(en_w2idx)
        self.cn_vocablen=len(cn_w2idx)
        for content in self.cn_contents_id_a:
            if len(content)>=self.seqLength:
                self.cn_con_len.append(self.seqLength)
            else:
                self.cn_con_len.append(len(content))
            if len(content)>self.seqLength:
                self.cn_con.append(content[:self.seqLength])
            else:
                self.cn_con.append(content+[cn_w2idx["<PAD>"]]*(self.seqLength-len(content)))
        for content in self.en_contents_id_a:
            self.en_con_len.append(self.seqLength)
            if len(content)>self.seqLength:
                self.en_con.append(content[:self.seqLength-1]+[en_w2idx["<EOS>"]])
            else:
                self.en_con.append(content+[en_w2idx["<EOS>"]]*(self.seqLength-len(content)))