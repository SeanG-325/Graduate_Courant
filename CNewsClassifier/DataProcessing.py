import json
from collections import Counter
import gensim

from Config import config
import numpy as np

class Dataset(object):
    def __init__(self, config):
        self.config=config
        self._dataSource=config.dataSource
        self._embeddingSize=config.model.embeddingSize
        self._batchSize=config.batchSize
        self._rate=config.rate
        self._sequenceLenghth=config.sequenceLength

        self._stopWordSource=config.stopWordScource

        self.stopWordDict={}

        self.trainContents=[]
        self.trainLabels=[]

        self.evalContents=[]
        self.evalLabels=[]

        self.wordEmbedding=[]

        self.labelList=[]

    def _readData(self, DataSource):
        TrainContentsPath=DataSource+"TrainContentswell.txt"
        TestContentsPath=DataSource+"TestContentswell.txt"
        TrainLabelsPath=DataSource+"TrainLabels.txt"
        TestLabelsPath=DataSource+"TestLabels.txt"
        trainContents=[]
        trainLabels=[]
        testContents=[]
        testLabels=[]
        with open(TrainContentsPath,"r",encoding="utf-8",errors="ignore") as fp:
            for line in fp:
                content=line.strip().split()
                trainContents.append(content)
        fp.close()
        with open(TestContentsPath,"r",encoding="utf-8",errors="ignore") as fp:
            for line in fp:
                content=line.strip().split()
                testContents.append(content)
        fp.close()
        with open(TrainLabelsPath,"r",encoding="utf-8",errors="ignore") as fp:
            for line in fp:
                label=line.split('\n')
                trainLabels.append(label)
        fp.close()
        with open(TestLabelsPath,"r",encoding="utf-8",errors="ignore") as fp:
            for line in fp:
                label=line.split('\n')
                testLabels.append(label)
        fp.close()
        return trainContents,trainLabels,testContents,testLabels


    def _labelToIndex(self, labels, label2idx):
        labelIds=[label2idx[label[0]] for label in labels]
        return labelIds

    def _contentsToIndex(self, contents, word2idx):
        contentIds=[[word2idx.get(item, word2idx["UNK"]) for item in content] for content in contents]
        return contentIds

    def _genTrainEvalData(self, x, y, word2idx, rate, split=0):
        if split==0:
            contents = []
            for content in x:
                if len(content) >= self._sequenceLenghth:
                    contents.append(content[:self._sequenceLenghth])
                else:
                    contents.append(content + [word2idx["PAD"]] * (self._sequenceLenghth - len(content)))

            trainIndex = int(len(x) * rate)

            trainContents = np.asarray(contents[:trainIndex], dtype="int64")
            trainLabels = np.array(y[:trainIndex], dtype="int64")

            evalContents = np.asarray(contents[trainIndex:], dtype="int64")
            evalLabels = np.array(y[trainIndex:], dtype="int64")

            return trainContents, trainLabels, evalContents, evalLabels

        else:
            contents = []
            for content in x:
                if len(content) >= self._sequenceLenghth:
                    contents.append(list(content[:self._sequenceLenghth]))
                else:
                    contents.append(content + [word2idx["PAD"]] * (self._sequenceLenghth - len(content)))
            return contents,y


    def _getVocab(self, contents, labels):
        allWords=[word for content in contents for word in content]

        subWords=[word for word in allWords if word not in self.stopWordDict]

        wordCount=Counter(subWords)

        sortWordCount=sorted(wordCount.items(),key=lambda x: x[1], reverse=True)

        words=[item[0] for item in sortWordCount if item[1]>=5]

        vocab, wordEmbedding=self._getwordEmbedding(words)
        self.wordEmbedding=wordEmbedding

        word2idx=dict(zip(vocab, list(range(len(vocab)))))

        uniqueLabel=config.preconfig.Labels
        label2idx=dict(zip(uniqueLabel,list(range(len(uniqueLabel)))))
        self.labelList=list(range(len(uniqueLabel)))

        with open(config.preconfig.word2idxPath,"w",encoding="utf8") as fp:
            json.dump(word2idx,fp,ensure_ascii=False)
        fp.close()

        with open(config.preconfig.label2idxPath,"w",encoding="utf8") as fp:
            json.dump(label2idx,fp,ensure_ascii=False)
        fp.close()

        return word2idx, label2idx

    def _getwordEmbedding(self, words):
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(config.preconfig.Word2VecModelPath,binary=True)
        vocab=[]
        wordEmbedding=[]

        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                wordvector=wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(wordvector)
            except:
                # print("No such word in dict.\n")
                pass

        return vocab, np.array(wordEmbedding)

    def _readStopWords(self, stopWordsFilename):
        with open(stopWordsFilename,"r",encoding="UTF-8") as fp:
            stopWordsList=fp.read().splitlines()
            self.stopWordDict=dict(zip(stopWordsList,list(range((len(stopWordsList))))))

    def dataGen(self):
        self._readStopWords(self._stopWordSource)

        trainContents, trainLabels, evalContents, evalLabels=self._readData(config.preconfig.Path)

        word2idx, label2idx=self._getVocab(trainContents,trainLabels)
        # print(word2idx)
        trainLabelsIds=self._labelToIndex(trainLabels,label2idx)
        # print(len(trainContents))
        trainContentsIds=self._contentsToIndex(trainContents, word2idx)
        #print(trainContentsIds)
        evalContentsIds=self._contentsToIndex(evalContents,word2idx)
        evalLabelsIds=self._labelToIndex(evalLabels,label2idx)


        trainContents1, trainLabels1 = self._genTrainEvalData(trainContentsIds, trainLabelsIds, word2idx,self._rate, split=1)
        evalContents1, evalLabels1 = self._genTrainEvalData(evalContentsIds, evalLabelsIds, word2idx,self._rate, split=1)

        self.trainContents = trainContents1
        self.trainLabels = trainLabels1

        self.evalContents = evalContents1

        self.evalLabels = evalLabels1


#data=Dataset(config)
#print("Data Processing...")
#data.dataGen()
#print("Success.")
