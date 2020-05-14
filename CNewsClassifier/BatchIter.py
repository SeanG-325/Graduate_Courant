from math import floor

import numpy as np
from random import shuffle

from Config import config


def nextBatch(x,y,batchSize):
    #print(x)
    z=list(zip(x,y))
    shuffle(z)
    x[:],y[:]=zip(*z)
    '''
    perm=np.arange(len(x))
    np.random.shuffle(perm)
    x=x[perm]
    y=y[perm]
    '''
    numBatches=floor(len(x)/config.batchSize)
    print(numBatches)



    for i in range(numBatches):
        start=i*batchSize
        end=start+batchSize
        batchX=np.array(x[start:end],dtype="int64")
        batchY=np.array(y[start:end],dtype="float32")

        yield batchX, batchY

