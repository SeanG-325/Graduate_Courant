import datetime
import os
from BatchIter import nextBatch
import tensorflow as tf
from DataProcessing import Dataset
from Metrics import get_multi_metrics, get_binary_metrics, mean
from Transformer import Transformer
from Config import config
from Transformer import fixedPositionEmbedding

data=Dataset(config)
print("Data Processing...")
data.dataGen()
print("Success.")
trainContents = data.trainContents

trainLabels= data.trainLabels
evalContents=data.evalContents
evalLabels=data.evalLabels
wordEmbedding=data.wordEmbedding
labelList=data.labelList

embeddingPosition=fixedPositionEmbedding(config.batchSize,config.sequenceLength)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # GPU config

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        transformer=Transformer(config,wordEmbedding)
        globalStep=tf.Variable(0,name="globalStep",trainable=False)

        optimizer=tf.train.AdamOptimizer(config.training.LearningRate)
        gradsAndVars=optimizer.compute_gradients(transformer.loss)
        trainOp=optimizer.apply_gradients(gradsAndVars,globalStep)

        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", transformer.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)

        savedModelPath = config.preconfig.modelPath_Transformer
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())

        def trainStep(batchX,batchY):
            feed_dict={
                transformer.inputX: batchX,
                transformer.inputY: batchY,
                transformer.dropKeepProb: config.model.dropkeepProb,
                transformer.embeddedPosition:embeddingPosition
            }
            _, summary, step, loss, predictions=sess.run([trainOp,summaryOp,globalStep,transformer.loss,transformer.predictions],feed_dict=feed_dict)

            timeStr=datetime.datetime.now().isoformat()

            acc,recall,prec,f_beta=get_multi_metrics(pred_y=predictions,true_y=batchY,labels=labelList)

            trainSummaryWriter.add_summary(summary,step)

            return loss,acc,prec,recall,f_beta

        def devStep(batchX,batchY):
            feed_dict={
                transformer.inputX:batchX,
                transformer.inputY:batchY,
                transformer.dropKeepProb:1.0,
                transformer.embeddedPosition:embeddingPosition
            }

            summary,step,loss,predictions=sess.run([summaryOp,globalStep,transformer.loss,transformer.predictions],feed_dict=feed_dict)

            acc,precision,recall,f_beta=get_multi_metrics(pred_y=predictions,true_y=batchY,labels=labelList)

            evalSummaryWriter.add_summary(summary,step)

            return loss,acc,precision,recall,f_beta

        for i in range(config.training.epoches):
            print("Start training...")
            for batchTrain in nextBatch(trainContents,trainLabels,config.batchSize):
                loss,acc,prec,recall,f_beta=trainStep(batchTrain[0],batchTrain[1])

                currentStep=tf.train.global_step(sess, globalStep)
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(currentStep,loss,acc,recall,prec,f_beta))
                if currentStep%config.training.evaluateEvery_Transformer==0:
                    print("\nEvaluation:")

                    losses=[]
                    accs=[]
                    f_betas=[]
                    precs=[]
                    recalls=[]

                    for batchEval in nextBatch(evalContents,evalLabels,config.batchSize):
                        loss,acc,precision,recall,f_beta=devStep(batchEval[0],batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        precs.append(precision)
                        recalls.append(recall)
                        f_betas.append(f_beta)

                    time_str=datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(precs),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))
        saver.save(sess, savedModelPath)






