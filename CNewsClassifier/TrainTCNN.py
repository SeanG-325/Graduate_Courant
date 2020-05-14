import datetime
import os
from BatchIter import nextBatch
import tensorflow as tf
from DataProcessing import Dataset
from Metrics import get_multi_metrics, get_binary_metrics, mean
from TextCNN import TextCNN
from Config import config


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


with tf.Graph().as_default():
    session_conf=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.7 # GPU config

    sess=tf.Session(config=session_conf)

    with sess.as_default():
        cnn=TextCNN(config,wordEmbedding)
        globalStep=tf.Variable(0,name="globalStep",trainable=False)
        optimizer=tf.train.AdamOptimizer(config.training.LearningRate)
        gradsAndVars=optimizer.compute_gradients(cnn.loss)
        trainOp=optimizer.apply_gradients(gradsAndVars,global_step=globalStep)

        gradSummaries=[]
        for g,v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir=os.path.abspath(os.path.join(os.path.curdir,"summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss",cnn.loss)
        summaryOp=tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        saver=tf.train.Saver(tf.global_variables(),max_to_keep=10)

        savedModelPath = config.preconfig.modelPath_TCNN
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        #builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        def trainStep(batchX,batchY):
            feed_dict={
                cnn.inputX:batchX,
                cnn.inputY:batchY,
                cnn.dropoutKeepProb:config.model.dropkeepProb
            }
            _,summary,step,loss,predictions=sess.run(
                [trainOp,summaryOp,globalStep,cnn.loss,cnn.prediction],
                feed_dict=feed_dict
            )

            acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                          labels=labelList)

            trainSummaryWriter.add_summary(summary,step)

            return loss,acc,prec,recall,f_beta

        def devStep(batchX,batchY):
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.prediction],
                feed_dict=feed_dict)

            acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, precision, recall, f_beta

        k=0
        for i in range(config.training.epoches):
            print("Training model in epoch {}\n".format(i))
            for batchTrain in nextBatch(trainContents,trainLabels,config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])
                currentStep = tf.train.global_step(sess, globalStep)
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery_TCNN == 0:
                     print("\nEvaluation:")

                     losses = []
                     accs = []
                     f_betas = []
                     precisions = []
                     recalls = []

                     for batchEval in nextBatch(evalContents, evalLabels, config.batchSize):
                         loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                         losses.append(loss)
                         accs.append(acc)
                         f_betas.append(f_beta)
                         precisions.append(precision)
                         recalls.append(recall)

                     time_str = datetime.datetime.now().isoformat()
                     print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean(accs),
                                                                                                         mean(precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))
        saver.save(sess, savedModelPath)
'''
                if currentStep % config.training.checkPointEvery == 0:
                     # 保存模型的另一种方法，保存checkpoint文件
                     path = saver.save(sess, "./model/textCNN/model/my-model", global_step=currentStep)
                     print("Saved model checkpoint to {}\n".format(path))

                inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                          "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

                outputs = {"predictions": tf.saved_model.utils.build_tensor_info(cnn.prediction)}

                prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                                              outputs=outputs,
                                                                                              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")

                if k==0:
                    k+=1
                    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                                         signature_def_map={"predict": prediction_signature},
                                                         legacy_init_op=legacy_init_op)
                else:
                    builder.add_meta_graph(tags=[tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={"predict": prediction_signature})
                builder.save()
'''

