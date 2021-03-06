import tensorflow as tf
import kafkajobs
import efficientSiameseNet
from skimage import io
import cv2

import numpy as np
import os

import gc

kafkaUrl = os.environ['KAFKA_URL']
inputQueueName = os.environ['INPUT_QUEUE']
outputQueueName = os.environ['OUTPUT_QUEUE']
animalType = os.environ["ANIMAL_TYPE"]

appName = "FeatureVectorExtractor-experiments-3-4-{0}".format(animalType)

seqLength = 8
l2regAlpha=0.0
doRate=0.0
imageSize=224

worker = kafkajobs.jobqueue.JobQueueWorker(appName, kafkaBootstrapUrl=kafkaUrl, topicName=inputQueueName, appName=appName)
resultQueue = kafkajobs.jobqueue.JobQueueProducer(kafkaUrl, outputQueueName, appName)

if animalType == "cat":
    modelWeightsFile = "featureExtractor_3_3.hdf5"
elif animalType == "dog":
    modelWeightsFile = "featureExtractor_4_3.hdf5"
else:
    raise "Unsupported pet type: {0}".format(animalType)

def log(message):
    print(message)

def coerceDimntionality(original_image):
    imShape = original_image.shape
    #print("imhsape is {0}".format(imShape))
    if len(imShape) == 2: # greyscale
        original_image = np.copy(np.tile(np.expand_dims(original_image, axis=2),(1,1,3)))
    else:    
        if imShape[2] == 4: #RGBA
            original_image = np.copy(original_image[:,:,0:3])
    return original_image

def coerceSeqSizeTF(imagePack, trainSequenceLength):
    imagePackShape = tf.shape(imagePack)
    outputShape = [
        trainSequenceLength,
        imagePackShape[1],
        imagePackShape[2],
        imagePackShape[3]
    ]
    T = imagePackShape[0]

    availableIndices = tf.range(T)


    # if T is less than trainSequenceLength we need to duplicate the layers
    seqRepCount = tf.cast(tf.math.ceil(trainSequenceLength / T), tf.int32)
    notTooShortIndicies = \
    tf.cond(seqRepCount > 1, \
        lambda : tf.tile(availableIndices, [seqRepCount]), \
        lambda : availableIndices)

    # if T is greater than trainSequenceLength we need to truncate it
    notTooLongIndices = notTooShortIndicies[0:trainSequenceLength]
    #notTooLong = tf.IndexedSlices(imagePack,notTooLongIndices, dense_shape = outputShape)
    notTooLong = tf.gather(imagePack, notTooLongIndices, axis=0)
    shapeSet = tf.reshape(notTooLong,outputShape)
    return shapeSet

def augment(imagePack):
    def augmentSingle(image):
        augSwitches = tf.cast(tf.math.round(tf.random.uniform([3],minval=0.0, maxval=1.0)),dtype=tf.bool)
        image = tf.cond(augSwitches[0], lambda: tf.image.rot90(image), lambda: image)
        image = tf.cond(augSwitches[1], lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(augSwitches[2], lambda: tf.image.flip_up_down(image), lambda:image)
        return image
    return tf.map_fn(augmentSingle, imagePack)

def loadImagePackNp(imagesNp):
    return np.reshape(np.stack([cv2.resize(coerceDimntionality(x), (imageSize, imageSize)) for x in imagesNp], axis=0),(len(imagesNp),imageSize,imageSize,3))


def work():
    log("Service started. Pooling for a job")
    featureExtractor = None
    while True:        
        job = worker.TryGetNextJob(5000)
        if job is None:
            # unloading model does not free the memory. Thus we do not unload for now

            # if not(featureExtractor is None):
            #     # unloading the model to free the memory
            #     model, backbone, featureExtractor = None, None, None
            #     tf.keras.backend.clear_session()
            #     gc.collect()
            #     log("Model unloaded to free the memory")
            continue
        else:
            #print("Got job {0}".format(job))
            uid = job["uid"]
            log("{0}: Starting to process the job".format(uid))
            images = job['detected_pet_images']
            if (job["animal"] == animalType) and len(images)>0:
                log("{0}: Extracting {1} images".format(uid, len(images)))
                
                imagesNp = kafkajobs.serialization.imagesFieldToNp(images)

                log("{0}: Extracted {1} images".format(uid, len(imagesNp)))

                if featureExtractor is None:
                    print("(Re)constructing model")
                    model, backbone, featureExtractor = efficientSiameseNet.constructSiameseTripletModel(seqLength, l2regAlpha, doRate, imageSize)
                    backbone.trainable = False
                    model = None
                    backbone = None
                    print("Loading model weights {0}".format(modelWeightsFile))
                    featureExtractor.load_weights(modelWeightsFile)

                resizedPack = loadImagePackNp(imagesNp)
                log("{0}: images are resized and packed".format(uid))
                inputData = tf.reshape(augment(coerceSeqSizeTF(resizedPack, seqLength)),[1,seqLength, imageSize,imageSize,3])
                featureVector = featureExtractor.predict(inputData)
                log("{0}: Got feature vector {1}".format(uid, featureVector))
                job["exp_3_4_features"] = kafkajobs.serialization.npArrayToBase64str(featureVector)
                
                resultQueue.Enqueue(uid, job)
                log("{0}: Posted result in output queue".format(uid))
            elif len(images) == 0:
                log("{0}: Skipping as the card contains 0 images".format(uid))
            else:
                log("{0}: Skipping as pet type {1} is not for current model {2}".format(uid, job["animal"], animalType))
            worker.Commit()
            log("{0}: Commited".format(uid))
    

if __name__ == '__main__':
    try:
        work()
    except SystemExit:
        pass

