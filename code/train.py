import tensorflow as tf
import numpy as np
import pandas as pd

import dataset as ds
import efficientSiameseNet

from skimage import io
import sys
import cv2
import json
import math
import multiprocessing

trainConfigPath = sys.argv[1]

with open(trainConfigPath, 'r') as json_file:
    trainConfig = json.load(json_file)

print("Train config:")
print(trainConfig)

catalogPath = trainConfig["catalogPath"]
petType = trainConfig["petType"]
extractedImagesPath = trainConfig["extractedImagesPath"]
minImagesPerCardForSimilarity = trainConfig["minImagesPerCardForSimilarity"]
seed = 344567
imageSize = 224
seqLength = 8
l2regAlpha = 0.0
doRate = 0.0
batchSize = 4
prefetchQueueLength = multiprocessing.cpu_count()
freezeBackbone = True

tf.random.set_seed(seed+667734)

catalog = pd.read_csv(catalogPath)
print("{0} images are available".format(len(catalog)))

petSpecificCatalog = catalog.loc[catalog.loc[:,'pet'] == petType,:]
print("{0} images of {1} pet type".format(len(petSpecificCatalog),petType))

ds1 = ds.SimilaritySet(petSpecificCatalog,extractedImagesPath, 123, minImagesPerCardForSimilarity=minImagesPerCardForSimilarity)

trSamplesInOneEpochs = ds1.getSimilarityPetsCount()

def loadImage(imagePath):
    #print("loadImage: imagePath is {0}".format(imagePath))
    if imagePath.endswith(".png"):
        original_image = io.imread(imagePath,plugin='imread')
    else:
        original_image = io.imread(imagePath)

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

def coerceSeqSizeInTuple(anchorPack, positivePack, negativePack):
    return (coerceSeqSizeTF(anchorPack, seqLength), coerceSeqSizeTF(positivePack, seqLength), coerceSeqSizeTF(negativePack, seqLength))

def loadImagePackNp(pathsPack):
        paths = np.split(pathsPack, pathsPack.shape[0], axis=0)
        # print("Paths:")
        # print(paths)
        # exit(1)
        return np.stack([cv2.resize(loadImage(x[0].decode(encoding='UTF-8')), (imageSize, imageSize)) for x in paths], axis=0)

@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def loadImagePackTF(pathTensor):
  y = tf.numpy_function(loadImagePackNp, [pathTensor], (tf.uint8))
  return y         

def trainGen():
    for sample in ds1.getSamples(cycled=True):
        yield sample

def augment(imagePack):
    def augmentSingle(image):
        augSwitches = tf.cast(tf.math.round(tf.random.uniform([3],minval=0.0, maxval=1.0)),dtype=tf.bool)
        image = tf.cond(augSwitches[0], lambda: tf.image.rot90(image), lambda: image)
        image = tf.cond(augSwitches[1], lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(augSwitches[2], lambda: tf.image.flip_up_down(image), lambda:image)
        return image
    return tf.map_fn(augmentSingle, imagePack)

def augmentTriple(anchorPack, positivePack, negativePack):
        return augment(anchorPack),augment(positivePack),augment(negativePack)

def loadImages(anchorPaths,positivePaths,negativePaths):
    anchorPack = loadImagePackTF(anchorPaths)
    positivePack = loadImagePackTF(positivePaths)
    negativePack = loadImagePackTF(negativePaths)
    return (anchorPack, positivePack, negativePack)

zerosDs = tf.data.Dataset.range(1).repeat()

trImagePathsDataset = tf.data.Dataset.from_generator(
     trainGen,
     (tf.string, tf.string, tf.string),
     (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))

trainDataset = trImagePathsDataset \
    .map(loadImages, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True) \
    .map(coerceSeqSizeInTuple, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True) \
    .map(augmentTriple, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True) \
    .shuffle(128,seed=seed+23)    

dummySupervisedTrainDataset = tf.data.Dataset.zip((trainDataset, zerosDs))

dummySupervisedBatchedTrainDataset = dummySupervisedTrainDataset \
    .batch(batchSize) \
    .prefetch(prefetchQueueLength)

model, backbone, featureExtractor = efficientSiameseNet.constructSiameseTripletModel(seqLength, l2regAlpha, doRate, imageSize)
print("model constructed")

if freezeBackbone:
    print("Backbone is FROZEN")
    backbone.trainable = False
else:
    print("Backbone is TRAINABLE")
    backbone.trainable = True

model.compile(
          #optimizer=tf.keras.optimizers.SGD(momentum=.5,nesterov=True, clipnorm=1.),
          optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4, clipnorm=1.),
          #optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
          #loss=loss,
          #metrics=[QuadraticWeightedKappa(input_format='scalar'), tf.keras.metrics.MeanAbsoluteError(name="mae") ] # tf.keras.metrics.MeanAbsoluteError(name="mae")
          )
print("model compiled")
print(model.summary())

print("train dataset")
print(dummySupervisedBatchedTrainDataset)

model.fit(x = dummySupervisedBatchedTrainDataset, \
      #validation_data = valDs,
      #validation_steps = int(math.floor(vaSamplesCount / batchSize)),
      #initial_epoch=initial_epoch,
      verbose = 1,
      #callbacks=callbacks,
      shuffle=False, # dataset is shuffled explicilty
      steps_per_epoch= int(math.ceil(trSamplesInOneEpochs / batchSize)),
      epochs=10)

print("Done")

# idx1 = 0
# for sample in trainDataset.as_numpy_iterator():
#     sim1,sim2,alt = sample
#     print("run 1:\t{0}:\t{1}\t{2}\t{3}]".format(idx1,sim1.shape,sim2.shape,alt.shape))
#     idx1 += 1    