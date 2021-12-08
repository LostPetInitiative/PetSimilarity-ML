from numpy.lib.function_base import delete
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
import os
import random
import multiprocessing

trainConfigPath = sys.argv[1]
trainRunConfigPath = sys.argv[2]
outputPath = sys.argv[3]
if len(sys.argv) > 4:
    checkpointPath = os.path.join(sys.argv[4],"weights.hdf5")
else:
    checkpointPath = "nonexistant"

with open(trainConfigPath, 'r') as json_file:
    trainConfig = json.load(json_file)

print("Train config:")
print(trainConfig)

with open(trainRunConfigPath, 'r') as json_file:
    trainRunConfig = json.load(json_file)

print("Train run config:")
print(trainRunConfig)

catalogPath = trainConfig["catalogPath"]
petType = trainConfig["petType"]
extractedImagesPath = trainConfig["extractedImagesPath"]
validationFoldNum = trainConfig["validationFoldNum"]
minImagesPerCardForSimilarity = trainRunConfig["minImagesPerCardForSimilarity"]
monitoredMetric = trainRunConfig["monitoredMetric"]
monitoredMode = trainRunConfig["monitoredMode"]
reduceLrPatience = trainRunConfig["reduceLrPatience"]
minAllowedLR = trainRunConfig["minAllowedLR"]
earlyStoppingPatience = trainRunConfig["earlyStoppingPatience"]
minMetricDelta = trainRunConfig["minMetricDelta"]    

checkpointBackboneFrozen = bool(trainRunConfig["checkpointBackboneFrozen"])
freezeBackbone = bool(trainRunConfig["freezeBackbone"])

seed = 344567
imageSize = 224
seqLength = 8
l2regAlpha = 0.0
doRate = 0.0
batchSize = trainRunConfig["batchSize"]
prefetchQueueLength = multiprocessing.cpu_count()

random.seed(seed)
tf.random.set_seed(seed+667734)

catalog = pd.read_csv(catalogPath)
print("{0} images are available in catalog".format(len(catalog)))

if "testSplitDfPath" in trainConfig:
    testSplitDf = pd.read_csv(trainConfig["testSplitDfPath"])
    trainCardIds = testSplitDf.loc[testSplitDf.loc[:,'dataset'] == "train",:] # train only
    trainIds = [int(x[2:]) for x in trainCardIds.loc[:,"cardId"].values] # stripping rl/rf prefix
    print(f'{len(trainIds)} cards are suitable for tr/val out of {len(testSplitDf)}')
    #print(f'preview {trainIds[0:10]}')
    #print(catalog.head())
    catalog = catalog[catalog['petId'].isin(trainIds)]
    print(f'{len(catalog)} images are available after excluding test images')
    testSplitDf = None
    trainCardIds = None
    trainIds = None
else:
    print("testSplitDfPath is not specifed in trainConfig. using entire catalog as tr/val (no test samples exclusion)")

if "hardAltSamplesPath" in trainRunConfig:
    hardAltSamplesDf = pd.read_csv(trainRunConfig['hardAltSamplesPath'])
    #ident,hardSamples (space separated)
    hardAltMap = {}
    for row in hardAltSamplesDf.itertuples():
        hardAltMap[int(row.ident[2:])] = [int(x[2:]) for x in row.hardSamples.split()] # stripping rf/rl prefix
    print(f'Loaded {len(hardAltMap)} hard negative cards')
    catalog = catalog[catalog['petId'].isin(hardAltMap.keys())]
    print(f'{len(catalog)} images are available after leaving only the cards with hard negatives cards available')
else:
    hardAltMap = None
    print(f'Hard negative cards path is not specifed in the trainRunConfig')

if len(catalog) == 0:
    exit(1)
    

petSpecificCatalog = catalog.loc[catalog.loc[:,'pet'] == petType,:]
print("{0} images of {1} pet type".format(len(petSpecificCatalog),petType))


isVal = (petSpecificCatalog.loc[:,'petId'].values % 10 == validationFoldNum*2) | \
    (petSpecificCatalog.loc[:,'petId'].values % 10 == validationFoldNum*2+1)
print(isVal)
trainCatalog = petSpecificCatalog.loc[~ (isVal),:]
valCatalog = petSpecificCatalog.loc[isVal,:]



print("Train DS")
trainDs = ds.SimilaritySet(trainCatalog,extractedImagesPath, seed+4221, minImagesPerCardForSimilarity=minImagesPerCardForSimilarity, hardNegativesMap=hardAltMap)

print("Val DS")
valDs = ds.SimilaritySet(valCatalog,extractedImagesPath, seed+4322, minImagesPerCardForSimilarity=minImagesPerCardForSimilarity, hardNegativesMap=hardAltMap)

trSamplesInOneEpochs = trainDs.getSimilarityPetsCount()
vaSamplesInOneEpochs = valDs.getSimilarityPetsCount()

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
    for sample in trainDs.getSamples(cycled=True):
        yield sample

def valGen():
    for sample in valDs.getSamples(cycled=False):
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

def createDataset(sampleGen, shuffleBufferSize=0):
    imagePathsDataset = tf.data.Dataset.from_generator(
        trainGen,
        (tf.string, tf.string, tf.string),
        (tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None])))

    processedDataset = imagePathsDataset \
        .map(loadImages, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True) \
        .map(coerceSeqSizeInTuple, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True) \
        .map(augmentTriple, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True)

    if shuffleBufferSize>0:
        processedDataset = processedDataset.shuffle(shuffleBufferSize,seed=seed+23)    

    dummySupervisedDataset = tf.data.Dataset.zip((processedDataset, zerosDs))

    dummySupervisedBatchedDataset = dummySupervisedDataset \
        .batch(batchSize, drop_remainder=True) \
        .prefetch(prefetchQueueLength)
    return dummySupervisedBatchedDataset

trainTfDs = createDataset(trainGen, 128)
valTfDs = createDataset(valGen, 0)

model, backbone, featureExtractor = efficientSiameseNet.constructSiameseTripletModel(seqLength, l2regAlpha, doRate, imageSize)
print("model constructed")

if os.path.exists(checkpointPath):
    if checkpointBackboneFrozen:
        backbone.trainable = False
    else:
        backbone.trainable = True
    print("Loading pretrained weights {0}".format(checkpointPath))
    model.load_weights(checkpointPath, by_name=True)
    print("Loaded pretrained weights {0}".format(checkpointPath))
else:
    print("Starting learning from scratch")

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
print(trainTfDs)

csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(outputPath,'training_log.csv'), append=False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitoredMetric, factor=0.1, verbose =1,
                                patience=reduceLrPatience, min_lr=minAllowedLR, mode=monitoredMode, min_delta=minMetricDelta)

callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    tf.keras.callbacks.EarlyStopping(patience=earlyStoppingPatience, monitor=monitoredMetric,mode=monitoredMode, min_delta=minMetricDelta),
    # Write TensorBoard logs to `./logs` directory
    #tf.keras.callbacks.TensorBoard(log_dir=outputPath, histogram_freq = 5, profile_batch=0),
    tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(outputPath,"weights.hdf5"),
            save_best_only=True,
            verbose=True,
            mode=monitoredMode,
            save_weights_only=True,
            #monitor='val_root_recall'
            monitor=monitoredMetric # as we pretrain later layers, we do not care about overfitting. thus loss instead of val_los
            ),
    tf.keras.callbacks.TerminateOnNaN(),
    csv_logger,
    reduce_lr
  ]


model.fit(x = trainTfDs, \
    steps_per_epoch= int(math.ceil(trSamplesInOneEpochs / batchSize)), \
    validation_data = valTfDs, \
    validation_steps = int(math.floor(vaSamplesInOneEpochs / batchSize)), \
    verbose = 2,
    callbacks=callbacks,
    shuffle=False, # dataset is shuffled explicilty
    epochs=1000)

print("Done")

# idx1 = 0
# for sample in trainDataset.as_numpy_iterator():
#     sim1,sim2,alt = sample
#     print("run 1:\t{0}:\t{1}\t{2}\t{3}]".format(idx1,sim1.shape,sim2.shape,alt.shape))
#     idx1 += 1    