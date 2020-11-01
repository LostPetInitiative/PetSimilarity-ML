import tensorflow as tf
import efficientSiameseNet
import sys

seqLength = 4
l2regAlpha = 0.0
doRate = 0.0
imageSize = 224

checkpointPath = sys.argv[1]
outputPath = sys.argv[2]

print("Consructing siamese net")
model, backbone, featureExtractor = efficientSiameseNet.constructSiameseTripletModel(seqLength, l2regAlpha, doRate, imageSize)
backbone.trainable = False
print("Constructed. Loading taringed parameter from {0}".format(checkpointPath))
model.load_weights(checkpointPath, by_name=True)
print("Loaded")
print("Feature extacted summary")
print(featureExtractor.summary())
print("Exporting extractor weights to {0}".format(outputPath))
featureExtractor.save_weights(outputPath)
print("Done")



