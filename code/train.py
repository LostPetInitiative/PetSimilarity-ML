import tensorflow as tf
import numpy as np
import pandas as pd
import dataset as ds
import sys
import json

trainConfigPath = sys.argv[1]

with open(trainConfigPath, 'r') as json_file:
    trainConfig = json.load(json_file)

print("Train config:")
print(trainConfig)

catalogPath = trainConfig["catalogPath"]
petType = trainConfig["petType"]
extractedImagesPath = trainConfig["extractedImagesPath"]


catalog = pd.read_csv(catalogPath)
print("{0} images are available".format(len(catalog)))

petSpecificCatalog = catalog.loc[catalog.loc[:,'pet'] == petType,:]
print("{0} images of {1} pet type".format(len(petSpecificCatalog),petType))

ds1 = ds.SimilaritySet(petSpecificCatalog,extractedImagesPath, 123, minImagesPerCardForSimilarity=4)

idx1 = 0
for sample in ds1.getSamples():
    print("run 1:\t{0}:\t{1}]".format(idx1,sample))
    idx1 += 1
    if idx1 == 3:
        break

idx1 = 0
for sample in ds1.getSamples():
    print("run 2:\t{0}:\t{1}]".format(idx1,sample))
    idx1 += 1
    if idx1 == 3:
        break