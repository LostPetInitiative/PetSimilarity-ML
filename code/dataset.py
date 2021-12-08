import pandas as pd
import numpy as np
import random
import math
import os

class SimilaritySet:
    def __init__(self, catalogDf,imagesBaseDir, randomSeed, minImagesPerCardForSimilarity=2, dataBinsCount = 100, hardNegativesMap=None):    
    
        uniquePets = set()
        self.petToImages = dict()
        self.r = random.Random(randomSeed)
        self.r_state = self.r.getstate()
        self.hardNegativesMap = hardNegativesMap
        
        for row in catalogDf.itertuples():
            petId = int(row.petId)
            imageId = int(row.imageFile[:-4])
            binNum = imageId % dataBinsCount
            binnedImagePath = os.path.join(imagesBaseDir,str(binNum),row.imageFile)

            if not(petId in self.petToImages):
                self.petToImages[petId] = [binnedImagePath]
            else:
                self.petToImages[petId].append(binnedImagePath)
            uniquePets.add(petId)

        self.allPets = list(uniquePets)
        self.similarityPets = list()
        for petId in self.allPets:
            if len(self.petToImages[petId]) >= minImagesPerCardForSimilarity:
                self.similarityPets.append(petId)
        print("Dataset constructed: {0} pets, {1} of which can be used for positive samples splits".format(len(self.allPets), len(self.similarityPets)))

    def getSimilarityPetsCount(self):
        return len(self.similarityPets)

    def getSamples(self, cycled=False, simImagesSplit='equal'):
        """simImagesSplit: equal or uniformRandom"""
        self.r.setstate(self.r_state)

        isFirstIter = True
        while cycled or isFirstIter:
            for similarityPetId in self.similarityPets:
                simImages = list(self.petToImages[similarityPetId])
                self.r.shuffle(simImages)
                simImagesCount = len(simImages)
                if simImagesSplit == "equal":
                    rightPartStartIdx = simImagesCount // 2                    
                else:
                    rightPartStartIdx = math.floor(self.r.random()*(simImagesCount-1) + 1)                    
                simPart1 = simImages[0:rightPartStartIdx]
                simPart2 = simImages[rightPartStartIdx:]

                alternativeFound = False
                noAlternativesExist = False
                if not(self.hardNegativesMap is None):
                    altPetIds = self.hardNegativesMap[similarityPetId]
                while (not alternativeFound) & (not noAlternativesExist):
                    if self.hardNegativesMap is None:
                        altIdx = math.floor(self.r.random() * len(self.allPets))
                        altPetId = self.allPets[altIdx]
                    else:                        
                        #print(f'anchor {similarityPetId}. hard samples {altPetIds}')
                        if len(altPetIds) == 0:
                            noAlternativesExist = True
                            break
                        idx = math.floor(self.r.random() * len(altPetIds))
                        altPetId = altPetIds[idx]

                        # this hard negative ID can have no photos
                        if not(altPetId in self.petToImages):
                            del altPetIds[idx]
                            continue                        
                    if altPetId != similarityPetId:
                        alternativeFound = True
                if noAlternativesExist:
                    continue
                altPart = self.petToImages[altPetId]
                yield (
                    np.array(simPart1),
                    np.array(simPart2),
                    np.array(altPart))
            isFirstIter = False


