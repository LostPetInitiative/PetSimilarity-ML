import json
import asyncio
import aiofiles
import imagehash
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from skimage import io
from PIL import Image
import pandas as pd
import multiprocessing

hashSize = 8
cpuCount = multiprocessing.cpu_count()

dbPath = sys.argv[1]
outFile = sys.argv[2]
print("Pet911.ru downloaded database is at {0}".format(dbPath))
print("Will save final summary into {0}".format(outFile))

async def LoadPetCardAsync(cardPath):
    async with aiofiles.open(cardPath, mode='r') as f:
        contents = await f.read()
        f.close()
        parsed = json.loads(contents)
        return parsed

def LoadPetCard(cardPath):
    with open(cardPath, mode='r') as f:
        contents = f.read()
        parsed = json.loads(contents)
        return parsed

def GetImageHash(imagePath):
    try:
        if imagePath.endswith(".png"):
            imNumpy = io.imread(imagePath,plugin='imread')
        else:
            imNumpy = io.imread(imagePath)
        im = Image.fromarray(imNumpy)
        a_hash = imagehash.average_hash(im, hash_size=hashSize)
        p_hash = imagehash.phash(im, hash_size=hashSize)
        d_hash = imagehash.dhash(im, hash_size=hashSize)
        w_hash = imagehash.whash(im, hash_size=hashSize)
        return False,(a_hash,p_hash,d_hash,w_hash)
    except:
        return True,(0,0,0,0)
    

async def work():
    loop = asyncio.get_running_loop()
    fileReadingSem = asyncio.Semaphore(8)

    print("Detected {0} CPUs".format(cpuCount))
    io_pool_exc = ThreadPoolExecutor(max_workers=cpuCount*2)


    async def GetPetCardImageInfo(petDirPath, hashSimilarityThreshold = 4):
        cardPath = os.path.join(petDirPath,"card.json")
        await fileReadingSem.acquire()
        try:
            card = await LoadPetCardAsync(cardPath)
        finally:
            fileReadingSem.release()
        #print("card is {0}. type {1}".format(card,type(card)))
        petStr = ""
        typeStr = ""
        pet = card['pet']
        if pet['animal'] == "2":
            petStr = "cat"
        else:
            petStr = "dog"
        if pet['art'][:2] == "rl":
            typeStr = "lost"
        else:
            typeStr = "found"
        imageFiles = [x for x in os.listdir(petDirPath) if x.endswith(".png") or x.endswith(".jpg")]
        fullImagePaths = [os.path.join(petDirPath, x) for x in imageFiles]

        def processImages():    
            hashesRes = [GetImageHash(x) for x in fullImagePaths]
            curruptedMap = [isCurrepted for (isCurrepted,_) in hashesRes]
            hashes = [hashVal for (_,hashVal) in hashesRes]
            dubplicateFlags = [False for x in hashes]
            hashesCount = len(hashes)
            result = []
            for idx1 in range(hashesCount-1):
                if curruptedMap[idx1]:
                    continue
                for idx2 in range(idx1+1,hashesCount):
                    if curruptedMap[idx2]:
                        continue
                    (a_hash_1, p_hash_1, d_hash_1, w_hash_1) = hashes[idx1]
                    (a_hash_2, p_hash_2, d_hash_2, w_hash_2) = hashes[idx2]
                    hashDiff = \
                        (a_hash_1 - a_hash_2) + \
                        (p_hash_1 - p_hash_2) + \
                        (d_hash_1 - d_hash_2) + \
                        (w_hash_1 - w_hash_2)
                    if hashDiff <= hashSimilarityThreshold:
                        # dubplicateFlags[idx1] = True # first one is not considered as duplicate
                        dubplicateFlags[idx2] = True
                        break
            for idx in range(hashesCount):
                result.append({'petId':int(pet['id']),'pet':petStr, "imageFile":imageFiles[idx],"type":typeStr, "dublicate":dubplicateFlags[idx], "currupted":curruptedMap[idx]})
            return result

        result = await loop.run_in_executor(io_pool_exc, processImages)
        return result

    petDirs = [os.path.join(dbPath,x) for x in os.listdir(dbPath) if os.path.isdir(os.path.join(dbPath,x))]
    #petDirs = petDirs[0:4096]
    print("Found {0} pet directories".format(len(petDirs)))


    tasks = [asyncio.create_task(GetPetCardImageInfo(petDir)) for petDir in petDirs]
    gatheredResults = []
    for coro in tqdm(asyncio.as_completed(tasks), desc="Pets processed", total=len(petDirs), ascii=True):
        taskRes = await coro
        gatheredResults.extend(taskRes)
    print("Analyzed {0} images. Dumping to CSV summary".format(len(gatheredResults)))
    df1 = pd.DataFrame.from_records(gatheredResults)
    df1.to_csv(outFile,index=False)
    print("Done")



asyncio.run(work(),debug=False)

    
