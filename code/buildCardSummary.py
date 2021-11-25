import json
import asyncio
import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
import multiprocessing

dbPath = sys.argv[1]
outFile = sys.argv[2]
print("Pet911.ru downloaded database is at {0}".format(dbPath))
print("Will save final summary into {0}".format(outFile))

async def work():
    loop = asyncio.get_running_loop()
    treadPool = ThreadPoolExecutor(max_workers=10)
    fileReadingSem = asyncio.Semaphore(8)

    def LoadPetCard(cardPath):
        with open(cardPath, mode='r') as f:
            contents = f.read()
            parsed = json.loads(contents)
            return parsed

    async def LoadPetCardAsync(cardPath):
        try:
            await fileReadingSem.acquire()
            return await loop.run_in_executor(treadPool, LoadPetCard, cardPath)
        finally:
            fileReadingSem.release()

    async def GetPetCardImageInfo(petDirPath):
        cardPath = os.path.join(petDirPath,"card.json")
        await fileReadingSem.acquire()
        try:
            card = await LoadPetCardAsync(cardPath)
        finally:
            fileReadingSem.release()
        #print("card is {0}. type {1}".format(card,type(card)))
        petStr = "unknown"
        typeStr = "unknown"
        sexStr = "unknown"
        pet = card['pet']
        if pet['animal'] == "2":
            petStr = "cat"
        elif pet['animal'] == "1":
            petStr = "dog"
        if pet['sex'] == "2":
            sexStr = "male"
        elif pet['sex'] == "3":
            sexStr = "female"
        
        if pet['art'][:2] == "rl":
            typeStr = "lost"
        else:
            typeStr = "found"
        imageFiles = [x for x in os.listdir(petDirPath) if x.endswith(".png") or x.endswith(".jpg")]

        return {
            "cardId": pet['art'],
            "cardType": typeStr,
            "species": petStr,
            "sex": sexStr,
            "photoCount": len(imageFiles)
            }

    petDirs = [os.path.join(dbPath,x) for x in os.listdir(dbPath) if os.path.isdir(os.path.join(dbPath,x))]
    #petDirs = petDirs[0:4096]
    print("Found {0} pet directories".format(len(petDirs)))


    tasks = [asyncio.create_task(GetPetCardImageInfo(petDir)) for petDir in petDirs]
    gatheredResults = []
    for coro in tqdm(asyncio.as_completed(tasks), desc="Pets processed", total=len(petDirs), ascii=True):
        taskRes = await coro
        gatheredResults.append(taskRes)
    print("Analyzed {0} images. Dumping to CSV summary".format(len(gatheredResults)))
    df1 = pd.DataFrame.from_records(gatheredResults)
    df1.to_csv(outFile,index=False)
    print("Done")



asyncio.run(work(),debug=False)

    
