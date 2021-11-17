from tqdm import tqdm
import pandas as pd
import sys
import urllib.request, urllib.error,json 
from urllib.parse import urlencode
from os.path import exists

import time
import datetime

import asyncio
from concurrent.futures import ThreadPoolExecutor

maxReturnCount = 1000
radiusThreshold = 400 # km
collectionName = "kashtankacards"

# tunnel can be activated with "kubectl port-forward service/solr 8983:80"
solrAddress = "http://localhost:8983"

featuresIdent = "exp_3_4"
featuresCount = 80

similarityThreshold = "0.99"

cardIdIndexTablePath = sys.argv[1]
species = sys.argv[2].capitalize()
hardCardsOutputFilePath = sys.argv[3]
hardCardsOutputFilePathMissing = f'{hardCardsOutputFilePath}.missing'

if not(exists(hardCardsOutputFilePath)):
    alreadyProcessed = set()
    with open(hardCardsOutputFilePath, "a") as file_object:
        file_object.write("ident,hardSamples\n")
else:
    df1 = pd.read_csv(hardCardsOutputFilePath)
    alreadyProcessed = set(df1['ident'].tolist())
print(f"{len(alreadyProcessed)} cards already have hard samples ready")

if exists(hardCardsOutputFilePathMissing):
    with open(hardCardsOutputFilePathMissing, "r") as file_object:
        missing = [x.strip() for x in file_object.readlines()]
        print(f'{len(missing)} cards are known to be missing in Cassandra')
        for m in missing:
            alreadyProcessed.add(m)

print(f"{len(alreadyProcessed)} cards can be skipped")

# building index map

indexMap = {}
identsToProcess = []
with open(cardIdIndexTablePath, 'r') as indexFile:
    counter = 0
    while True:
        # Get next line from file
        line = indexFile.readline().strip()
        if not line:
            break
        indexMap[counter] = line
        if(line in alreadyProcessed):
            continue
        identsToProcess.append(line)
        counter += 1

print("Loaded card index of {0} cards".format(len(indexMap)))

#identsToProcess = identsToProcess[0:6]
print(f'{len(identsToProcess)} cards to process')

solrStreamingExpressionsURL = f"{solrAddress}/solr/{collectionName}/stream";
storagePrefixURL = "https://kashtanka.pet/api/storage/PetCards/pet911ru/"


featureDims = ",".join(f"features_{featuresIdent}_{d}_d" for d in range(0,80))

def constructQuery(ident):
    with urllib.request.urlopen(f"{storagePrefixURL}{ident}") as conn:
        data = json.loads(conn.read().decode())
        anchorLat = data["location"]["lat"]
        anchorLon = data["location"]["lon"]
        featuresTargetVal = ','.join(str(round(x,3)) for x in data["features"][featuresIdent])
    distFilterTerm = f"{{!bbox sfield=location pt={anchorLat},{anchorLon} d={radiusThreshold} cache=false}}"
    #query = f"top(n={maxReturnCount},having(select(search({collectionName},q=\"animal:{species} AND ({distFilterTerm})\",fl=\"id, {featureDims}\",sort=\"id asc\",qt=\"/export\"),id,cosineSimilarity(array({featureDims}), array({featuresTargetVal})) as similarity), gt(similarity, {similarityThreshold})),sort=\"similarity desc\")"    
    #query = f"having(select(search({collectionName},q=\"*:*\",fq=\"animal:{species}\",fq=\"NOT ({distFilterTerm})\",fl=\"id,{featureDims}\",sort=\"id asc\",qt=\"/export\"),id,cosineSimilarity(array({featureDims}), array({featuresTargetVal})) as similarity), gt(similarity, {similarityThreshold}))"    
    query = f"top(n={maxReturnCount},having(select(search({collectionName},q=\"*:*\",fq=\"animal:{species}\",fq=\"NOT ({distFilterTerm})\",fl=\"id,{featureDims}\",sort=\"id asc\",qt=\"/export\"),id,cosineSimilarity(array({featureDims}), array({featuresTargetVal})) as similarity), gt(similarity, {similarityThreshold})),sort=\"similarity desc\")"
    return query

def fetchHardSamples(query):
    data = urlencode({'expr':query}).encode('ascii')
    with urllib.request.urlopen(f"{solrStreamingExpressionsURL}",data) as conn:
        data = json.loads(conn.read().decode())
        samples = list()
        for elem in data['result-set']['docs']:
            if 'EXCEPTION' in elem:
                print(elem['EXCEPTION'])
                exit(4)
            elif 'EOF' in elem:
                break
            elif ('id' in elem) and ('similarity' in elem):
                samples.append(elem['id'])
            else:
                print("unexpected document")
                print(elem)
                exit(5)
        return samples

async def work():
    concurrentTasksSem = asyncio.Semaphore(1)
    treadPool = ThreadPoolExecutor(max_workers=10)

    loop = asyncio.get_running_loop()

    async def constructQueryAsync(ident):
        try:
            query = await loop.run_in_executor(treadPool, constructQuery, ident)
            return query
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            else:
                exit(1)
        except:
            exit(2)

    async def fetchHardSamplesAsync(query):
        return await loop.run_in_executor(treadPool, fetchHardSamples, query)

    async def getHardSamplesFor(ident):
        await concurrentTasksSem.acquire()
        try:
            query = await constructQueryAsync(ident)
            if query is None:
                return (ident, None)
            else:
                samples = await fetchHardSamplesAsync(query)
                if len(samples) == 0:
                    return (ident, None)
                else:
                    return (ident, samples)
        except:
            exit(6)
        finally:
            concurrentTasksSem.release()
    
    tasks = list()
    for ident in tqdm(identsToProcess, desc="queued", ascii=True, total=len(identsToProcess)):
        tasks.append(asyncio.create_task(getHardSamplesFor(ident)))

    tasksCount = len(tasks)
    counter = 0
    startTime = time.time()
    for coro in tasks:
        (ident,hardSamples) = await coro
        curTime = time.time()
        elapsed = curTime - startTime
        counter += 1

        avgTaskDuration = elapsed / counter
        counterStr = f'{counter} of {tasksCount}'
        paceStr = f'{avgTaskDuration:.2f} sec/task'
        
        tasksLeft = tasksCount - counter
        leftTimeSec = tasksLeft * avgTaskDuration
        eta = datetime.datetime.fromtimestamp(curTime + leftTimeSec, datetime.timezone(datetime.timedelta(hours=3)))
        etaStr = eta.strftime('%Y-%m-%d %H:%M:%S')
        
        if hardSamples is None:
            with open(hardCardsOutputFilePathMissing, "a") as file_object:
                file_object.write(f'{ident}\n')
            print(f'{ident}({counterStr}): hard samples not found. {paceStr}. eta {etaStr}.')
        else:
            shortIds = [x[9:] for x in hardSamples] # removing 'pet911ru/'
            joinedShortIds = ' '.join(shortIds)
            lineToDump = f'{ident},{joinedShortIds}\n'
            with open(hardCardsOutputFilePath, "a") as file_object:
                file_object.write(lineToDump)
            print(f'{ident}({counterStr}): {len(hardSamples)} hard samples generated. {paceStr}. eta {etaStr}.')
        sys.stderr.flush()
        sys.stdout.flush()

asyncio.run(work())
exit(0)