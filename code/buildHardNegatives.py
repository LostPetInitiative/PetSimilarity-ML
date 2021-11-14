from tqdm import tqdm
import pandas as pd
import sys
import urllib.request, json 

cardIdIndexTablePath = sys.argv[1]
species = sys.argv[2].capitalize()
hardCardsOutputFilePath = sys.argv[3]

# building index map

indexMap = {}
idents = []
with open(cardIdIndexTablePath, 'r') as indexFile:
    counter = 0
    while True:
        # Get next line from file
        line = indexFile.readline()    
        if not line:
            break
        indexMap[counter] = line
        idents.append(line)
        counter += 1

print("Loaded card index of {0} cards".format(len(indexMap)))

maxReturnCount = 10
radiusThreshold = 400 # km
collectionName = "kashtankacards"
#solrStreamingExpressionsURL = f"{solrAddress}/solr/{collectionName}/stream";
storagePrefixURL = "https://kashtanka.pet/api/storage/PetCards/pet911ru/"

featuresIdent = "exp_3_4"
featuresCount = 80

similarityThreshold = "0.95"

featureDims = ",".join(f"features_{featuresIdent}_{d}_d" for d in range(0,80))



for ident in idents:
    print(ident)
    with urllib.request.urlopen(f"{storagePrefixURL}{ident}") as url:
        data = json.loads(url.read().decode())
        anchorLat = data["location"]["lat"]
        anchorLon = data["location"]["lon"]
        featuresTargetVal = ','.join(str(round(x,3)) for x in data["features"][featuresIdent])
    distFilterTerm = f"{{!geofilt sfield=location pt={anchorLat},{anchorLon} d={radiusThreshold}}}"
    #query = f"top(n={maxReturnCount},having(select(search({collectionName},q=\"animal:{species} AND ({distFilterTerm})\",fl=\"id, {featureDims}\",sort=\"id asc\",qt=\"/export\"),id,cosineSimilarity(array({featureDims}), array({featuresTargetVal})) as similarity), gt(similarity, {similarityThreshold})),sort=\"similarity desc\")"    
    query = f"top(n={maxReturnCount},select(search({collectionName},q=\"animal:{species}\",fl=\"id,{featureDims}\",sort=\"id asc\",qt=\"/export\"),id,cosineSimilarity(array({featureDims}), array({featuresTargetVal})) as similarity),sort=\"similarity desc\")"    
    print(query)
    break

exit(1)