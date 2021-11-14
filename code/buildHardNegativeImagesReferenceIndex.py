from tqdm import tqdm
import pandas as pd
import sys

imageAnalysisTablePath = sys.argv[1]
species = sys.argv[2] # cat / dog
outputTablePath = sys.argv[3]

df1 = pd.read_csv(imageAnalysisTablePath)

print("Species {0}".format(species))
print("Loaded {0} rows of image analysis table".format(len(df1)))
print("Results will be written to {0}".format(outputTablePath))


df2 = df1[(df1['pet'] == species) & (df1['dublicate'] == False)]
print("Filtered table preview")
print(df2)
switcher = {
    "lost": "rl",
    "found" : "rf"
}

cards = set()
for row in df2.itertuples(False):
    prefix = switcher.get(row.type)
    cards.add("{0}{1}".format(prefix,row.petId))

print("{0} cards at least one photo of {1} species".format(len(cards), species))
cardList = list(cards)
cardList.sort()
print("output preview: {0}".format(cardList[0:10]))
with open(outputTablePath, 'w') as outputFile:
    for ident in cardList:
        outputFile.write('{0}\n'.format(ident))

print("Done")