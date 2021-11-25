import pandas as pd
import glob
from PIL import Image
from tqdm import tqdm
import sys

import asyncio
from concurrent.futures import ThreadPoolExecutor

inputDirPath = sys.argv[1]
outputPath = sys.argv[2]

async def work():
    concurrentTasksSem = asyncio.Semaphore(8)
    treadPool = ThreadPoolExecutor(max_workers=10)

    loop = asyncio.get_running_loop()

    def getDims(path):
        with Image.open(path) as img:
            width, height = img.size
            return (width,height)

    async def getDimsAsync(path):
        try:
            await concurrentTasksSem.acquire()
            return await loop.run_in_executor(treadPool, getDims, path)
        except:
            print(f"{path} is corrupted image")
            return None
        finally:
                concurrentTasksSem.release()

    print(f"imput dir is {inputDirPath}")

    jpg = glob.glob(f"{inputDirPath}/*/*.jpg")
    print(f"{len(jpg)} images jpg")
    jpg2 = glob.glob(f"{inputDirPath}/*/*.JPG")
    print(f"{len(jpg2)} images JPG")
    jpg3 = glob.glob(f"{inputDirPath}/**/*.jpeg")
    print(f"{len(jpg3)} images jpeg")
    jpg4 = glob.glob(f"{inputDirPath}/**/*.JPEG")
    print(f"{len(jpg4)} images JPEG")
    png = glob.glob(f"{inputDirPath}/**/*.png")
    print(f"{len(png)} images png")
    png2 = glob.glob(f"{inputDirPath}/**/*.PNG")
    print(f"{len(png2)} images PNG")
    concatenated = jpg + jpg2 + jpg3 + jpg4 + png + png2
    #concatenated = concatenated[0:100]
    set1 = set(concatenated)
    print(f"{len(set1)} images to process")
    print(concatenated[0:10])

    tasks = list()
    for ident in tqdm(set1, desc="queued", ascii=True):
        tasks.append(asyncio.create_task(getDimsAsync(ident)))

    widths = list()
    heights = list()
    for coro in tqdm(tasks, desc="evaluated", ascii=True):
        size = await coro
        if size is None:
            continue
        w,h = size
        widths.append(w)
        heights.append(h)

    df = pd.DataFrame(list(zip(widths, heights)),
        columns =['width', 'height'])
    df.to_csv(outputPath, index=False)

asyncio.run(work())
exit(0)
        
    


