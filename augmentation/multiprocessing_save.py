from augment import Augment
import numpy as np
from multiprocessing import Pool
import sys
import random as rd
from tqdm import tqdm
import argparse
import parmap
global counter
counter = 0

def run(ind,a, ANOMALY, x, blocksize):
    out,did,hh,_,_ = a.augment_random(ANOMALY[0])
    np.save(f"{ANOMALY[2]}/{ANOMALY[0]}_{ind+(x*blocksize)}.npy", out)
    return out


def main(PLT_CHANNELS,ITERATIONS, a,ANOMALY, del_same=False, blocksize=None):
    global BLOCKSIZE
    BLOCKSIZE = blocksize
    if blocksize is not None:
        full = ITERATIONS//blocksize
        for x in tqdm(range(full)):
            with Pool(8) as pool:
                iterable = zip(range(blocksize),[a for _ in list(range(blocksize))], [ANOMALY for _ in range(blocksize)])
                d = np.array(parmap.starmap(run,list(iterable),x, blocksize, pm_pbar=False))
                defs = list(np.load("definitions.npy", allow_pickle=True))
                for x in range(len(defs)):
                    if defs[x][0] == ANOMALY[0]:
                        out_ann_ = int(defs[x][1])
                try:
                    #print(have.shape, np.array(d).shape)
                    have_anns = np.load(f"{ANOMALY[2]}/{ANOMALY[0]}_ann.npy")
                    out_ann = np.concatenate((have_anns, [[out_ann_] for _ in range(len(d))]))
                except Exception as e:
                    out_ann = np.array([[out_ann_] for _ in range(len(d))])
                    np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', np.array(out_ann))
                else:
                    np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', out_ann)
        leftover = ITERATIONS % blocksize
        if leftover != 0:
            with Pool(8) as pool:
                    iterable = zip(range(leftover),[a for _ in list(range(leftover))], [ANOMALY for _ in range(leftover)])
                    d = parmap.starmap(run,list(iterable),full,blocksize,pm_pbar=True)
                    defs = list(np.load("definitions.npy", allow_pickle=True))
                    for x in range(len(defs)):
                        if defs[x][0] == ANOMALY[0]:
                            out_ann_ = int(defs[x][1])
                    have_anns = np.load(f"{ANOMALY[2]}/{ANOMALY[0]}_ann.npy")
                    out_ann = np.concatenate((have_anns, np.array([[out_ann_] for _ in range(len(d))])))
                    np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', out_ann)
    else:
        with Pool(8) as pool:
            iterable = zip(range(ITERATIONS),[a for _ in list(range(ITERATIONS))], [ANOMALY for _ in range(ITERATIONS)])

            d = parmap.starmap(run,list(iterable),0,0,pm_pbar=True)
            defs = list(np.load("definitions.npy", allow_pickle=True))
            for x in range(len(defs)):
                if defs[x][0] == ANOMALY[0]:
                    out_ann_ = int(defs[x][1])
            try:
                have_anns = np.load(f"{ANOMALY[2]}/{ANOMALY[0]}_ann.npy")
                out_ann = np.concatenate((have_anns, [[out_ann_] for _ in range(len(d))]))
            except:
                out_ann = [[out_ann_] for _ in range(len(d))]
                np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', np.array(out_ann))
            else:
                np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', out_ann)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inPath", type=str, help="path to -fecg and -ann npy files")
    parser.add_argument("-o", "--outPath", type=str, help="path to save generated anomaly segments and annotations")
    parser.add_argument("-t", "--anomaly", type=str, help="anomaly type")
    parser.add_argument("-n", "--iterations", type=int, help="number of segments to generate")
    parser.add_argument("-b", "--blockSize", type=int, help="optional, how often to save anomaly file")
    args = parser.parse_args()
    PLT_CHANNELS = list(range(12))
    a = Augment(use_path=True, path=args.inPath,anomaly_type=args.anomaly)
    ANOMALY = (args.anomaly, "", args.outPath) # ANOMALY type, "", out path
    ITERATIONS = args.iterations
    try:
        BLOCKSIZE = args.blockSize
    except:
        BLOCKSIZE = None
    main(PLT_CHANNELS, ITERATIONS, a, ANOMALY, blocksize=BLOCKSIZE)
