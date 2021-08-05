from augment import Augment
import numpy as np
from multiprocessing import Pool
import sys
import random as rd
from tqdm import tqdm
import parmap
import argparse
global counter
counter = 0
global BLOCKSIZE
BLOCKSIZE = 1024

def run(ind,a, ANOMALY, x, blocksize, ret=False):
    out,did,hh,_,_ = a.augment_random(ANOMALY[0])
    if not ret:
        np.save(f"{ANOMALY[2]}/{ANOMALY[0]}_{ind+(x*blocksize)}.npy", out) # only save if not return
    return out


def main(PLT_CHANNELS,ITERATIONS, a,ANOMALY, del_same=False, blocksize=None, ret=False):
    global BLOCKSIZE
    retu, retu_ann = [], []
    BLOCKSIZE = blocksize
    if blocksize is not None:
        full = ITERATIONS//blocksize
        for x in tqdm(range(full)):
            with Pool(8) as pool:
                iterable = zip(range(blocksize),[a for _ in list(range(blocksize))], [ANOMALY for _ in range(blocksize)])
                d = np.array(parmap.starmap(run,list(iterable),x, blocksize,ret, pm_pbar=False))
                if ret:
                    retu.append(d)
                defs = list(np.load("definitions.npy", allow_pickle=True))
                for x in range(len(defs)):
                    if defs[x][0] == ANOMALY[0]:
                        out_ann_ = int(defs[x][1])
                if not ret:
                    try:
                        #print(have.shape, np.array(d).shape)
                        have_anns = np.load(f"{ANOMALY[2]}/{ANOMALY[0]}_ann.npy")
                        out_ann = np.concatenate((have_anns, [[out_ann_] for _ in range(len(d))]))
                    except Exception as e:
                        out_ann = np.array([[out_ann_] for _ in range(len(d))])
                        np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', np.array(out_ann))
                    else:
                        np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', out_ann)
                else:
                    retu_ann.append([out_ann_] for _ in range(len(d)))

        leftover = ITERATIONS % blocksize
        if leftover != 0:
            with Pool(8) as pool:
                    iterable = zip(range(leftover),[a for _ in list(range(leftover))], [ANOMALY for _ in range(leftover)])
                    d = parmap.starmap(run,list(iterable),full,blocksize,ret,pm_pbar=True)
                    if ret:
                        retu.append(d)
                    defs = list(np.load("definitions.npy", allow_pickle=True))
                    for x in range(len(defs)):
                        if defs[x][0] == ANOMALY[0]:
                            out_ann_ = int(defs[x][1])
                    if not ret:
                        have_anns = np.load(f"{ANOMALY[2]}/{ANOMALY[0]}_ann.npy")
                        out_ann = np.concatenate((have_anns, np.array([[out_ann_] for _ in range(len(d))])))
                        np.save(f'{ANOMALY[2]}/{ANOMALY[0]}_ann.npy', out_ann)
                    else:
                        retu_ann.append([out_ann_] for _ in range(len(d))) 
        return retu, retu_ann
                    
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

def augment(in_path: str, anomaly_type: str, iterations: int, block_size: int=1024) -> (np.ndarray, np.ndarray):
    global BLOCKSIZE
    PLT_CHANNELS = list(range(12))
    a = Augment(use_path=True, path=in_path,anomaly_type=anomaly_type)
    ANOMALY = (anomaly_type, "", ".") # ANOMALY type, "", out path
    ITERATIONS = iterations
    try:
        BLOCKSIZE = block_size
    except:
        BLOCKSIZE = None
    ecg, ann = main(PLT_CHANNELS, ITERATIONS, a, ANOMALY, blocksize=BLOCKSIZE, ret=True)
    return np.array(ecg), np.array(ann)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str,
                    help="the location of the input -fecg.npy files")
    parser.add_argument("anomaly_type", type=str,
                    help="anomaly type to augment ex. RBBB")
    parser.add_argument("iterations", type=int,
                    help="number of augmented samples to return")
    parser.add_argument("out_path", type=str,
                    help="location to save the output segments and annotations")

    args = parser.parse_args()
    augmented_segments, annotations = augment(args.input_path, args.anomaly_type, args.iterations)
    np.save(args.out_path + "/augmented_segments_" + args.anomaly_type + ".npy", augmented_segments)
    np.save(args.out_path + "/augmented_annotations_" + args.anomaly_type + ".npy", annotations)
