#!/Library/Frameworks/Python.framework/Versions/3.8/bin/python3
from scipy import signal
from scipy import stats
import numpy as np
from pylab import rcParams
from glob import glob
import random as rd
import sys
import math
import matplotlib.pyplot as plt

NUM_PEAKS = 5

VARIANCE_MIN = 0.45
VARIANCE_MAX = 0.5 # for smoothing

AUGMENT_MAX = 1.5 # max peak multiplier
AUGMENT_MIN = 0.5

AUGMENT_STEP = 0.01 # increase to get less "random" segments

SIMILAR_MAX = 20 # +- 20 samples to search for similar peak across different channels

SMOOTHING_FUNCTION = lambda x,y,z: z*(((1/((y)*math.sqrt(2*math.pi)))*math.e**-(x**2/(2*(y**2))))) #x data, y variance, z sign

CHANNELS = list(range(12))

class Augment:
    def __init__(self, annotations=None, data=None, current_ind=0, path=None, use_path=True, anomaly_type=None):
        self.annotation_mapping = np.load("definitions.npy", allow_pickle=True) # what annotations mean
        if use_path and path is not None and current_ind == 0 and data is None and annotations is None and anomaly_type is not None:
            self._load_all(path, anomaly_type)
        elif data is not None and annotations is not None and current_ind is not None and anomaly_type is not None:
            self._load_ind(data, annotations, current_ind, anomaly_type)
        else:
            print("check params, either give a path name or an index")
            sys.exit(0)

    def augment_random(self, anomaly_type: str) -> tuple:
        """augments a random index from path data

        Args:
            anomaly_type (str): anomaly type to augment

        Returns:
            tuple: 
        """
        assert self.type == "path"

        self.data = np.array(rd.choice(self.data_)) # self.data_ contains tons of segments from all patients... therefore need singular patient
        self.baseline = [self.find_baseline(channel) for channel in CHANNELS]
        self.init_peaks_indices = [self.find_peaks(channel) for channel in CHANNELS]
        self.init_peaks_seg = [[self.find_whole_peak(peak=i, channel=channel, ind=None)[1] for i in self.init_peaks_indices[channel]] for channel in CHANNELS] # get segment values for each of the found peaks for channel CH
        self.init_peaks_ranges = [[self.find_whole_peak(peak=i, channel=channel)[0] for i in self.init_peaks_indices[channel]] for channel in CHANNELS] # get segment values for each of the found peaks for channel CH
        self.get_similar_timed()
        return self.run_augment(anomaly_type)

    def augment_ind(self, ind: int, anomaly_type: str) -> tuple:
        """augments and returns a singular index of the data

        Args:
            ind (int): data index to augment
            anomaly_type (str): anomaly type to augment

        Returns:
            tuple:
        """
        assert self.type == 'ind'

        return self.run_augment(anomaly_type, ind=ind)


    def _load_ind(self, data: np.ndarray, annotations: np.ndarray, current_ind: int, anomaly_type: str) -> None:
        """load data from a given index only

        Args:
            data (np.ndarray): ECG data
            annotations (np.ndarray): numpy array of the annotations that correspond to the current index
            current_ind (int): the current index to load
            anomaly_type (str): anomaly type to load
        """
        self.type = 'ind'
        self.data = data
        self.annotations = annotations
        self.curr = current_ind
        self.init_peaks_indices = [self.find_peaks(channel,ind=current_ind) for channel in CHANNELS]
        self.init_peaks_seg = [[self.find_whole_peak(peak=i, channel=channel, ind=current_ind)[1] for i in self.init_peaks_indices[channel]] for channel in CHANNELS] # get segment values for each of the found peaks for channel CH
        self.init_peaks_ranges = [[self.find_whole_peak(peak=i, channel=channel, ind=current_ind)[0] for i in self.init_peaks_indices[channel]] for channel in CHANNELS] # get segment values for each of the found peaks for channel CH
        self.baseline = [self.find_baseline(channel, ind=current_ind) for channel in CHANNELS]
        
    
    def _load_all(self, path: str, anomaly_type: str) -> None:
        """load all ECG data from path

        Args:
            path (str): path to load .npy files from
            anomaly_type (str): anomaly type to load
        """
        self.type = 'path'
        self.data_ = []
        filenames = glob(f"{path}/*-fecg.npy")
        #print(self.annotation_mapping)
        data, data_ann, order = [], [], []
        for filename in filenames:
            #print(filename)
            tmp = np.load(filename)
            annotations = (np.load(filename.split("-")[0] + "-ann.npy"))
            for n,i in enumerate(tmp):
                if annotations[n] != 1 and self.annotation_mapping[annotations[n][0]][0] == anomaly_type:
                    data.append(i) # dont care about the index anymore just need raw data that is of type anomaly_type
        self.data_ = np.array(data)
        assert len(self.data_) != 0
        

    def find_baseline(self, channel: int, ind: int=None) -> list:
        """finds the baseline of a signal at channel

        Args:
            channel (int): channel of data to find baseline of
            ind (int, optional): index of data to look at. Defaults to None.

        Returns:
            list: list of baseline the same length of the input data segment for comparison
        """
        if ind is None:
            nans = list(filter(lambda x: np.isnan(x), self.data[:,channel]))
            return [np.nanmean(self.data[:,channel])]*(len(self.data[:,channel])-len(nans))
        else:
            nans = list(filter(lambda x: np.isnan(x), self.data[ind][:,channel]))
            return [np.nanmean(self.data[ind][:,channel])]*(len(self.data[ind][:,channel])-len(nans))

    def get_anomalous_types(self) -> list:
        """returns the list of anomalies in the current dataset

        Returns:
            [type]: list of anomaly types
        """
        types = []
        for n,i in enumerate(self.data):
            if self.annotations[n] != 1:
                types.append((self.annotation_mapping[self.annotations[n][0]][0], n)) 
        return (types)

    def get_anomalous_type(self, ind: int) -> str:
        """returns the anomalous type at index ind

        Args:
            ind (int): index in dataset

        Returns:
            str: anomalous type
        """
        return self.annotation_mapping[self.annotations[ind][0]][0]

    def get_similar_timed(self) -> None:
        """tries to find peaks in different channels that are at about the same "time"
        """
        self.similars = [[[]]for _ in range(NUM_PEAKS)]
        comp = [[[]for channel in range(len(CHANNELS))]for _ in range(NUM_PEAKS)]
        
        for n,x in enumerate(self.init_peaks_indices): # 12
            for peak in range(len(x)): # 5
                comp[peak][n] = x[peak]

        
        for m,x in enumerate(comp): # for each peak detected
            compare = x[0] # compare against the first channel lead for each peak that is detected
            self.similars[m] = [[0,compare]] # set initial
            for channel in range(1,len(CHANNELS[1:])+1):
                for stretch in range(0,SIMILAR_MAX):
                    if x[channel] + stretch == compare or x[channel] - stretch == compare:
                        # similar peak to compare
                        if self.similars[m] == [[]]:
                            self.similars[m] = [[channel,x[channel]]]
                        else:
                            self.similars[m].append([channel,x[channel]])
                        break
    
        #[[[0, 80], [2, 79], [4, 90], [5, 65], [6, 66], [7, 72], [8, 78], [11, 81]], [[0, 90], [2, 90], [3, 79], [4, 96], [5, 79], [6, 82], [7, 85], [8, 88], [9, 79], [10, 80], [11, 90]], [[0, 110], [2, 96], [4, 110], [7, 96], [8, 95], [11, 96]], [[0, 138], [2, 138], [4, 138], [11, 138]], [[0, 143], [2, 142], [3, 134], [4, 143], [5, 138], [6, 138], [7, 142], [8, 138], [9, 138], [10, 138], [11, 162]]]
        
    def _flip(self, data: np.ndarray, sign: int) -> list:
        """flip data about the baseline instead of about 0 (makes peak detection for both positive and negative peaks more reliable)

        Args:
            data (np.ndarray): in np.ndarray to flip
            sign (int): sign to flip about

        Returns:
            list: flipped data
        """
        data = np.copy(data)
        out = []
        for x in data:
            if sign == -1:
                if x > np.nanmean(data):
                    #out.append(np.nanmean(data))
                    out.append(x)
                elif x < np.nanmean(data):
                    out.append(np.nanmean(data)+(np.nanmean(data)-x))
                else:
                    out.append(x)
            else:
                if x < np.nanmean(data):
                    #out.append(np.nanmean(data))
                    out.append(np.nan)
                elif x > np.nanmean(data):
                    out.append(np.nanmean(data)-(np.nanmean(data)-x))
                else:
                    out.append(x)
        assert len(out) == len(data)
        return out
    
    def find_peaks(self, channel: int, ind: int=None, override: int=0) -> list:
        """find the peaks of an ECG segment using scipy find_peaks

        Args:
            channel (int): channel to search for peaks in
            ind (int, optional): index of data. Defaults to None.
            override (int, optional): [description]. Defaults to 0.

        Returns:
            list: [description]
        """
        data = self._flip(self.data[ind][:,channel],-1) if ind is not None else self._flip(self.data[:,channel],-1)
        
        peaks, _ = signal.find_peaks(data, height=np.nanmean(self.data[ind][:,channel])+0.01-override) if ind is not None else signal.find_peaks(data, height=np.nanmean(self.data[:,channel])+0.01-override)
        #plt.scatter(peaks, self.data[ind][:,channel][peaks])
        #plt.show()
        proms = signal.peak_prominences(data, peaks=peaks)[0]
        collect = list(zip(proms,peaks))
        collect = sorted(collect, key=lambda x: x[0])
        collect = [c[1] for c in collect]
        collect_indices = collect
        start = 5
        if len(collect_indices) == NUM_PEAKS:
            return sorted(collect_indices)
        elif len(collect_indices) > NUM_PEAKS:
            s = collect
            
            return s[-NUM_PEAKS:]
        else:
            if override - 0.001 > 0:
                self.find_peaks(ind, channel, override=override-0.001)
            else:
                for x in range(5-len(collect_indices)):
                    collect_indices.append(0)
                #print('error, did not find enough peaks... maybe change NUM_PEAKS parameter for this anomaly type?')
                #sys.exit(0)
        return sorted(collect_indices)


    def find_whole_peak(self, peak: int, channel: int, ind: int=None) -> list:
        """finds the entire length of a peak, rather than just the exact index of peak. searches for baseline crosses to the left and right of peaks found by find_peaks().

        Args:
            peak (int): which peak currently looking at
            channel (int): channel of the data
            ind (int, optional): index of data. Defaults to None.

        Returns:
            list: list of peak ranges
        """
        
        baseline = self.find_baseline(channel, ind)
        left, right = 0, 1
        #print(self.data[:,channel], peak, self.data[:,channel][peak])
        if ind is not None:
            if (self.data[ind][:,channel][peak]) > baseline[0]:
                for i in range(peak, len(self.data[ind][:,channel])):
                    if self.data[ind][:,channel][i] <= baseline[0]:
                        right = i
                        break
                for i in range(peak, 0, -1):
                    if self.data[ind][:,channel][i] <= baseline[0]:
                        left = i
                        break
            else:
                for i in range(peak, len(self.data[ind][:,channel])):
                    if self.data[ind][:,channel][i] >= baseline[0]:
                        right = i
                        break
                for i in range(peak, 0, -1):
                    if self.data[ind][:,channel][i] >= baseline[0]:
                        left = i
                        break
            
            if right == 1:
                right = len(self.data[ind][:,channel]) - 1
            if right < left:
                right = len(self.data[ind][:,channel]) - 1
            return list(range(left, right)), self.data[ind][:,channel][left:right]
        else:
            #print(baseline[0])
            #print(self.data[:,channel][peak] > baseline[0])
            if (self.data[:,channel][peak]) > baseline[0]:
                for i in range(peak, len(self.data[:,channel])):
                    if self.data[:,channel][i] <= baseline[0]:
                        right = i
                        break
                for i in range(peak, 0, -1):
                    if self.data[:,channel][i] <= baseline[0]:
                        left = i
                        break
            else:
                for i in range(peak, len(self.data[:,channel])):
                    if self.data[:,channel][i] >= baseline[0]:
                        right = i
                        break
                for i in range(peak, 0, -1):
                    if self.data[:,channel][i] >= baseline[0]:
                        left = i
                        break
            
            if right == 1:
                right = len(self.data[:,channel]) - 1
            if right < left:
                right = len(self.data[:,channel]) - 1
                left = left - 2
            return list(range(left, right)), self.data[:,channel][left:right]

    def get_data(self, channel, ind):
        return self.data[ind][:,channel]

    def run_augment(self, anomaly_type: str, ind: int=None, use_noise: bool=False) -> tuple:
        """Main augmentation program

        Args:
            anomaly_type (str): string for the anomaly type
            ind (int, optional): index of data. Defaults to None.
            use_noise (bool, optional): add noise to signal?. Defaults to False.

        Returns:
            tuple: augmented data
        """
        try:
            self.get_similar_timed()
            chosen_factors = []
            factorss = []
            rem_factors = [None for _ in range(5)]
            factors_dict = {}
            for m,peak_ in enumerate(self.similars):
                chosen_factor = rd.choice(np.arange(AUGMENT_MIN, AUGMENT_MAX, AUGMENT_STEP))
                #chosen_factor = AUGMENT_MAX
                for channel in range(len(CHANNELS)):
                    try:
                        index = list(filter(lambda x: x[0] == channel, peak_))[0][1]
                    except:
                        print('arb')
                        index = self.init_peaks_indices[channel][m]
                        factors_dict[(m,channel)] = round(rd.choice(np.arange(AUGMENT_MIN, AUGMENT_MAX, AUGMENT_STEP)),3) # choose arb factor
                        #factors_dict[(m,channel)] = round(AUGMENT_MAX,3)
                    else:
                        factors_dict[(m,channel)] = round(chosen_factor,3) # set all similar to same value (peak, channel, index)
            assert len(factors_dict) == NUM_PEAKS * len(CHANNELS)
            
            out = np.copy(self.data[ind]) if ind is not None else np.copy(self.data) # format output and only apply scaling on peaks
            np.save("orig.npy", self.data[ind])
            augmented_ = []
            augmented__ = [[] for _ in range(len(CHANNELS))]
            for augmented in range(1):
                num_to_augment = rd.randint(2,NUM_PEAKS-1) # choose a random number of peaks to augment
                #num_to_augment = 5
                peaks_ = list(range(NUM_PEAKS))
                
                for peak in range(num_to_augment):
                    factorss.append([])
                    chosen_factors.append([])
                    choice_to_augment = rd.choice(peaks_)
                    augmented_.append(choice_to_augment)
                    del peaks_[peaks_.index(choice_to_augment)] # don't apply augmentation to same peak more than once
                    for channel in CHANNELS:
                        chosen_factors[peak].append([])
                        factorss[peak].append([])
                        signn = -1 if self.data[:,channel][self.init_peaks_indices[channel][peak]] < self.baseline[channel][0] else 1

                        variance = rd.choice(np.arange(VARIANCE_MIN,VARIANCE_MAX,0.01))

                        chosen_factors[peak][channel].append(chosen_factor)
                        
                        range_ = abs(np.linspace(0, math.sqrt(abs(2*math.log(1/(variance))-math.log(2*math.pi)))*variance, num=len(self.init_peaks_ranges[channel][choice_to_augment]))) # smoothing function reaches 0 at about +- 1.6
                        
                        factors = []
                        augmented__[channel].append(chosen_factor)

                        for g in range(len(self.init_peaks_ranges[channel][choice_to_augment])):
                            factors.append(abs(SMOOTHING_FUNCTION(range_[g],variance,signn))) # apply smoothing function
                        factorss[peak][channel].append(factors)
                        chosen_factor = factors_dict[(choice_to_augment, channel)]
                        bot = min(self.init_peaks_ranges[channel][choice_to_augment])
                        top = max(self.init_peaks_ranges[channel][choice_to_augment])
                        for n,h in enumerate(range(bot, top+1)):
                            if 1==1:
                                if self.init_peaks_seg[channel][choice_to_augment][n] < self.baseline[channel][0]: # if negative peak subtract changes
                                    if self.baseline[channel][0] < 0:
                                        out[:,channel][h] = -abs(self.baseline[channel][0] - self.init_peaks_seg[channel][choice_to_augment][n])*chosen_factor*factors[n]+abs((self.baseline[channel][0]-self.init_peaks_seg[channel][choice_to_augment][n]))-abs(self.init_peaks_seg[channel][choice_to_augment][n])
                                    else:
                                        out[:,channel][h] = -(abs(self.baseline[channel][0] - self.init_peaks_seg[channel][choice_to_augment][n])*chosen_factor*factors[n]-self.baseline[channel][0])
                                elif self.init_peaks_seg[channel][choice_to_augment][n] > self.baseline[channel][0]: # if positive peak add change
                                    out[:,channel][h] = abs(self.init_peaks_seg[channel][choice_to_augment][n]-self.baseline[channel][0])*chosen_factor*factors[n]+self.baseline[channel][0]
            return out, list(map(lambda x: 1 if x in augmented_ else 0, list(range(NUM_PEAKS)))), augmented__, factorss, chosen_factors
        except Exception as e:
            return self.augment_random(anomaly_type) # min arg error
