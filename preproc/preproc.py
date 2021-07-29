import argparse
import numpy as np
import scipy.signal as ss
import wfdb


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Record name without extension.')
    parser.add_argument('ecg_output', type=str, help='ECG output filename')
    parser.add_argument('ann_output', type=str, help='Annotation output filename')
    return parser


def filter_ecg(ecg, ssf=None, **kwargs):
    """
    Filter the ECG signals to remove baseline wander.

    Parameters
    ----------
    ssf : function
        A Scipy.Signal filter design function.
    **kwargs
        Arguments to pass into ssf.

    Returns
    -------
    fecg_data : ndarray
        An array formatted identically to array returned by self.samples of
        filtered signals.
    """
    fecg_data = np.copy(ecg)
    b = ss.firwin(77, cutoff=[0.02, 0.35], pass_zero=False)

    for i in range(len(fecg_data[0])):
        fecg = ss.filtfilt(b, [1], fecg_data[:, i])
        np.put(fecg_data[:, i], range(fecg_data.shape[0]), fecg)

    return fecg_data

def segment(ecg, timestamps, fill=True):
    """
    Extract segments from the record given a list of segmentation
    timestamps.

    Parameters
    ----------
    ecg : array
        ECG data
    timestamps : list of 2-tuples
        Segmentation timestamps associated for the record.
    fill : bool
        Ensure that the time length is equal length. Missing values in the
        returned array are filled with np.nan.

    Returns
    -------
    ndarray
        An [N, T, V] array representing N segments with T samples each from
        V leads.
    """
    segs = []
    max_seglen = -1
    for i, ts in enumerate(timestamps):
        segment_start, segment_end = (ts[0], ts[1])
        seg = ecg[segment_start:segment_end]
        max_seglen = len(seg) if len(seg) > max_seglen else max_seglen
        segs.append(seg)

    if fill:
        np_segs = np.full((len(segs), max_seglen, ecg.shape[-1]),
                            fill_value=np.nan)
        for i, seg in enumerate(segs):
            np_segs[i, :seg.shape[0], :seg.shape[1]] = seg
    else:
        np_segs = np.array(segs)

    return np_segs

def timestamps(ecg, fs, rpeaks):
    """
    Get time slices of heartbeats from a signal, given a list of R-peak
    locations. Adapted from biosppy.extract_heartbeats.

    Parameters
    ----------
    ecg : array
        The ECG record
    fs : double
        Sampling frequency
    rpeaks : array
        R-peak indices for the associated record.

    Returns
    -------
    list of ints
        The corresponding index of the rpeaks.
    list of 2-tuples of ints
        Time slices of heart beats corresponding to R-peak indices.

    Notes
    -----
    rpeaks and seg_ts may not have the same length when the segment starts
    before or continues after the data domain if match_lists is not set.
    """
    # get heartbeats
    corr_idx = []
    seg_ts = []
    qr_max = 0.08 * fs
    qtc_max = 0.42

    for i, r in enumerate(rpeaks):
        if i == 0:
            rr_prev = r
        else:
            rr_prev = r - rpeaks[i-1]

        if i == len(rpeaks) - 1:
            rr_next = len(ecg) - r
        else:
            rr_next = rpeaks[i+1] - r

        a = int(r - (qr_max + 0.2 * rr_prev + 0.1*fs))
        if a < 0:
            continue
        b = int(r + (1.5 * qtc_max * fs * np.sqrt(rr_next/fs)))
        if b > len(ecg):
            break
        corr_idx.append(i)
        seg_ts.append((a, b))

    return corr_idx, seg_ts


def main():
    args = init_parser().parse_args()
    record = wfdb.io.rdrecord(args.file)
    f_ecg = filter_ecg(record.p_signal)

    annotations = wfdb.io.rdann(args.file, 'atr')

    rpeaks = annotations.sample
    corr_idx, seg_ts = timestamps(f_ecg, record.fs, rpeaks)
    segs = segment(f_ecg, seg_ts)

    assert segs.shape[0] == len(annotations.symbol)

    np.save(args.ecg_output, segs)
    np.save(args.ann_output, annotations.symbol)


if __name__ == "__main__":
    main()
