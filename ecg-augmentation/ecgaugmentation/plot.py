import matplotlib.pyplot as plt
import numpy as np

import scipy.signal as ss
Y_LABELS = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]

def _preproc(record):

    sfreq = 257
    samps = np.copy(record)
    
    # MA filter coefficients for powerline interference
    # averages samples from signal in one period of the powerline
    # interference frequency with a first zero at this frequency.
    b1 = np.ones(int(sfreq / 50)) / 50

    # MA filter coefficients for electromyogram noise
    # averages samples in 28 ms interval with first zero at 35 Hz
    b2 = np.ones(int(sfreq / 35)) / 35

    # Butterworth filter coefficients for BLW suppresion
    normfreq1 = 2*40/sfreq
    blb, bla = ss.butter(7, normfreq1, btype='lowpass', analog=False)
    normfreq2 = 2*9/sfreq
    bhb, bha = ss.butter(7, normfreq2, btype='highpass', analog=False)

    a = [1]

    # Filter out PLI
    filt_samps = ss.filtfilt(b1, a, samps, padtype=None, axis=1)

    # Filter out EMG
    filt_samps = ss.filtfilt(b2, a, filt_samps, padtype=None, axis=1)

    # Filter out BLW
    filt_samps = ss.filtfilt(blb, bla, filt_samps, padtype=None, axis=1)
    filt_samps = ss.filtfilt(bhb, bha, filt_samps, padtype=None, axis=1)

    # Complex lead
    cl, n_leads = [], filt_samps.shape[1]

    for i in range(1, len(filt_samps) - 1):
        val = np.sum(np.abs(filt_samps[i+1] - filt_samps[i-1]))
        cl.append(val)
    cl = 1/n_leads * np.array(cl)

    # MA filter coefficients for magnified noise by differentiation used
    # in synthesis of complex lead.
    # averages samples inl 40 ms interval with first zero at 25 Hz
    b3 = np.ones(int(sfreq / 25)) / 25

    cl = ss.lfilter(b3, a, cl)

    return cl


orig = np.load("orig.npy")
new = np.load("RBBB_0.npy")

plt.rcParams.update({'font.size': 26})
plt.rcParams.update({'figure.figsize': (10,10)})

plt.tight_layout()

for channel in range(12):
    plt.plot(orig[0,:,channel], marker='o', linewidth=1, color="blue")
    plt.plot(np.nanargmax(orig[0,:,channel]), np.nanmax(orig[0,:,channel]), 'yo', markersize=10)
    plt.legend([f"Lead {Y_LABELS[channel]} ECG", "R peak"])
    plt.title("RBBB Heartbeat")
    plt.ylabel("mV", labelpad=2)
    plt.savefig(f"orig_{Y_LABELS[channel]}.png")
    plt.figure()

    plt.plot(new[:,channel], marker='o', linewidth=1, color="red")
    plt.plot(np.nanargmax(new[:,channel]), np.nanmax(new[:,channel]), 'yo', markersize=10)
    plt.legend([f"Lead {Y_LABELS[channel]} ECG", "R peak"])
    plt.title("Augmented RBBB Heartbeat")
    plt.ylabel("mV", labelpad=2)
    plt.savefig(f"aug_{Y_LABELS[channel]}.png")
    plt.figure()

plt.plot(_preproc(new[:]), marker='o', linewidth=1, color="red")
plt.plot(np.nanargmax(_preproc(new[:])), np.nanmax(_preproc(new[:])), 'yo', markersize=10)
plt.legend(["Complex ECG", "R peak"])
plt.title("Augmented RBBB Heartbeat", y=1.08)
plt.savefig("aug_complex.png")
plt.figure()

plt.plot(_preproc(orig[0,:]), marker='o', linewidth=1, color="blue")
plt.plot(np.nanargmax(_preproc(orig[0,:])), np.nanmax(_preproc(orig[0,:])), 'yo', markersize=10)
plt.legend(["Complex ECG", "R peak"])
plt.title("RBBB Heartbeat")
plt.savefig("orig_complex.png")
plt.figure()

plt.show()
