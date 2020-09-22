from dtw import *
import tensorflow as tf
import tensorflow.keras as keras
import itertools
from tqdm import tqdm
import scipy.signal as ss
from pylab import rcParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
import matplotlib.collections as mcoll
import matplotlib.path as mpath
rcParams['figure.figsize'] = 16, 16
rcParams['figure.dpi'] = 300
Y_LABELS = ["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"]
plt.rcParams.update({'font.size': 50})

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def generate_patterns(y_true, y_pred):
    class_names = ["Normal","RBBB","PVC", "FUSION", "APC", "SVPB", "NESC","UNKNOWN", "SVESC"]

    preds = [(n,class_names[i],class_names[y_true[n]]) for n,i in enumerate(y_pred)]
    _names = []
    for pred,true in set(itertools.combinations_with_replacement((class_names), 2)):
        _names.append((pred, true))
        exec(f"{pred}_{true} = []", globals())

    for ind, pred, true in preds:
        exec(f"global {pred}_{true}; {pred}_{true}.append({(ind,pred,true)})",globals())

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

def colorlinea(
    x, y, z=None, cmap=plt.get_cmap('jet'), norm=plt.Normalize(0.0, 1.0),
        linewidth=2, alpha=1.0, ax=1, layer_name="", pre=False):
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap,linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)

    if ax.get_ylim()[1] > 0.8: ax.set_ylim(0,0.000001)
    ax.set_ylim(np.nanmin(y) if np.nanmin(y) < ax.get_ylim()[0] - 0.001 else ax.get_ylim()[0], np.nanmax(y) if np.nanmax(y) > ax.get_ylim()[1] else ax.get_ylim()[1])
    ax.set_xlim(0, len(x) if len(x) > ax.get_xlim()[1] else ax.get_xlim()[1])
    return lc

def cam(channel, model, xs, ys):
  #layers = [x.name for x in model.layers if "add" not in x.name and "batch" not in x.name and "input" not in x.name and "dense" not in x.name and "global" not in x.name]
  layers = ["activation_8"]
  raw = []
  for pred, true in [("Normal", "Normal")]:
    exec(f"tmp = {pred}_{true}", globals())
    print(f"have {len(tmp)} samples of: {pred} _ {true}")
    #np.random.shuffle(tmp)
    avgs = []
    length = 10
    of = 0
    fig, ax = plt.subplots()
    for trial in tqdm(range(of,length+of)):
      ind,a,b = tmp[trial]
      vis_channel = channel
      xx = xs[ind,:,:].reshape((-1, 339, 12))
      yy = model.predict(xx)
      y_idx = np.argmax(yy)
      act_y = ys[ind]
      for layer_name in layers:
          model.get_layer(index=-1).activation = keras.activations.linear
          block = model.get_layer(name=layer_name)
          submodel = keras.models.Model(inputs=[model.inputs], outputs=[block.output, model.output])
          with tf.GradientTape() as tape:
              # outputs of previous convolutional layer, predicted class before softmax for data
              conv_output, predictions = submodel(xx)
              preda = predictions[:, y_idx]

          # derived from conv_output without batch dimension

          activations = conv_output[0]

          grads = tape.gradient(preda, conv_output)[0]

          gate_f = tf.cast(activations > 0, 'float32')
          gate_r = tf.cast(grads > 0, 'float32')
          guided_grads = gate_f * gate_r * grads

          # 1D: axis=0 (# of channels in conv_output)

          weights = tf.reduce_mean(guided_grads, axis=0)

          # 1D: activations.shape[0] (Length of the time series, importance of all channels at timestep t)
          cam = np.ones(activations.shape[0], dtype=np.float32)

          for i, w in enumerate(weights):
              # 1D: activations[:, i] (Length of timeseries, importance for channel i)
              cam += w * activations[:, i]

          # effectively relu
          cam = np.maximum(cam, 0)
          heatmap = np.expand_dims(np.maximum(cam, 0) / np.max(cam), axis=1)
          
          if vis_channel == "complex":
            record = xs[ind,:,:]
            slice_len = record[:,0][record[:,0][::-1] != 0].shape[0]
            for n,xx in enumerate(record[:,0][::-1]):
              if xx == 0.0 and record[:,0][::-1][n+1] != 0.0:
                slice_len = record[:,0][::-1].shape[0] - n
                break
            samps = np.zeros((slice_len - 1, 12))
            for channela in range(1,record.shape[-1]-1):
              samps[:,channela] = record[0:slice_len-1,channela]
            y = _preproc(samps)
          elif vis_channel == "all":
            for channela in range(12):
              y = xs[ind, :, channela]
              record = xs[ind,:,:]
              slice_len = record[:,0][record[:,0][::-1] != 0].shape[0]
              samps = np.zeros((slice_len, 1))
              samps[:,0] = record[0:slice_len,channela]
              y = samps
          else:
            plt.title(f"channel: {Y_LABELS[vis_channel]} type: pred:{pred}, true:{true}")
            plt.ylabel("mV")
            y = xs[ind, :, vis_channel]
            record = xs[ind,:,:]
            slice_len = record[:,0][record[:,0][::-1] != 0].shape[0]
            samps = np.zeros((slice_len, 1))
            samps[:,0] = record[0:slice_len,vis_channel]
            y = samps

          act_y = ys[ind]

          x = np.array(list(range(y.shape[0])))
          z = heatmap[:].flatten()
          enc = preprocessing.MinMaxScaler()
          z = enc.fit_transform(z.reshape((-1,1)))[:].flatten()
          path = mpath.Path(np.column_stack([x, y]))
          verts = path.interpolated(steps=1).vertices
          x, y = verts[:, 0], verts[:, 1]
          en = preprocessing.MinMaxScaler()
          #y = en.fit_transform(y.reshape((-1,1)))[:].flatten()
          raw.append({'x':x, 'y':y, "z":z})

          ## align to first
          data0 = raw[0]
          dx = np.mean(np.diff(data0["x"]))
          shift = (np.argmax(ss.correlate(data0["y"], y)) - len(y)) * dx
          x = x + shift

          lc = colorlinea(x, y, z, cmap=plt.get_cmap('jet'), linewidth=2, ax=ax, pre=True if vis_channel == "complex" else False)
  #plt.colorbar(lc)

if __name__ == "__main__":
    cam("complex")
    """for channel in range(12):
    cam(channel)
    plt.figure()"""
    plt.show()