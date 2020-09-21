
import tensorflow as tf

import numpy as np
from tensorflow.keras.callbacks import Callback
import keras.backend as K
import tensorflow as tf
import gc
import time
import os
import matplotlib
import datetime
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,classification_report,plot_confusion_matrix
from sklearn.model_selection import KFold # kfold cross validation
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import matplotlib
from glob import glob
import datetime
import gc
import sklearn
import sklearn.preprocessing as preprocessing

import tensorflow.keras as keras 

tf.compat.v1.enable_eager_execution()
K.clear_session()
tf.compat.v1.reset_default_graph()

def main(training: str, validation: str, splits: int = 10):
    g = h5py.File(validation, 'r')
    f = h5py.File(training, 'r')

    enc = preprocessing.OneHotEncoder()

    x_tmp = np.array(g["REALS"])
    y_tmp = np.array(g["REALS_ann"])

    val_dataset_x, val_dataset_y = x_tmp, enc.fit_transform(y_tmp.reshape(-1,1)).toarray()

    x = g['REALS']
    y = g['REALS_ann']

    if len(f.keys()) > 2:
        # have multiple dfs in h5 file -> need to combine
        X, Y = combine(f)
    else:
        X, Y = f["X"], f["Y"]

    # REMOVE UNKNOWN CLASS
    X = X[(Y != 13).flatten()]
    Y = Y[(Y != 13).flatten()]

    # split into k folds
    kfold = StratifiedKFold(n_splits=splits, shuffle=True)

    # train
    counter = 0
    hists = []
    for train_index, test_index in kfold.split(X, Y):
        # model
        model, callbacks = resnet(f"raw_training_{counter}.log", classes=8, init_lr=0.1, tensorboard_dir="logs/")
        # split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_train = enc.transform(y_train).toarray()
        y_test = enc.transform(y_test).toarray()
        # fit
        hist = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), callbacks=callbacks,shuffle=True, batch_size=256, epochs=150)

        counter += 1
        hists.append(hist)

    ## TEST DATA
    model = resnet(f"raw_training_final.log", classes=8, init_lr=0.1, tensorboard_dir="logs/")
    preds = model.predict(val_dataset_x)

    model.save("final_fold_model.hdf5")


    """y = enc.transform(np.array(y)).toarray()
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(np.array(x),np.array(y),test_size=0.2)

    # save the split
    np.save("raw_x_train.npy", x_train)
    np.save("raw_y_train.npy", y_train)
    np.save("raw_x_test.npy", x_test)
    np.save("raw_y_test.npy", y_test)

    model, callbacks = resnet(f"raw_training.log", classes=9, init_lr=0.1, tensorboard_dir="logs/")

    y_test = enc.transform(y_test).toarray()
    y_train = enc.transform(y_train).toarray()

    hist = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), callbacks=callbacks,shuffle=True, batch_size=256, epochs=150)"""

    

def combine(d: h5py.File, enc: preprocessing.OneHotEncoder = None) -> (np.ndarray, np.ndarray):

    if enc is not None:
        out_X, out_Y = [], []
        dfs = list(d.keys())
        for df in dfs:
            if '_ann' not in df:
                anns = d[df+"_ann"]
                for n,xx in enumerate(d[df]):
                    out_X.append(xx)
                    out_Y.append(enc.transform(anns[n].reshape((-1,1))).toarray())
    else:
        out_X, out_Y = [], []
        dfs = list(d.keys())
        for df in dfs:
            if '_ann' not in df:
                anns = d[df+"_ann"]
                for n,xx in enumerate(d[df]):
                    out_X.append(xx)
                    out_Y.append(anns[n])
    return (np.array(out_X), np.array(out_Y))


def resnet(training_log_path: str, classes: int, init_lr: float, tensorboard_dir: str, n_feature_maps: int=64):
    
    class_names = ["Normal","RBBB","PVC", "FUSION", "APC", "SVPB", "NESC","UNKNOWN", "SVESC"]
    def single_class_accuracy(interesting_class_id):
        # compute the accuracy of a single class
        def fn(y_true, y_pred):
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        return fn

    class CustomCallback(Callback):
        def __init__(self):
            self.dat = []
            self.seen = 0
            self.epoch = 1
        def on_batch_end(self,batch,logs={}):
            self.dat = logs

        def on_epoch_end(self,batch,logs={}):
            np.save(f"epoch_dat_{self.epoch}.npy", self.dat, allow_pickle=True)
            self.epoch += 1
            gc.collect() # try to clear up some memory??


    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    input_layer = keras.layers.Input((339,12),dtype='float32')

    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)
    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)
    # no need to expand channels because they are equal
    shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = keras.layers.Dense(classes, activation='softmax')(gap_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def my_sparse_categorical_crossentropy(y_true, y_pred): # need from logits TRUE override default
        return K.categorical_crossentropy(y_true, y_pred, from_logits=False)

    class spec(tf.keras.metrics.Metric):

        def __init__(self,name, typea,**kwargs):
            super(spec, self).__init__(name=name, **kwargs)
            self.typea = typea

        def update_state(self, y_true, y_pred,sample_weight=None):
            class_id_true = K.argmax(y_true, axis=-1) # one-hot -> int
            class_id_preds = K.argmax(y_pred, axis=-1)
            recall_mask = K.cast(K.equal(class_id_true, self.typea), 'int32')
            class_recall_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * recall_mask
            self.tmp = K.sum(class_recall_tensor) / K.maximum(K.sum(recall_mask), 1)
            tf.cond(K.cast(K.equal(K.cast(self.tmp, tf.float64), tf.constant(0.0, dtype=tf.float64)),dtype=tf.bool),true_fn=self.if_true, false_fn=self.if_false)

        def result(self):
            return tf.math.subtract(tf.constant(1.0, dtype=tf.float64), self.recall)

        def if_false(self):
            self.recall = self.tmp

    class sens(tf.keras.metrics.Metric):
    # sensitivity metric
        def __init__(self,name, typea,**kwargs):
            super(sens, self).__init__(name=name, **kwargs)
            self.typea = typea

        def update_state(self, y_true, y_pred,sample_weight=None):
            class_id_true = K.argmax(y_true, axis=-1) # one-hot -> int
            class_id_preds = K.argmax(y_pred, axis=-1)
            recall_mask = K.cast(K.equal(class_id_true, self.typea), 'int32')
            class_recall_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * recall_mask
            self.tmp = K.sum(class_recall_tensor) / K.maximum(K.sum(recall_mask), 1)
            tf.cond(K.cast(K.equal(K.cast(self.tmp, tf.float64), tf.constant(0.0, dtype=tf.float64)),dtype=tf.bool),true_fn=self.if_true, false_fn=self.if_false) # only update recall if not = 0, pesty that it resets if self.typea not in the current batch

        def if_false(self):
            self.recall = self.tmp


    model.compile(loss=my_sparse_categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=init_lr),metrics=[m for m in METRICS], weighted_metrics=["accuracy"])

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=0.0000001)

    file_path = f'raw_train_aug_{str(datetime.datetime.now())}.hdf5'
    cc = CustomCallback()

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-4)

    csv = tf.keras.callbacks.CSVLogger(training_log_path, separator=",", append=False)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(f"{tensorboard_dir}/model"+str(datetime.datetime.now()), histogram_freq=1, update_freq="batch")

    callbacksa = [reduce_lr, model_checkpoint, callback, cc, csv, tensorboard_callback]

    return (model, callbacksa)

if __name__ == "__main__":
	main("training", "reals")
