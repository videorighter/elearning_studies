# 202100207 videorigher
# keras modeling

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.initializers import HeNormal
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import time
import pickle
import os
import glob
import shutil
import argparse


def model_execution(X_train, X_val, X_test, y_train, y_val, y_test, num, args):
    verbose = args.verbose # 1
    epochs = args.epochs # 200
    batch_size = args.batch_size # 256
    n_timesteps = X_train.shape[1]  # 150
    n_features = X_train.shape[2]  # 109
    n_outputs = y_train.shape[1]  # 2
    hidden = args.n_hidden # 150
    dropout = args.drop_out # 0.1
    lr = args.lr # 1e-4
    decay = args.decay # 1e-6
    device = args.device

    if os.path.isdir('./gru_models/'):
        try:
            os.rmdir('./gru_models/')
        except OSError:
            shutil.rmtree('./gru_models/')
        os.mkdir('./gru_models/')
    else:
        os.mkdir('./gru_models/')

    MODEL_SAVE_FOLDER_PATH = './gru_models/'
    filename = MODEL_SAVE_FOLDER_PATH + 'GRU_{epoch:02d}-{val_loss:.4f}_' + f'{num + 1}' + '.hdf5'

    check_point = ModelCheckpoint(filepath=filename, monitor='val_loss',
                                  verbose=verbose, save_best_only=True, mode='min')

    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=verbose)

    cb_list = [check_point, early_stopping]
    # cb_list = [check_point]

    nadam = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=decay)

    with tf.device(f'/{device}:0'):
        model = Sequential()
        model.add(Bidirectional(GRU(hidden,
                                    recurrent_dropout=dropout,
                                    dropout=dropout,
                                    recurrent_initializer="orthogonal",
                                    bias_initializer="zeros"),
                                input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(hidden,
                        kernel_initializer=HeNormal(),
                        bias_initializer='zeros',
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(n_outputs,
                        kernel_initializer=HeNormal(),
                        bias_initializer='zeros',
                        activation='softmax'))
        model.compile(loss='binary_crossentropy',
                      optimizer=nadam,
                      metrics=['accuracy'])

        # fit network
        history = model.fit(X_train,
                            y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=verbose,
                            callbacks=cb_list)

    # load best model
    best_model_path = sorted(glob.glob('./gru_models/*.hdf5'), key=os.path.getctime)
    best_model = load_model(best_model_path[-1])

    # evaluate model
    _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

    # confusion matrix
    y_pred = model.predict(X_test)
    y_prob = y_pred[:, 1]  # concentration prob.
    y_pred_class = (y_pred > 0.5)
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    report = classification_report(y_test, y_pred_class, target_names=['uncon', 'con'])
    print(cm)
    print(report)
    print(model.summary())

    if os.path.isfile("./gru_models/gru_result.txt"):
        os.remove('./gru_models/gru_result.txt')
        with open("./gru_models/gru_result.txt", 'w') as f:
            f.write(f"GRU {num + 1}'s execution \n"
                    f"confusion matrix: \n {cm} \n"
                    f"accuracy: \n {accuracy} \n"
                    f"classification report: \n {report} \n"
                    f"history: \n {history} \n")
    else:
        with open("./gru_models/gru_result.txt", 'w') as f:
            f.write(f"GRU {num + 1}'s execution"
                    f"confusion matrix: \n {cm} \n"
                    f"accuracy: \n {accuracy} \n"
                    f"classification report: \n {report} \n"
                    f"history: {history} \n")

    return y_prob, cm, accuracy, history


def draw_plot(y_test, y_pred, cm, num, history):
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentage = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentage)]
    labels = np.asarray(labels).reshape(2, 2)
    df_cm = pd.DataFrame(cm, index=['T_uncon', 'T_con'], columns=['P_uncon', 'P_con'])
    fig = plt.figure(figsize=(30, 8))
    fig.suptitle('GRU classifier', fontsize=23)

    ax1 = fig.add_subplot(131)
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('Accuracy/Loss plot for GRU classifier', fontsize=20)
    ax1.set_ylabel('Accuracy/Loss', fontsize=17)
    ax1.set_xlabel('Epochs', fontsize=17)
    ax1.legend(['Train acc', 'Val acc', 'Train loss', 'Val loss'], loc='center right')
    ax1.grid(True)

    ax2 = fig.add_subplot(132)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    ax2.plot(fpr, tpr)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_title('ROC curve for concentration GRU classifier', fontsize=20)
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.grid(True)

    ax3 = fig.add_subplot(133)
    ax3.set_title('Confusion matrix for GRU classifier', fontsize=20)
    ax3.tick_params(axis='both', labelsize=15)
    sn.heatmap(df_cm, annot=labels, fmt='', annot_kws={"size": 20}, cmap='Blues')

    plt.savefig(f"./gru_models/gru_{num + 1}.png")
    plt.show()


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    if os.path.isfile("./gru_models/gru_result.txt"):
        with open("./gru_models/gru_result.txt", 'a') as f:
            f.write('Accuracy: %.3f%% (+/-%.3f) \n' % (m, s))
    else:
        with open("./gru_models/gru_result.txt", 'w') as f:
            f.write('Accuracy: %.3f%% (+/-%.3f) \n' % (m, s))


# run an experiment
def run_experiment(args):
    # load data
    with open('./X_train.bin', 'rb') as f:
        X_train = pickle.load(f)
    with open('./X_val.bin', 'rb') as f:
        X_val = pickle.load(f)
    with open('./X_test.bin', 'rb') as f:
        X_test = pickle.load(f)
    with open('./y_train.bin', 'rb') as f:
        y_train = pickle.load(f)
    with open('./y_val.bin', 'rb') as f:
        y_val = pickle.load(f)
    with open('./y_test.bin', 'rb') as f:
        y_test = pickle.load(f)

    # repeat experiment
    scores = list()
    for r in range(args.repeats):
        start = time.time()
        y_prob, cm, score, history = model_execution(X_train, X_val, X_test, y_train, y_val, y_test, r, args)
        draw_plot(y_test[:, 1], y_prob, cm, r, history)
        score = score * 100.0
        print(f'>#{r + 1}: {score:.3f}')
        scores.append(score)
        print(f"Run {r + 1}'s time: {time.time() - start}")

        if os.path.isfile("./gru_models/gru_result.txt"):
            with open("./gru_models/gru_result.txt", 'a') as f:
                f.write(f'>#{r + 1}: {score:.3f} \n'
                        f"Run {r + 1}'s time: {time.time() - start} \n")
        else:
            with open("./gru_models/gru_result.txt", 'w') as f:
                f.write(f'>#{r + 1}: {score:.3f} \n'
                        f"Run {r + 1}'s time: {time.time() - start} \n")
    # summarize results
    summarize_results(scores)

def main():
    # run the experiment

    parser = argparse.ArgumentParser()

    parser.add_argument('--repeats', type=int, default=1, help='Number of experiment')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='Input the size of batch')
    parser.add_argument('--drop_out', type=float, default=.1, help='Probability of dropout')
    parser.add_argument('--lr', type=float, default=.0001, help='Input the lenaring rate')
    parser.add_argument('--n_hidden', type=int, default=150, help='Input the number of hidden node of RNN cells')
    parser.add_argument('--device', type=str, default='gru', help='Select device to execute')
    parser.add_argument('--decay', type=float, default=1e-6, help='Weight decay')
    args = parser.parse_args()

    total_start = time.time()

    run_experiment(args)
    print(f"Total running time: {time.time() - total_start}")

    if os.path.isfile("./gru_models/gru_result.txt"):
        with open("./gru_models/gru_result.txt", 'a') as f:
            f.write(f"Total running time: {time.time() - total_start} \n")
    else:
        with open("./gru_models/gru_result.txt", 'w') as f:
            f.write(f"Total running time: {time.time() - total_start} \n")


if __name__ == '__main__':
    main()