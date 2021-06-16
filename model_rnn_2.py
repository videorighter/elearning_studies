# 202100207 videorigher
# keras modeling

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.initializers import HeNormal
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, auc
import matplotlib.pyplot as plt
import seaborn as sn
import time
import pickle
import os
import glob
import shutil
import argparse
from itertools import cycle


def model_execution(X_train, X_val, X_test, y_train, y_val, y_test, num, args):
    verbose = args.verbose  # 1
    epochs = args.epochs  # 200
    batch_size = args.batch_size  # 256
    n_timesteps = X_train.shape[1]  # 150
    n_features = X_train.shape[2]  # 109
    n_outputs = y_train.shape[1]  # 2
    hidden = args.n_hidden  # 150
    dropout = args.drop_out  # 0.1
    lr = args.lr  # 1e-4
    decay = args.decay  # 1e-6
    device = args.device

    MODEL_SAVE_FOLDER_PATH = f'./rnn_models_2_{args.epochs}_{args.batch_size}_{args.drop_out}_{args.lr}/'

    if not args.plot_only:

        if os.path.isdir(MODEL_SAVE_FOLDER_PATH):
            try:
                os.rmdir(MODEL_SAVE_FOLDER_PATH)
            except OSError:
                shutil.rmtree(MODEL_SAVE_FOLDER_PATH)
            os.mkdir(MODEL_SAVE_FOLDER_PATH)
        else:
            os.mkdir(MODEL_SAVE_FOLDER_PATH)

        filename = MODEL_SAVE_FOLDER_PATH + 'Vanilla_RNN_{epoch:02d}-{val_loss:.4f}_' + f'{num + 1}' + '.hdf5'

        check_point = ModelCheckpoint(filepath=filename, monitor='val_loss',
                                      verbose=verbose, save_best_only=True, mode='min')

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=verbose)

        cb_list = [check_point, early_stopping]
        # cb_list = [check_point]

        nadam = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=decay)

        with tf.device(f'/{device}:0'):
            model = Sequential()
            model.add(Bidirectional(SimpleRNN(hidden,
                                        return_sequences=True,
                                        recurrent_dropout=dropout,
                                        dropout=dropout,
                                        recurrent_initializer="orthogonal",
                                        bias_initializer="zeros"),
                                    input_shape=(n_timesteps, n_features)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
            model.add(Bidirectional(SimpleRNN(hidden,
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

            history = history.history

        # load best model
        best_model_path = sorted(glob.glob(MODEL_SAVE_FOLDER_PATH + '*.hdf5'), key=os.path.getctime)
        best_model = load_model(best_model_path[-1])

        # evaluate model
        _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

        # confusion matrix
        y_pred = best_model.predict(X_test)
        y_pred_class = (y_pred > 0.5)
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        report = classification_report(y_test, y_pred_class, target_names=['uncon', 'con'])
        print(cm)
        print(report)
        print(best_model.summary())

    else:
        # load best model
        best_model_path = sorted(glob.glob(MODEL_SAVE_FOLDER_PATH + '*.hdf5'), key=os.path.getctime)
        best_model = load_model(best_model_path[-1])
        history = best_model.history

        # evaluate model
        _, accuracy = best_model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

        # confusion matrix
        y_pred = best_model.predict(X_test)
        y_pred_class = (y_pred > 0.5)
        cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        report = classification_report(y_test, y_pred_class, target_names=['uncon', 'con'])
        print(cm)
        print(report)
        print(best_model.summary())

    if os.path.isfile(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt"):
        os.remove(MODEL_SAVE_FOLDER_PATH + 'rnn_result.txt')
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'w') as f:
            f.write(f"Vanilla RNN {num + 1}'s execution \n"
                    f"confusion matrix: \n {cm} \n"
                    f"accuracy: \n {accuracy} \n"
                    f"classification report: \n {report} \n"
                    f"history: \n {history} \n")
    else:
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'w') as f:
            f.write(f"Vanilla RNN {num + 1}'s execution"
                    f"confusion matrix: \n {cm} \n"
                    f"accuracy: \n {accuracy} \n"
                    f"classification report: \n {report} \n"
                    f"history: {history} \n")

    return y_pred, cm, accuracy, history


def draw_plot(y_test, y_pred, cm, num, history, args):
    MODEL_SAVE_FOLDER_PATH = f'./rnn_models_2_{args.epochs}_{args.batch_size}_{args.drop_out}_{args.lr}/'
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentage = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentage)]
    labels = np.asarray(labels).reshape(2, 2)
    df_cm = pd.DataFrame(cm, index=['T_uncon', 'T_con'], columns=['P_uncon', 'P_con'])

    if not args.plot_only:
        plt.figure(figsize=(10, 8))
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Accuracy/Loss plot for Vanilla RNN classifier', fontsize=20)
        plt.ylabel('Accuracy/Loss', fontsize=17)
        plt.xlabel('Epochs', fontsize=17)
        plt.tick_params(axis='both', labelsize=14)
        plt.legend(['Train acc', 'Val acc', 'Train loss', 'Val loss'], loc='lower left', fontsize=17)
        plt.grid(True)
        plt.savefig(MODEL_SAVE_FOLDER_PATH + f"rnn_{num + 1}_loss.png")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
    plt.plot(fpr["micro"], tpr["micro"], lw=2, label=f'average ROC curve (area = {roc_auc["micro"]:0.4f})')

    colors = cycle(['darkorange', 'cornflowerblue'])

    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC curve for concentration Vanilla RNN classifier', fontsize=20)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=17)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=17)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(loc="lower right", fontsize=17)
    plt.grid(True)
    plt.savefig(MODEL_SAVE_FOLDER_PATH + f"rnn_{num + 1}_auroc.png")

    plt.figure(figsize=(10, 8))
    plt.title('Confusion matrix for Vanilla RNN classifier', fontsize=20)
    plt.tick_params(axis='both', labelsize=14)
    sn.heatmap(df_cm, annot=labels, fmt='', annot_kws={"size": 20}, cmap='Blues')

    plt.savefig(MODEL_SAVE_FOLDER_PATH + f"rnn_{num + 1}_cm.png")


# summarize scores
def summarize_results(scores, args):
    MODEL_SAVE_FOLDER_PATH = f'./rnn_models_2_{args.epochs}_{args.batch_size}_{args.drop_out}_{args.lr}/'
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    if os.path.isfile(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt"):
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'a') as f:
            f.write('Accuracy: %.3f%% (+/-%.3f) \n' % (m, s))
    else:
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'w') as f:
            f.write('Accuracy: %.3f%% (+/-%.3f) \n' % (m, s))


# run an experiment
def run_experiment(args):
    MODEL_SAVE_FOLDER_PATH = f'./rnn_models_2_{args.epochs}_{args.batch_size}_{args.drop_out}_{args.lr}/'

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
        draw_plot(y_test, y_prob, cm, r, history, args)
        score = score * 100.0
        print(f'>#{r + 1}: {score:.3f}')
        scores.append(score)
        print(f"Run {r + 1}'s time: {time.time() - start}")

        if os.path.isfile(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt"):
            with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'a') as f:
                f.write(f'>#{r + 1}: {score:.3f} \n'
                        f"Run {r + 1}'s time: {time.time() - start} \n")
        else:
            with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'w') as f:
                f.write(f'>#{r + 1}: {score:.3f} \n'
                        f"Run {r + 1}'s time: {time.time() - start} \n")
    # summarize results
    summarize_results(scores, args)


def main():
    # run the experiment

    parser = argparse.ArgumentParser()

    parser.add_argument('--repeats', type=int, default=1, help='Number of experiment')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='Input the size of batch')
    parser.add_argument('--drop_out', type=float, default=.5, help='Probability of dropout')
    parser.add_argument('--lr', type=float, default=.0001, help='Input the lenaring rate')
    parser.add_argument('--n_hidden', type=int, default=150, help='Input the number of hidden node of RNN cells')
    parser.add_argument('--device', type=str, default='gpu', help='Select device to execute')
    parser.add_argument('--decay', type=float, default=1e-6, help='Learning rate decay')
    parser.add_argument('--plot_only', type=bool, default=False, help='If only plot')
    args = parser.parse_args()

    total_start = time.time()
    MODEL_SAVE_FOLDER_PATH = f'./rnn_models_2_{args.epochs}_{args.batch_size}_{args.drop_out}_{args.lr}/'
    run_experiment(args)
    print(f"Total running time: {time.time() - total_start}")

    if os.path.isfile(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt"):
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'a') as f:
            f.write(f"Total running time: {time.time() - total_start} \n")
    else:
        with open(MODEL_SAVE_FOLDER_PATH + "rnn_result.txt", 'w') as f:
            f.write(f"Total running time: {time.time() - total_start} \n")


if __name__ == '__main__':
    main()
