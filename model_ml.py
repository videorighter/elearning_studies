'''
videorighter
elearning studies machine learning models
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, roc_auc_score, auc
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pickle
import glob
import time
import label_preprocess
from tqdm import tqdm_gui
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sn
import shutil


def data_load(bininthefolderpath, args):
    column_names = ['frame', 'face_id', 'timestamp','confidence', 'success',
                    'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
                    'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3', 'eye_lmk_X_4', 'eye_lmk_X_5', 'eye_lmk_X_6', 'eye_lmk_X_7', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30', 'eye_lmk_X_31','eye_lmk_X_32', 'eye_lmk_X_33', 'eye_lmk_X_34', 'eye_lmk_X_35',
                    'eye_lmk_Y_0', 'eye_lmk_Y_1', 'eye_lmk_Y_2', 'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5', 'eye_lmk_Y_6', 'eye_lmk_Y_7', 'eye_lmk_Y_28', 'eye_lmk_Y_29', 'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32', 'eye_lmk_Y_33', 'eye_lmk_Y_34', 'eye_lmk_Y_35',
                    'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5', 'eye_lmk_Z_6', 'eye_lmk_Z_7', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32', 'eye_lmk_Z_33', 'eye_lmk_Z_34', 'eye_lmk_Z_35',
                    'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16',
                    'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16',
                    'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15', 'Z_16',
                    'label']

    if args.data_shape == 'all':
        if args.split:
            start = time.time()

            # load CSV file
            csv_list = glob.glob(f'{bininthefolderpath}/output_csv/*.csv')

            # At home
            # csv_list = glob.glob('/media/oldman/새 볼륨/output_csv/*.csv')
            # preprocessing label dictionary
            label_dict = label_preprocess.labeling('./label_100.csv')

            label_dict_copy = label_dict.copy()

            # crate empty dataframe
            tmp = []
            for path in tqdm_gui(csv_list):
                print(path)
                data = pd.read_csv(path)

                data = data[column_names[:-1]]
                name = os.path.basename(path)
                file_name = os.path.splitext(name)[0]

                data = data.drop_duplicates(['frame'])
                condition_list = []
                for i, label in enumerate(label_dict[f'{file_name}']):
                    '''
                    1, 6인 경우 pass
                    2, 3, 5인 경우 미집중(0)
                    4인 경우 
                        이전 5초가 4라면 다음 조건 아니면 0
                        이전 10초~5초가 4라면 다음조건 아니면 0
                        다음 5초가 4라면 1 아니면 0
                        ->
                        첫 10초 미집중(0)
                        마지막 5초 미집중(0)
                        나머지는 집중(1)
                    '''
                    if label == '2' or label == '3' or label == '5':
                        label_dict_copy[f'{file_name}'][i] = '0'
                    elif label == '4':
                        if label == label_dict[f'{file_name}'][i - 1]:  # 이전 5초가 4라면 다음 조건 아니면 0
                            if label == label_dict[f'{file_name}'][i - 2]:  # 이전 10초~5초가 4라면 다음조건 아니면 0
                                if label != label_dict[f'{file_name}'][i + 1]:  # 다음 5초가 4라면 1 아니면 0
                                    label_dict_copy[f'{file_name}'][i] = '0'
                                else:
                                    label_dict_copy[f'{file_name}'][i] = '1'
                            else:
                                label_dict_copy[f'{file_name}'][i] = '0'
                        else:
                            label_dict_copy[f'{file_name}'][i] = '0'
                    else:
                        pass
                    # numpy select를 위한 condition list
                    condition_list.append((data['timestamp'] >= i * 5) & (data['timestamp'] < (i + 1) * 5))

                # condition마다 레이블 적용하여 원 데이터에 레이블 부여
                data['label'] = np.select(condition_list, label_dict_copy[f'{file_name}'])
                # 레이블 6 제거
                idx_nm_expt_6 = data[data['label'] == '6'].index
                data_expt_6 = data.drop(idx_nm_expt_6, axis=0).reset_index()

                # scaling
                trans = StandardScaler()
                tmp_label = data_expt_6.iloc[:, -1:]
                scaled_data = trans.fit_transform(data_expt_6.iloc[:, 4:-1])
                scaled_data = pd.DataFrame(scaled_data, columns=column_names[3:-1])
                scaled_data['label'] = tmp_label
                print(scaled_data.shape)

                # 빈 데이터프레임에 concat
                tmp.append(scaled_data)
            result = pd.concat(tmp, ignore_index=True)
            print(result.shape)

            X_train, X_test, y_train, y_test = train_test_split(result.iloc[:, :-1],
                                                                result.iloc[:, -1],
                                                                test_size=0.1,
                                                                train_size=0.9,
                                                                random_state=200,
                                                                shuffle=True,
                                                                stratify=result.iloc[:, -1])

            X_train = pd.DataFrame(X_train)
            y_train = pd.DataFrame(y_train)
            X_test = pd.DataFrame(X_test)
            y_test = pd.DataFrame(y_test)

            with open('ml_all_X_train.bin', 'wb') as file:
                pickle.dump(X_train, file)
            with open('ml_all_y_train.bin', 'wb') as file:
                pickle.dump(y_train, file)
            with open('ml_all_X_test.bin', 'wb') as file:
                pickle.dump(X_test, file)
            with open('ml_all_y_test.bin', 'wb') as file:
                pickle.dump(y_test, file)

            print("data load time: ", time.time() - start)

        elif not args.split:
            with open('./ml_all_X_train.bin', 'rb') as f:
                X_train = pickle.load(f)
            with open('./ml_all_X_test.bin', 'rb') as f:
                X_test = pickle.load(f)
            with open('./ml_all_y_train.bin', 'rb') as f:
                y_train = pickle.load(f)
            with open('./ml_all_y_test.bin', 'rb') as f:
                y_test = pickle.load(f)

    elif args.data_shape == 'avg':
        print(args.split)
        if args.split:
            with open('./X_train.bin', 'rb') as f:
                pre_X_train = pickle.load(f)
            with open('./X_val.bin', 'rb') as f:
                pre_X_val = pickle.load(f)
            with open('./X_test.bin', 'rb') as f:
                pre_X_test = pickle.load(f)
            with open('./y_train.bin', 'rb') as f:
                pre_y_train = pickle.load(f)
            with open('./y_val.bin', 'rb') as f:
                pre_y_val = pickle.load(f)
            with open('./y_test.bin', 'rb') as f:
                pre_y_test = pickle.load(f)

            pre_X_train = np.append(pre_X_train, pre_X_val, axis=0)
            X_train = []
            for i in pre_X_train:
                X_train.append(np.average(i, axis=0))
            X_train = pd.DataFrame(X_train, columns=column_names[3:-1])

            X_test = []
            for i in pre_X_test:
                X_test.append(np.average(i, axis=0))
            X_test = pd.DataFrame(X_test, columns=column_names[3:-1])

            pre_y_train = np.append(pre_y_train, pre_y_val, axis=0)
            y_train = pre_y_train[:, 1]
            y_test = pre_y_test[:, 1]

            with open('ml_avg_X_train.bin', 'wb') as file:
                pickle.dump(X_train, file)
            with open('ml_avg_y_train.bin', 'wb') as file:
                pickle.dump(y_train, file)
            with open('ml_avg_X_test.bin', 'wb') as file:
                pickle.dump(X_test, file)
            with open('ml_avg_y_test.bin', 'wb') as file:
                pickle.dump(y_test, file)

            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

        elif not args.split:
            with open('./ml_avg_X_train.bin', 'rb') as f:
                X_train = pickle.load(f)
            with open('./ml_avg_X_test.bin', 'rb') as f:
                X_test = pickle.load(f)
            with open('./ml_avg_y_train.bin', 'rb') as f:
                y_train = pickle.load(f)
            with open('./ml_avg_y_test.bin', 'rb') as f:
                y_test = pickle.load(f)

            print(X_train.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_test.shape)

    return X_train, X_test, y_train, y_test


def model(X_train, X_test, y_train, y_test, args):

    start = time.time()
    if args.model == 'xgb':
        model_name = 'XGBoost Classifier'
        clf = XGBClassifier(n_estimators=300,
                            max_features='sqrt',
                            max_depth=16,
                            min_samples_leaf=24,
                            verbose=2,
                            n_jobs=-1)

    elif args.model == 'svc':
        model_name = 'Support Vector Machine with Bagging Classifier'
        clf = SVC(verbose=True, probability=True)

    elif args.model == 'cat':
        model_name = 'CatBoost Classifier'
        clf = CatBoostClassifier(iterations=300,
                                 max_depth=16,
                                 learning_rate=1,
                                 min_data_in_leaf=24,
                                 thread_count=-1)

    elif args.model == 'lgb':
        model_name = 'LightGBM Classifier'
        clf = LGBMClassifier(n_estimators=300,
                             max_depth=16,
                             learning_rate=1,
                             min_child_samples=24,
                             n_jobs=-1)

    elif args.model == 'rfc':
        model_name = 'Random Forest Classifier'
        clf = RandomForestClassifier(n_estimators=300,
                                     max_depth=16,
                                     min_samples_leaf=24,
                                     min_samples_split=24,
                                     verbose=2,
                                     n_jobs=-1)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    y_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    uncon_recall = cm[0][0]/(cm[0][0]+cm[0][1])
    uncon_precision = cm[0][0]/(cm[0][0]+cm[1][0])
    con_recall = cm[1][1]/(cm[1][0]+cm[1][1])
    con_precision = cm[1][1]/(cm[0][1]+cm[1][1])
    report = classification_report(y_test, y_pred, target_names=['uncon', 'con'])

    print(f"{model_name} confusion matrix: \n", cm)
    print(f"{model_name} unconcentration recall: ", uncon_recall)
    print(f"{model_name} unconcentration precesion: ", uncon_precision)
    print(f"{model_name} concentration recall: ", con_recall)
    print(f"{model_name} concentration precesion: ", con_precision)
    print(f"{model_name} test socre: ", score)
    print(f"{model_name} auc_roc score: ", roc_auc)
    print(f"{model_name} time: ", time.time() - start)

    if os.path.isdir(f"./{model_name}/"):
        try:
            os.rmdir(f"./{model_name}/")
        except OSError:
            shutil.rmtree(f"./{model_name}/")
        os.mkdir(f"./{model_name}/")
    else:
        os.mkdir(f"./{model_name}/")

    if os.path.isfile(f"./{model_name}/" + f"{model_name}.txt"):
        os.remove(f"./{model_name}/" + f"{model_name}.txt")
        with open(f"./{model_name}/" + f"{model_name}.txt", 'w') as f:
            f.write(f"{model_name} \n"
                    f"confusion matrix: \n {cm} \n"
                    f"test accuracy: \n {score} \n"
                    f"uncon reall: {uncon_recall} \n"
                    f"uncon precision: {uncon_precision} \n"
                    f"con recall: {con_recall} \n"
                    f"con precision: {con_precision} \n"
                    f"classification report: \n {report} \n")
    else:
        with open(f"./{model_name}/" + f"{model_name}.txt", 'w') as f:
            f.write(f"{model_name} \n"
                    f"confusion matrix: \n {cm} \n"
                    f"test accuracy: \n {score} \n"
                    f"uncon reall: {uncon_recall} \n"
                    f"uncon precision: {uncon_precision} \n"
                    f"con recall: {con_recall} \n"
                    f"con precision: {con_precision} \n"
                    f"classification report: \n {report} \n")

    return y_prob, cm, model_name


def draw_plot(y_test, y_prob, cm, model_name):
    cm = np.array(cm)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentage = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentage)]
    labels = np.asarray(labels).reshape(2, 2)
    df_cm = pd.DataFrame(cm)

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='random classifier')
    plt.plot(fpr, tpr, lw=2, label=f'average ROC curve (area = {roc_auc:0.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'ROC curve for concentration {model_name}', fontsize=20)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=17)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=17)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(loc="lower right", fontsize=17)
    plt.grid(True)
    plt.savefig(f"./{model_name}/" + f"{model_name}_roc_curve.png")

    plt.figure(figsize=(10, 8))
    plt.title('Confusion matrix for Vanilla RNN classifier', fontsize=20)
    plt.tick_params(axis='both', labelsize=14)
    sn.heatmap(df_cm, annot=labels, fmt='', annot_kws={"size": 20}, cmap='Blues')

    plt.savefig(f"./{model_name}/" + f"{model_name}_cm.png")


def main():
    total_time_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=bool, default=False, help='데이터 로드할 때 스플릿해서 저장할 것인지 여부')
    parser.add_argument('--data_shape', type=str, default='avg',
                        help='전체 데이터를 사용할 것인지, 아니면 150프레임씩 평균낸 데이터를 사용할 것인지 여부: all | avg')
    parser.add_argument('--model', '-m', type=str, default='rfc',
                        help="Choose model: 'rfc', 'svc', 'cat', 'xgb', 'lgb'")
    args = parser.parse_args()
    # for mac
    X_train, X_test, y_train, y_test = data_load('/Users/oldman', args)

    # for ubuntu
    # X_train, X_test, y_train, y_test = data_load('/media/oldman/새 볼륨', args)

    y_prob, cm, model_name = model(X_train, X_test, y_train, y_test, args)
    draw_plot(y_test, y_prob, cm, model_name)

    if os.path.isfile(f"./{model_name}/" + f"{model_name}.txt"):
        with open(f"./{model_name}/" + f"{model_name}.txt", 'a') as f:
            f.write(f"Total running time: {time.time() - total_time_start} \n")
    else:
        with open(f"./{model_name}/" + f"{model_name}.txt", 'w') as f:
            f.write(f"Total running time: {time.time() - total_time_start} \n")


if __name__ == "__main__":
    main()
