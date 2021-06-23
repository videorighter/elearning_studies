import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import glob
import time
import label_preprocess
from data_preprocessing_keras_bi import Dataloader
from tqdm import tqdm_gui
import os
import argparse


def data_load(bininthefolderpath):
    start = time.time()

    # load CSV file
    csv_list = glob.glob(f'{bininthefolderpath}/output_csv/*.csv')

    # At home
    # csv_list = glob.glob('/media/oldman/새 볼륨/output_csv/*.csv')
    # preprocessing label dictionary
    label_dict = label_preprocess.labeling('./label_100.csv')

    label_dict_copy = label_dict.copy()
    column_names = ['frame', 'face_id', 'timestamp',
                    'confidence', 'success', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
                    'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3', 'eye_lmk_X_4', 'eye_lmk_X_5','eye_lmk_X_6', 'eye_lmk_X_7', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30', 'eye_lmk_X_31', 'eye_lmk_X_32', 'eye_lmk_X_33', 'eye_lmk_X_34', 'eye_lmk_X_35',
                    'eye_lmk_Y_0', 'eye_lmk_Y_1', 'eye_lmk_Y_2', 'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5', 'eye_lmk_Y_6', 'eye_lmk_Y_7', 'eye_lmk_Y_28', 'eye_lmk_Y_29', 'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32', 'eye_lmk_Y_33', 'eye_lmk_Y_34', 'eye_lmk_Y_35',
                    'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5', 'eye_lmk_Z_6', 'eye_lmk_Z_7', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32', 'eye_lmk_Z_33', 'eye_lmk_Z_34', 'eye_lmk_Z_35',
                    'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16',
                    'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16',
                    'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15', 'Z_16',
                    'label']
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

    with open('svm_X_train.bin', 'wb') as file:
        pickle.dump(X_train, file)

    # dump standardized traget pickle file
    with open('svm_y_train.bin', 'wb') as file:
        pickle.dump(y_train, file)

    with open('svm_X_test.bin', 'wb') as file:
        pickle.dump(X_test, file)

    # dump standardized traget pickle file
    with open('svm_y_test.bin', 'wb') as file:
        pickle.dump(y_test, file)

    print("data load time: ", time.time() - start)

    return X_train, X_test, y_train, y_test


def model(X_train, X_test, y_train, y_test):
    start = time.time()

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    cf = confusion_matrix(y_test, pred)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print(y_test.count(0))
    print(y_test.count(1))

    print("Supprot vector machine confusion matrix: \n", cf)
    print("Supprot vector machine socre: ", clf.score(X_test, y_test))
    print("Support vector machine auc_roc score: ", roc_auc_score(y_test, pred))
    print("SVM time: ", time.time() - start)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=bool, default=False, help='Data load and split and save')
    args = parser.parse_args()

    if args.split:
        # mac
        X_train, X_test, y_train, y_test = data_load('/Users/oldman')
        # desktop
        # X_train, X_test, y_train, y_test = data_load('/media/oldman/새 볼륨')
    else:
        with open('./svm_X_train.bin', 'rb') as f:
            X_train = pickle.load(f)
        with open('./svm_X_test.bin', 'rb') as f:
            X_test = pickle.load(f)
        with open('./svm_y_train.bin', 'rb') as f:
            y_train = pickle.load(f)
        with open('./svm_y_test.bin', 'rb') as f:
            y_test = pickle.load(f)

    model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
