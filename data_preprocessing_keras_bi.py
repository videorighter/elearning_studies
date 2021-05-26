# 2021-02-22 videorighter
# data preprocessing(binary)

# At macbook
from elearning_study_keras import label_preprocess
# At home
# import label_preprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import glob
from tqdm import tqdm_notebook
import pickle
import time

warnings.filterwarnings('ignore')


def result(tmp_input, tmp_target):
    result_input = tmp_input.reshape(-1, 150, 109)

    max_tmp_target = tmp_target.max()
    shape = (tmp_target.size, max_tmp_target.astype(int) + 1)

    result_target = np.zeros(shape)
    rows = np.arange(tmp_target.size)

    # 0 = [1, 0](미집중) / 1 = [0, 1](집중)
    result_target[rows.astype(int), tmp_target.astype(int)] = 1

    return result_input, result_target


def split_dataset(input_dir, target_dir, random_state=200):
    with open(input_dir, 'rb') as f:
        input = pickle.load(f)
    with open(target_dir, 'rb') as f:
        target = pickle.load(f)

    X_train, X_pre, y_train, y_pre = train_test_split(input,
                                                      target,
                                                      test_size=0.2,
                                                      random_state=random_state,
                                                      shuffle=True,
                                                      stratify=target)

    X_val, X_test, y_val, y_test = train_test_split(X_pre,
                                                    y_pre,
                                                    test_size=0.5,
                                                    random_state=random_state,
                                                    shuffle=True,
                                                    stratify=y_pre)

    return X_train, X_val, X_test, y_train, y_val, y_test


class Dataloader:

    def __init__(self, csvlist, labeldict):
        self.tmp_input = np.array([])
        self.tmp_target = np.array([])
        self.csv_list = csvlist
        self.label_dict = labeldict
        self.label_dict_copy = labeldict.copy()

    def loop(self):

        for path in tqdm_notebook(self.csv_list):
            print(path)
            data = pd.read_csv(path)

            # 109 features
            data = data[
                ['frame', 'face_id', 'timestamp', 'confidence', 'success', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z',
                 'gaze_1_x',
                 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',
                 'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3', 'eye_lmk_X_4', 'eye_lmk_X_5',
                 'eye_lmk_X_6',
                 'eye_lmk_X_7', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30', 'eye_lmk_X_31', 'eye_lmk_X_32',
                 'eye_lmk_X_33',
                 'eye_lmk_X_34', 'eye_lmk_X_35',
                 'eye_lmk_Y_0', 'eye_lmk_Y_1', 'eye_lmk_Y_2', 'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5',
                 'eye_lmk_Y_6',
                 'eye_lmk_Y_7', 'eye_lmk_Y_28', 'eye_lmk_Y_29', 'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32',
                 'eye_lmk_Y_33',
                 'eye_lmk_Y_34', 'eye_lmk_Y_35',
                 'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5',
                 'eye_lmk_Z_6',
                 'eye_lmk_Z_7', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32',
                 'eye_lmk_Z_33',
                 'eye_lmk_Z_34', 'eye_lmk_Z_35',
                 'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13',
                 'X_14',
                 'X_15', 'X_16',
                 'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13',
                 'Y_14',
                 'Y_15', 'Y_16',
                 'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13',
                 'Z_14',
                 'Z_15', 'Z_16'
                 ]]

            name = os.path.basename(path)
            file_name = os.path.splitext(name)[0]

            data = data.drop_duplicates(['frame'])

            condition_list = []
            for i, label in enumerate(self.label_dict[f'{file_name}']):
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
                    self.label_dict_copy[f'{file_name}'][i] = '0'
                elif label == '4':
                    if label == self.label_dict[f'{file_name}'][i - 1]:  # 이전 5초가 4라면 다음 조건 아니면 0
                        if label == self.label_dict[f'{file_name}'][i - 2]:  # 이전 10초~5초가 4라면 다음조건 아니면 0
                            if label != self.label_dict[f'{file_name}'][i + 1]:  # 다음 5초가 4라면 1 아니면 0
                                self.label_dict_copy[f'{file_name}'][i] = '0'
                            else:
                                self.label_dict_copy[f'{file_name}'][i] = '1'
                        else:
                            self.label_dict_copy[f'{file_name}'][i] = '0'
                    else:
                        self.label_dict_copy[f'{file_name}'][i] = '0'
                else:
                    pass
                # numpy select를 위한 condition list
                condition_list.append((data['timestamp'] >= i * 5) & (data['timestamp'] < (i + 1) * 5))

            # for _ in self.label_dict_copy[f'{file_name}']:
            #     # 위 상태에서 label 부여
            data['label'] = np.select(condition_list, self.label_dict_copy[f'{file_name}'])

            idx_nm_expt_6 = data[data['label'] == '6'].index
            data_expt_6 = data.drop(idx_nm_expt_6, axis=0)

            for i in condition_list:
                sample_data = data_expt_6[i]

                if sample_data.empty:
                    continue

                # 109 features
                sample_input = sample_data[[
                    'confidence', 'success', 'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
                    'gaze_angle_x', 'gaze_angle_y', 'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3',
                    'eye_lmk_X_4',
                    'eye_lmk_X_5', 'eye_lmk_X_6', 'eye_lmk_X_7', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30',
                    'eye_lmk_X_31',
                    'eye_lmk_X_32', 'eye_lmk_X_33', 'eye_lmk_X_34', 'eye_lmk_X_35', 'eye_lmk_Y_0', 'eye_lmk_Y_1',
                    'eye_lmk_Y_2',
                    'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5', 'eye_lmk_Y_6', 'eye_lmk_Y_7', 'eye_lmk_Y_28',
                    'eye_lmk_Y_29',
                    'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32', 'eye_lmk_Y_33', 'eye_lmk_Y_34', 'eye_lmk_Y_35',
                    'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5',
                    'eye_lmk_Z_6',
                    'eye_lmk_Z_7', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32',
                    'eye_lmk_Z_33', 'eye_lmk_Z_34', 'eye_lmk_Z_35',
                    'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12',
                    'X_13',
                    'X_14', 'X_15', 'X_16',
                    'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12',
                    'Y_13',
                    'Y_14', 'Y_15', 'Y_16',
                    'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12',
                    'Z_13',
                    'Z_14', 'Z_15', 'Z_16'
                ]]

                trans = StandardScaler()

                input_data = trans.fit_transform(sample_input)
                target_data = sample_data['label'].astype(int)

                self.tmp_input = np.append(self.tmp_input,
                                           np.pad(input_data, ((0, 150 - len(input_data)), (0, 0)), 'constant'))
                self.tmp_target = np.append(self.tmp_target, target_data.unique()[0])

        return self.tmp_input, self.tmp_target


if __name__ == '__main__':

    start = time.time()

    # At macbook
    csv_list = glob.glob('/Users/oldman/output_csv/*.csv')

    # At home
    # csv_list = glob.glob('/media/oldman/새 볼륨/output_csv/*.csv')
    label_dict = label_preprocess.labeling('./label_100.csv')

    label_dict_copy = label_dict.copy()

    dataloader = Dataloader(csvlist=csv_list, labeldict=label_dict_copy)
    tmp_input, tmp_target = dataloader.loop()

    result_input, result_target = result(tmp_input=tmp_input,
                                         tmp_target=tmp_target)

    # dump standardized input pickle file
    with open('pre_result_input_bi.bin', 'wb') as file:
        pickle.dump(result_input, file)

    # dump standardized traget pickle file
    with open('pre_result_target_bi.bin', 'wb') as file:
        pickle.dump(result_target, file)

    # data split and dump pickle file
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset('./pre_result_input_bi.bin',
                                                                   './pre_result_target_bi.bin')

    with open('X_train.bin', 'wb') as file:
        pickle.dump(X_train, file)

    with open('X_val.bin', 'wb') as file:
        pickle.dump(X_val, file)

    with open('X_test.bin', 'wb') as file:
        pickle.dump(X_test, file)

    with open('y_train.bin', 'wb') as file:
        pickle.dump(y_train, file)

    with open('y_val.bin', 'wb') as file:
        pickle.dump(y_val, file)

    with open('y_test.bin', 'wb') as file:
        pickle.dump(y_test, file)

    print("Execution time: ", start - time.time())
