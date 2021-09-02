'''
videorighter
elearning studies inference code
'''

from sklearn.preprocessing import StandardScaler
import label_preprocess
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from sklearn.metrics import confusion_matrix, roc_curve, classification_report, auc
import time
import os
import glob
import warnings
import matplotlib.pyplot as plt



def result(tmp_input, tmp_target):
    result_input = tmp_input.reshape(-1, 150, 109)

    max_tmp_target = tmp_target.max()
    shape = (tmp_target.size, max_tmp_target.astype(int) + 1)

    result_target = np.zeros(shape)
    rows = np.arange(tmp_target.size)

    # 0 = [1, 0](미집중) / 1 = [0, 1](집중)
    result_target[rows.astype(int), tmp_target.astype(int)] = 1

    return result_input, result_target


class Dataloader:

    def __init__(self, csvlist, labeldict):
        self.tmp_input = np.array([])
        self.tmp_target = np.array([])
        self.tmp_idx = np.array([])
        self.csv_list = csvlist
        self.label_dict = labeldict
        self.label_dict_copy = labeldict.copy()

    def data_load(self, bininthefolderpath):

        for file_name in self.csv_list:
            column_names = ['frame', 'face_id', 'timestamp','confidence', 'success', # 2
                            'gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y', # 8
                            'eye_lmk_X_0', 'eye_lmk_X_1', 'eye_lmk_X_2', 'eye_lmk_X_3', 'eye_lmk_X_4', 'eye_lmk_X_5', 'eye_lmk_X_6', 'eye_lmk_X_7', 'eye_lmk_X_28', 'eye_lmk_X_29', 'eye_lmk_X_30', 'eye_lmk_X_31','eye_lmk_X_32', 'eye_lmk_X_33', 'eye_lmk_X_34', 'eye_lmk_X_35', # 16
                            'eye_lmk_Y_0', 'eye_lmk_Y_1', 'eye_lmk_Y_2', 'eye_lmk_Y_3', 'eye_lmk_Y_4', 'eye_lmk_Y_5', 'eye_lmk_Y_6', 'eye_lmk_Y_7', 'eye_lmk_Y_28', 'eye_lmk_Y_29', 'eye_lmk_Y_30', 'eye_lmk_Y_31', 'eye_lmk_Y_32', 'eye_lmk_Y_33', 'eye_lmk_Y_34', 'eye_lmk_Y_35', # 16
                            'eye_lmk_Z_0', 'eye_lmk_Z_1', 'eye_lmk_Z_2', 'eye_lmk_Z_3', 'eye_lmk_Z_4', 'eye_lmk_Z_5', 'eye_lmk_Z_6', 'eye_lmk_Z_7', 'eye_lmk_Z_28', 'eye_lmk_Z_29', 'eye_lmk_Z_30', 'eye_lmk_Z_31', 'eye_lmk_Z_32', 'eye_lmk_Z_33', 'eye_lmk_Z_34', 'eye_lmk_Z_35', # 16
                            'X_0', 'X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8', 'X_9', 'X_10', 'X_11', 'X_12', 'X_13', 'X_14', 'X_15', 'X_16',
                            'Y_0', 'Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5', 'Y_6', 'Y_7', 'Y_8', 'Y_9', 'Y_10', 'Y_11', 'Y_12', 'Y_13', 'Y_14', 'Y_15', 'Y_16',
                            'Z_0', 'Z_1', 'Z_2', 'Z_3', 'Z_4', 'Z_5', 'Z_6', 'Z_7', 'Z_8', 'Z_9', 'Z_10', 'Z_11', 'Z_12', 'Z_13', 'Z_14', 'Z_15', 'Z_16']

            data = pd.read_csv(f'{bininthefolderpath}/output_csv/{file_name}.csv')
            data = data[column_names]
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

            # 시간대 리스트 생성
            time_list = self.label_dict['time']
            # 파일명에 해당하는 레이블 리스트 생성
            label_list = self.label_dict[f'{file_name}']
            # 인덱스 리스트 생성
            idx = [f'{file_name}' + '_' + time_list[i] for i, label in enumerate(label_list)]
            # 위 상태에서 레이블, 인덱스 부여
            data['label'] = np.select(condition_list, self.label_dict_copy[f'{file_name}'])
            data['idx'] = np.select(condition_list, idx)

            idx_nm_expt_6 = data[data['label'] == '6'].index
            data_expt_6 = data.drop(idx_nm_expt_6, axis=0)

            for i in condition_list:
                sample_data = data_expt_6[i]

                if sample_data.empty:
                    continue

                # 109 features
                sample_input = sample_data[column_names[3:]]

                trans = StandardScaler()

                input_data = trans.fit_transform(sample_input)
                target_data = sample_data['label'].astype(int)
                idx_data = sample_data['idx'].astype(str)

                self.tmp_input = np.append(self.tmp_input, np.pad(input_data, ((0, 150 - len(input_data)), (0, 0)), 'constant'))
                self.tmp_target = np.append(self.tmp_target, target_data.unique()[0])
                self.tmp_idx = np.append(self.tmp_idx, idx_data.unique()[0])

        return self.tmp_input, self.tmp_target, self.tmp_idx

def draw_plot(input, tmp_idx, idx):
    idx_num = list(tmp_idx).index(idx)
    ts = input[idx_num]

    face_arr = np.array([list(ts[:, 58].reshape(150,)),
                         list(ts[:, 66].reshape(150,)),
                         list(ts[:, 74].reshape(150,)),
                         list(ts[:, 75].reshape(150,)),
                         list(ts[:, 83].reshape(150,)),
                         list(ts[:, 91].reshape(150,)),
                         list(ts[:, 92].reshape(150,)),
                         list(ts[:, 100].reshape(150,)),
                         list(ts[:, 108].reshape(150,))
                         ]).T

    eye_df = pd.DataFrame(ts[:, 2:10], columns=['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y',])
    face_df = pd.DataFrame(face_arr, columns=['X_0', 'X_8', 'X_16', 'Y_0', 'Y_8', 'Y_16', 'Z_0', 'Z_8', 'Z_16'])

    eye_df.plot()
    plt.title("Eye Gaze Plot " + f"{idx[-7:]} ~ {str(int(idx[-2:]) + 5).zfill(2)}", fontsize=12)
    plt.ylim(-7, 7)
    plt.show()

    face_df.plot()
    plt.title("Face Direction Plot " + f"{idx[-7:]} ~ {str(int(idx[-2:]) + 5).zfill(2)}", fontsize=12)
    plt.ylim(-7, 7)
    plt.show()


def main():
    start = time.time()
    warnings.filterwarnings("ignore")

    csv_list = ['78_A', '78_B']
    label_dict = label_preprocess.labeling('./label_100.csv')

    dataloader = Dataloader(csv_list, label_dict)
    tmp_input, tmp_target, tmp_idx = dataloader.data_load('/Users/oldman')
    X_infer, y_infer = result(tmp_input, tmp_target)

    MODEL_SAVE_FOLDER_PATH = './gru_models_1_2_300_256_0.1_0.0001/'
    # load best model
    best_model_path = sorted(glob.glob(MODEL_SAVE_FOLDER_PATH + '*.hdf5'), key=os.path.getctime)
    best_model = load_model(best_model_path[-1])

    # evaluate model
    _, accuracy = best_model.evaluate(X_infer, y_infer, batch_size=256, verbose=1)

    # confusion matrix
    y_pred = best_model.predict(X_infer)
    cm = confusion_matrix(y_infer.argmax(axis=1), y_pred.argmax(axis=1))
    y_pred_class = (y_pred > 0.5)
    report = classification_report(y_infer, y_pred_class, target_names=['uncon', 'con'])
    uncon_recall = cm[0][0] / (cm[0][0] + cm[1][0])
    uncon_precision = cm[0][0] / (cm[0][0] + cm[0][1])
    con_recall = cm[1][1] / (cm[0][1] + cm[1][1])
    con_precision = cm[1][1] / (cm[1][0] + cm[1][1])
    y_infer_class = (y_infer > 0.5)[:, 1]
    y_pred_prob = y_pred[:, 1]
    print(cm)
    print(report)
    print(best_model.summary())
    print(f"GRU execution \n"
          f"confusion matrix: \n {cm} \n"
          f"accuracy: \n {accuracy} \n"
          f"uncon reall: {uncon_recall} \n"
          f"uncon precision: {uncon_precision} \n"
          f"con recall: {con_recall} \n"
          f"con precision: {con_precision} \n"
          f"classification report: \n {report} \n")
    merge = np.vstack((tmp_idx, y_infer_class, y_pred_class[:, 1], y_pred_prob)).T
    result_df = pd.DataFrame(merge)
    result_df.columns = ['idx', 'GT', 'pred', 'con_prob']

    draw_plot(X_infer, tmp_idx, '78_A_0:03:00')
    draw_plot(X_infer, tmp_idx, '78_A_0:09:55')
    draw_plot(X_infer, tmp_idx, '78_A_0:16:50')
    draw_plot(X_infer, tmp_idx, '78_A_0:06:15')

    result_df.to_csv('infer_result.csv')

    print('Total time: ', time.time() - start)


if __name__ == '__main__':
    main()