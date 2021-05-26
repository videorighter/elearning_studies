import pickle
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
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
