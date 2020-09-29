import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold


def prepare_feature_cross_validation(data_feature, data_label, k: int):
    """
    create object for split (a/s and n/ng balanced in train and evaluation), edits class labels
    :param data_feature: input for classifier as array
    :param data_label: expected output of classifier in the order of input data as array
    :param k: number of splits
    :return: split as 2D array shaped (k, 2) with each row [[train_indices][test_indices]], modified class labels
    """

    # object to split data into train and eval
    split_array = split_k_fold_cross_validation_np(data_feature, data_label, k)

    # edit class labels
    data_label = edit_class_label_np(data_label)

    return split_array, data_label


def prepare_data_cross_validation(path: str, tolower: bool, k: int):
    """
    reads in data, creates object for split (a/s and n/ng balanced in train and evaluation), edits class labels
    :param path: string with path for csv data file, <path>.csv
    :param tolower: bool, if true sentences are turned to lower case
    :param k: number of splits
    :return: split as 2D array shaped (k, 2) with each row [[train_indices][test_indices]], dataframe with modified labels
    """

    # read in data
    data_df = read_in_data(path, tolower)

    # object to split data into train and eval
    split_array = split_k_fold_cross_validation_df(data_df, k)

    # edit class labels
    data_df = edit_class_labels_df(data_df)

    return split_array, data_df


def prepare_feature_train_test(data_feature, data_label, train_size: float):
    """
    splits data into train and evaluation (a/s and n/ng balanced in train and evaluation), edits class labels
    :param data_feature: input for classifier as array
    :param data_label: expected output of classifier in the order of input data as array
    :param train_size: float between 0 and 1, percentage of data to be train data
    :return: training feature matrix, train labels array, evaluation feature matrix, evaluation labels array
    """

    # split data into train and eval
    train_feature, train_class, eval_feature, eval_class = split_data_np(data_feature, data_label, train_size)

    # edit class labels
    train_label = edit_class_label_np(train_class)
    eval_label = edit_class_label_np(eval_class)

    return train_feature, train_label, eval_feature, eval_label


def prepare_data_train_test(path: str, train_size: float, tolower: bool):
    """
    reads in data, splits data into train and evaluation (a/s and n/ng balanced in train and evaluation), edits class labels
    :param path: string with path for csv data file, <path>.csv
    :param train_size: float between 0 and 1, percentage of data to be train data
    :param tolower: bool, if true sentences are turned to lower case
    :return: train dataframe, evaluation dataframe with labels 0 and 1
    """

    # read in data
    data_df = read_in_data(path, tolower)

    # split data into train and eval
    train_df, eval_df = split_data_df(data_df, train_size)

    # edit class labels
    train = edit_class_labels_df(train_df)
    eval = edit_class_labels_df(eval_df)

    return train, eval


def read_in_data(path: str, tolower: bool):
    """
    turns data from <path>.csv to dataframe
    :param path: string with path for csv data file, <path>.csv
    :param tolower: bool, if true sentences are turned to lower case
    :return: dataframe containing all data in <path>.csv
    """

    data_df = pd.read_csv(path + '.csv')
    if tolower:
        data_df.loc[:, 'Parent'] = data_df.loc[:, 'Parent'].str.lower()
        data_df.loc[:, 'Child'] = data_df.loc[:, 'Child'].str.lower()
    return data_df


def split_data_df(data_df, train_size: float):
    """
    splits data_df into train dataframe and evaluation dataframe balanced according Class
    :param data_df: dataframe to be split
    :param train_size: float between 0 and 1, percentage of data to be train data
    :return: train dataframe, evaluation dataframe
    """

    data_parentchild = data_df.loc[:, ['Parent', 'Child']].to_numpy()
    data_class = data_df.loc[:, 'Class'].to_numpy()

    train_parentchild, train_class, eval_parentchild, eval_class = split_data_np(data_parentchild, data_class,
                                                                                 train_size)

    train_df = pd.DataFrame(data=train_parentchild, columns=['Parent', 'Child'])
    train_df['Class'] = pd.DataFrame(data=train_class)
    eval_df = pd.DataFrame(data=eval_parentchild, columns=['Parent', 'Child'])
    eval_df['Class'] = pd.DataFrame(data=eval_class)

    return train_df, eval_df


def split_data_np(data_parentchild, data_class, train_size):
    """
    splits data_parentchild with data_class into train_parentchild with train_class and eval_parentchild with eval_class balanced according Class
    :param data_parentchild: input data for classifier as array
    :param data_class: expected output of classifier in the order of input data as array
    :param train_size: float between 0 and 1, percentage of data to be train data
    :return: train parentchild, train class, eval parentchild, eval class
    """

    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=17)
    train_parentchild = []
    eval_parentchild = []
    train_class = []
    eval_class = []
    for train_index, eval_index in sss.split(data_parentchild, data_class):
        train_parentchild, eval_parentchild = data_parentchild[train_index], data_parentchild[eval_index]
        train_class, eval_class = data_class[train_index], data_class[eval_index]

    return train_parentchild, train_class, eval_parentchild, eval_class


def split_k_fold_cross_validation_np(data_parentchild, data_class, k: int):
    """
    splits data_parentchild into sets for k-fold cross validation balanced according class
    :param data_parentchild: input data for classifier as array
    :param data_class: expected output of classifier in the order of input data as array
    :param k: number of splits
    :return: split as 2D array shaped (k, 2) with each row [[train_indices][test_indices]]
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=17)
    return skf.split(data_parentchild, data_class)


def split_k_fold_cross_validation_df(data_df, k: int):
    """
    splits data_parentchild into sets for k-fold cross validation balanced according class
    :param data_df: dataframe to be split
    :param k: number of spilts
    :return: split as 2D array shaped (k, 2) with each row [[train_indices][test_indices]]
    """

    data_parentchild = data_df.loc[:, ['Parent', 'Child']].to_numpy()
    data_class = data_df.loc[:, 'Class'].to_numpy()

    return split_k_fold_cross_validation_np(data_parentchild, data_class, k)


def edit_class_labels_df(data_df):
    """
    replaces labels n and ng with 0 and lables a and s with 1
    :param data_df: data dataframe
    :return: data dataframe with labels 0 and 1
    """

    data_df = data_df.replace({'n': 0, 'ng': 0, 'a': 1, 's': 1})

    return data_df


def edit_class_label_np(data_class):
    """
    replaces labels n and ng with 0 and lables a and s with 1
    :param data_class: labels of data as array
    :return: data class array with labels 0 and 1
    """

    data_class = np.where((data_class == 'n') | (data_class == 'ng'), '0', data_class)
    data_class = np.where((data_class == 'a') | (data_class == 's'), '1', data_class)

    return data_class
