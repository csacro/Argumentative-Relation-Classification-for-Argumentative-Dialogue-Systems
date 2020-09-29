import numpy as np

import dataset


def main():
    # values
    datadir = "datasets/"
    filename = "MyDataset_balanced"
    train_size = 0.8
    k = 5

    # read in data
    data_df = dataset.read_in_data(datadir + filename, True)

    # generate statistics from data
    data_statistic, data_label = make_statistics(data_df)

    fold = 0
    f = open(datadir + "statistics/" + filename + ".csv", 'w')
    f.write("input size, fold, k, train/eval\n")

    # no fold
    # split into train and eval
    train_statistics, train_label, eval_statistics, eval_label = dataset.prepare_feature_train_test(data_statistic,
                                                                                                    data_label,
                                                                                                    train_size)
    f.write(output_statistics(train_statistics, eval_statistics, fold, k))

    # folds
    if k > 0:
        k_fold_split, data_label = dataset.prepare_feature_cross_validation(data_statistic, data_label, k)
        for train_indices, eval_indices in k_fold_split:
            fold += 1
            # split into train and eval
            train_statistics, train_label = data_statistic[train_indices], data_label[train_indices]
            eval_statistics, eval_label = data_statistic[eval_indices], data_label[eval_indices]
            f.write(output_statistics(train_statistics, eval_statistics, fold, k))

    f.close()


def output_statistics(train_statistics, eval_feature, fold: int, k: int):
    ret = output_specified_statistics(train_statistics, fold, k, "train")
    ret += output_specified_statistics(eval_feature, fold, k, "eval")
    return ret


def output_specified_statistics(statistics, fold: int, k: int, spec: str):
    ret = ""
    for val in statistics:
        ret += str(val) + "," + str(fold) + "," + str(k) + "," + spec + "\n"
    return ret


def make_statistics(data_df):
    # prepare data for feature generation
    parentchild = data_df.loc[:, ['Parent', 'Child']].to_numpy()
    label = data_df.loc[:, 'Class'].to_numpy()

    # generate statistics
    statistics = []
    for pair in parentchild:
        statistics.append(len(str(pair[0]).split()) + len(str(pair[1]).split()))

    return np.asarray(statistics), label


if __name__ == "__main__":
    main()
