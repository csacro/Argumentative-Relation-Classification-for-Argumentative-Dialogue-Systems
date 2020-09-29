import util
import logging

import dataset
import features

import classifier_bert
import classifier_sklearn


def main():
    # values
    datadir = "dataset/"
    filename = "MyDataset_balanced"
    train_size = 0.8
    logfile_name = "relationModel_MyDatasetBalanced"

    # init logging
    util.init_logging(logging.DEBUG, logfile_name)

    # sklearn classifiers
    run_sklearn_classifiers(datadir, filename, train_size, k=5)

    # bert fine-tuning
    run_bert_finetuning(datadir, filename, train_size)


def run_sklearn_classifiers(datadir: str, filename: str, train_size: float, k: int = 0, feature_range=range(0, 15), specific_model=None):
    # read in data
    data_df = dataset.read_in_data(datadir + filename, True)

    for counter in feature_range:
        util.init_logging(logging.DEBUG, "relationModel_sklearn_" + str(counter) + "_" + str(k))
        # generate feature vectors from data
        data_feature, data_label = features.make_featurevector(data_df,
                                                               'features/' + filename + '/bert/sentence_pair.joblib',
                                                               'features/' + filename + '/bert/sentence.joblib',
                                                               counter)

        if k > 0:
            k_fold_split, data_label = dataset.prepare_feature_cross_validation(data_feature, data_label, k)
            for train_indices, eval_indices in k_fold_split:
                # split feature vectors into train and eval
                train_feature, train_label = data_feature[train_indices], data_label[train_indices]
                eval_feature, eval_label = data_feature[eval_indices], data_label[eval_indices]
                # run sklearn classifier bench (training and evaluation)
                classifier_sklearn.models_with_sklearn(train_feature, train_label, eval_feature, eval_label,
                                                       filename.rsplit('.')[0], specific_model)
        elif train_size == 1.0:
            # run sklearn classifier bench (100% training)
            data_label = dataset.edit_class_label_np(data_label)
            classifier_sklearn.models_with_sklearn(data_feature, data_label, None, None, filename.rsplit('.')[0],
                                                   specific_model)
        else:
            # split feature vectors into train and eval
            train_feature, train_label, eval_feature, eval_label = dataset.prepare_feature_train_test(data_feature,
                                                                                                      data_label,
                                                                                                      train_size)
            # run sklearn classifier bench (training and evaluation)
            classifier_sklearn.models_with_sklearn(train_feature, train_label, eval_feature, eval_label,
                                                   filename.rsplit('.')[0], specific_model)


def run_bert_finetuning(datadir: str, filename: str, train_size: float, k: int = 0):
    util.init_logging(logging.DEBUG, "relationModel_bert_" + str(k))
    if k > 0:
        k_fold_split, data_df = dataset.prepare_data_cross_validation(datadir + filename, True, k)
        for train_indices, eval_indices in k_fold_split:
            # split feature vectors into train and eval
            train_df = data_df.loc[train_indices, :]
            eval_df = data_df.loc[eval_indices, :]

            # run BERT fine-tuning (training and evaluation)
            classifier_bert.model_with_simpletransformers(train_df, eval_df, filename.rsplit('.')[0])
    elif train_size == 1.0:
        # run BERT fine-tuning (100% training)
        train_df = dataset.edit_class_labels_df(dataset.read_in_data(datadir + filename, True))
        classifier_bert.model_with_simpletransformers(train_df, None, filename.rsplit('.')[0])
    else:
        # split data into train and eval
        train_df, eval_df = dataset.prepare_data_train_test(datadir + filename, train_size, True)
        # run BERT fine-tuning (training and evaluation)
        classifier_bert.model_with_simpletransformers(train_df, eval_df, filename.rsplit('.')[0])


if __name__ == "__main__":
    main()
