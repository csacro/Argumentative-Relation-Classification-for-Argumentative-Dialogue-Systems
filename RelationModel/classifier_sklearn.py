import logging

import os
import sys
import time
from joblib import dump

from sklearn import svm, ensemble, tree
from sklearn import metrics


def models_with_sklearn(train_feature, train_label, eval_feature, eval_label, filename: str, specific_model=None):
    """
    trains and evaluates all kind of sklearn classifiers
    :param train_feature: numpy array of training data as feature vectors
    :param train_label: numpy array of labels (0 or 1) fitting to order of train_features
    :param eval_feature: numpy array of evaluation data as feature vectors
    :param eval_label: numpy array of labels (0 or 1) fitting to order of eval_features
    :param filename: filename of dataset without .csv
    :param specific_model: if not none only specified model is trained (array of sklearn model and name as str)
    """

    model_dir = 'model/' + filename + '/sklearn-' + time.strftime("%Y%m%d-%H%M%S")
    os.mkdir(model_dir)

    if specific_model is None:
        models_params_svm(train_feature, train_label, eval_feature, eval_label, model_dir)
        models_general(train_feature, train_label, eval_feature, eval_label, model_dir)
    else:
        model_dir = model_dir + '/specific_model/'
        os.mkdir(model_dir)
        model = specific_model[0]
        name = specific_model[1]
        train_and_evaluate_sklearn_classifier(model, name, model_dir, train_feature, train_label, eval_feature,
                                              eval_label)


def models_params_svm(train_feature, train_label, eval_feature, eval_label, model_dir: str):
    """
    trains svm and evaluates with sklearn
    :param train_feature: numpy array of training data as feature vectors
    :param train_label: numpy array of labels (0 or 1) fitting to order of train_features
    :param eval_feature: numpy array of evaluation data as feature vectors
    :param eval_label: numpy array of labels (0 or 1) fitting to order of eval_features
    :param model_dir: directory with timestamp to save models in
    """

    model_dir = model_dir + '/params_svm/'
    os.mkdir(model_dir)

    params_svm_list = [[svm.SVC(random_state=17, kernel='linear', C=1), 'svc_kernellinear'],
                [svm.SVC(random_state=17, kernel='linear', C=10), 'svc_kernellinear_C10'],
                [svm.SVC(random_state=17, kernel='linear', C=100), 'svc_kernellinear_C100'],
                [svm.NuSVC(random_state=17, nu=0.5), 'nusvc'],
                [svm.NuSVC(random_state=17, nu=0.05), 'nusvc_nu.05'],
                [svm.NuSVC(random_state=17, nu=0.005), 'nusvc_nu.005'],
                [svm.NuSVC(random_state=17, kernel='linear', nu=0.5), 'nusvc_kernellinear'],
                [svm.NuSVC(random_state=17, kernel='linear', nu=0.05), 'nusvc_kernellinear_nu.05'],
                [svm.NuSVC(random_state=17, kernel='linear', nu=0.005), 'nusvc_kernellinear_nu.005']]

    handle_classifier_list(params_svm_list, model_dir, train_feature, train_label, eval_feature, eval_label)


def models_general(train_feature, train_label, eval_feature, eval_label, model_dir: str):
    """
    trains models and evaluates with sklearn
    :param train_feature: numpy array of training data as feature vectors
    :param train_label: numpy array of labels (0 or 1) fitting to order of train_features
    :param eval_feature: numpy array of evaluation data as feature vectors
    :param eval_label: numpy array of labels (0 or 1) fitting to order of eval_features
    :param model_dir: directory with timestamp to save models in
    """

    model_dir = model_dir + '/general/'
    os.mkdir(model_dir)

    general_list = [[ensemble.RandomForestClassifier(random_state=17), 'RandomForest'],
                 [tree.DecisionTreeClassifier(random_state=17), 'DecisionTree'],
                 [svm.SVC(random_state=17), 'svc'],
                 [svm.SVC(random_state=17, kernel='linear'), 'svc_kernellinear'],
                 [svm.NuSVC(random_state=17), 'nusvc'],
                 [svm.NuSVC(random_state=17, kernel='linear'), 'nusvc_kernellinear']]

    handle_classifier_list(general_list, model_dir, train_feature, train_label, eval_feature, eval_label)


def handle_classifier_list(classifer_list, model_dir: str, train_feature, train_label, eval_feature, eval_label):
    """
    trains and evaluates sklearn classifier model
    :param classifer_list: list of classifiers (classifiers containing model and name)
    :param model_dir: directory with timestamp and model type to save evaluation results in
    :param train_feature: numpy array of training data as feature vectors
    :param train_label: numpy array of labels (0 or 1) fitting to order of train_features
    :param eval_feature: numpy array of evaluation data as feature vectors
    :param eval_label: numpy array of labels (0 or 1) fitting to order of eval_features
    """

    for classifier in classifer_list:
        model = classifier[0]
        name = classifier[1]
        train_and_evaluate_sklearn_classifier(model, name, model_dir, train_feature, train_label, eval_feature,
                                              eval_label)


def train_and_evaluate_sklearn_classifier(model, name, model_dir: str, train_feature, train_label, eval_feature,
                                          eval_label):
    """
    trains and evaluates sklearn classifier model
    :param model: sklearn classifier to be evaluated
    :param name: name of the sklearn classifier for saving evaluation result
    :param model_dir: directory with timestamp and model type to save evaluation results in
    :param train_feature: numpy array of training data as feature vectors
    :param train_label: numpy array of labels (0 or 1) fitting to order of train_features
    :param eval_feature: numpy array of evaluation data as feature vectors
    :param eval_label: numpy array of labels (0 or 1) fitting to order of eval_features
    """

    logging.info('train and evaluate: ' + name)
    try:
        # train model
        model.fit(train_feature, train_label)
        # save trained model
        dump(model, model_dir + name + '.joblib')

        # evaluate model
        # train data
        train_output = model.predict(train_feature)
        train_metrics = metrics.classification_report(train_label, train_output)
        dump(train_metrics, model_dir + name + '-trainresult.txt')
        if eval_feature is not None and eval_label is not None:
            # eval data
            eval_output = model.predict(eval_feature)
            eval_metrics = metrics.classification_report(eval_label, eval_output)
            dump(eval_metrics, model_dir + name + '-evalresult.txt')
    except ValueError as err:
        logging.warning(sys.exc_info()[0])
        logging.warning(err)
