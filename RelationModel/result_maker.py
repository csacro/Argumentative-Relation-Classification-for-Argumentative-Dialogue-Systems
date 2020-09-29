general_list = ['RandomForest', 'DecisionTree', 'svc', 'svc-kernellinear', 'nusvc', 'nusvc-kernellinear']
params_svm_list = ['svc_kernellinear', 'svc_kernellinear_C10', 'svc_kernellinear_C100', 'nusvc', 'nusvc_nu.05',
                   'nusvc_nu.005', 'nusvc_kernellinear', 'nusvc_kernellinear_nu.05', 'nusvc_kernellinear_nu.005']

bert_subdirs_batch16 = ['./', 'best_model/', 'checkpoint-198-epoch-1/', 'checkpoint-396-epoch-2/',
                        'checkpoint-594-epoch-3/', 'checkpoint-792-epoch-4/', 'checkpoint-990-epoch-5/']
bert_subdirs_batch32 = ['./', 'best_model/', 'checkpoint-99-epoch-1/', 'checkpoint-198-epoch-2/',
                        'checkpoint-297-epoch-3/', 'checkpoint-396-epoch-4/', 'checkpoint-495-epoch-5/']


def main():
    result_maker_bert()
    result_maker_sklearn()


def result_maker_bert():
    path = 'model/MyDataset_balanced_bert_batchXlearnrateX/'
    directories = ['bert-date-time/', 'bert-date-time/', 'bert-date-time/', 'bert-date-time/', 'bert-date-time/']
    bert_subdirs = bert_subdirs_batch32

    f = open(path + 'result.csv', 'w')
    f.write("epoch, run, eval\n")
    run = 0
    for d in directories:
        bert_dir = path + d
        run += 1
        for i in range(0, len(bert_subdirs)):
            epoch = -i
            if len(bert_subdirs[i]) > 20:
                epoch = bert_subdirs[i][len(bert_subdirs[i]) - 2]
            f.write(get_classifier_accuracy_bert(bert_dir + bert_subdirs[i], int(epoch), run) + '\n')
    f.close()


def get_classifier_accuracy_bert(path: str, epoch: int, run: int) -> str:
    ret = str(epoch) + "," + str(run) + "," + get_eval_accuracy_bert(path + 'eval_results.txt')
    return ret


def get_eval_accuracy_bert(filename: str) -> str:
    return get_accuracy(filename, 9)


def result_maker_sklearn():
    path = 'model/MyDataset_balanced/'
    directories = [
        ['Feature#/',
         ['sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/']],
        ['Feature#/',
         ['sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/',
          'sklearn-date-time/']]
    ]

    f = open(path + 'result.csv', 'w')
    f.write("classifier, feature, run, eval, train\n")
    for (result_directory, sklearn_directory) in directories:
        feature_dir = path + result_directory
        run = 0
        for d in sklearn_directory:
            run += 1
            general_dir = feature_dir + d + 'params_svm/'
            params_svm_dir = feature_dir + d + 'general/'

            for c in general_list:
                f.write(get_classifier_accuracy_sklearn(general_dir, c, result_directory[7] + result_directory[8],
                                                        run) + '\n')
            for c in params_svm_list:
                f.write(get_classifier_accuracy_sklearn(params_svm_dir, c, result_directory[7] + result_directory[8],
                                                        run) + '\n')
    f.close()


def get_classifier_accuracy_sklearn(path: str, classifier: str, feature: int, run: int) -> str:
    path += classifier
    ret = classifier + "," + str(feature) + "," + str(run) + "," + get_accuracy_sklearn(
        path + '-evalresult.txt') + "," + get_accuracy_sklearn(path + '-trainresult.txt')
    return ret


def get_accuracy_sklearn(filename: str) -> str:
    return get_accuracy(filename, 5)


def get_accuracy(filename: str, rowofacc: int) -> str:
    try:
        test = open(filename, "r").read()
        return test.splitlines()[rowofacc].split()[1]
    except:
        print(filename)
        return ""


if __name__ == "__main__":
    main()
