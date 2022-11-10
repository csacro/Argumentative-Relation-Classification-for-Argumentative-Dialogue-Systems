import time

import sklearn
from simpletransformers.classification import ClassificationModel


def model_with_simpletransformers(train_df, eval_df, filename: str):
    """
    fine-tunes BERT and evaluates with simpletransformers
    :param train_df: train dataframe with labels 0 and 1
    :param eval_df: evaluation dataframe with labels 0 and 1
    :param filename: filename of dataset without .csv
    """

    # prepare data for simpletransformers bert
    train_df = train_df.rename(columns={'Parent': 'text_a', 'Child': 'text_b', 'Class': 'labels'}).dropna()
    if eval_df is not None:
        eval_df = eval_df.rename(columns={'Parent': 'text_a', 'Child': 'text_b', 'Class': 'labels'}).dropna()

    # config options
    my_args = {
        "output_dir": "model/" + filename + "/bert-" + time.strftime("%Y%m%d-%H%M%S") + "/",
        "cache_dir": "cache/",
        "best_model_dir": "model/" + filename + "/bert-" + time.strftime("%Y%m%d-%H%M%S") + "/best_model/",

        #"fp16": True,
        #"fp16_opt_level": "O1",
        "max_seq_length": 256,
        "train_batch_size": 16,  # 16 or 32
        "eval_batch_size": 16,
        #"gradient_accumulation_steps": 1,
        "num_train_epochs": 5,  # 2 or 3 or 4
        #"weight_decay": 0,
        #"learning_rate": 4e-5,  # 5e-5 or 3e-5 or 2e-5
        #"adam_epsilon": 1e-8,
        #"warmup_ratio": 0.06,
        #"warmup_steps": 0,
        #"max_grad_norm": 1.0,
        "do_lower_case": True,

        #"logging_steps": 50,
        "evaluate_during_training": True,
        #"evaluate_during_training_steps": 2000,
        #"evaluate_during_training_verbose": False,
        #"use_cached_eval_features": False,
        #"save_eval_checkpoints": True,
        #"save_steps": 2000,
        #"no_cache": False,
        "save_model_every_epoch": True,
        #"tensorboard_dir": None,

        "overwrite_output_dir": False,
        #"reprocess_input_data": True,

        #"process_count": cpu_count() - 2 if cpu_count() > 2 else 1
        #"n_gpu": 1,
        #"silent": False,
        #"use_multiprocessing": True,

        "wandb_project": "RelationModel",
        "wandb_kwargs": {"mode": "offline"},

        #"use_early_stopping": False,
        #"early_stopping_consider_epochs": False,
        #"early_stopping_patience": 3,
        #"early_stopping_delta": 0,
        #"early_stopping_metric": "eval_loss",
        #"early_stopping_metric_minimize": True,

        "manual_seed": 17,
        #"encoding": None,
        #"config": {}
    }

    # create bert classifier
    model = ClassificationModel('bert', 'bert-base-uncased',
                                args=my_args, use_cuda=False)

    if eval_df is not None:
        # train model
        model.train_model(train_df, eval_df=eval_df, sklearn_report=sklearn.metrics.classification_report)
        # evaluate model
        model.eval_model(eval_df, sklearn_report=sklearn.metrics.classification_report)
    else:
        # train model
        model.train_model(train_df, eval_df=train_df, sklearn_report=sklearn.metrics.classification_report)
