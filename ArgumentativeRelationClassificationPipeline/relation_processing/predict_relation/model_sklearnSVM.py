import logging

import numpy as np
from joblib import load
from simplerepresentations import RepresentationModel


def generate_prob_matrix(arguments):
    representation_model = RepresentationModel(
        model_type='bert',
        model_name='bert-base-uncased',
        batch_size=16,
        max_seq_length=256,
        combination_method='cat',
        last_hidden_to_use=4,
        use_cuda=False
    )

    model = load("relation_processing/predict_relation/model/sklearnSVM/nusvc.joblib")
    num_arguments = len(arguments)
    prob_matrix = np.zeros((num_arguments, num_arguments))
    for rel_from in range(1, num_arguments):
		# starting from 1 as the Major Claim (the 0 argument) has no outgoing relation
        for rel_to in arguments[rel_from].compare_list:
            if rel_from == rel_to:
				# no argument can target itself
                continue
            logging.info("calculating: " + str(rel_from) + "-->" + str(rel_to))
            feature, _ = representation_model(text_a=[arguments[rel_to].sentence.lower()],
                                              text_b=[arguments[rel_from].sentence.lower()])
            rel = model.decision_function(feature)
            logging.debug(rel)
            prob_matrix[rel_to][rel_from] = rel[0]
    return prob_matrix
