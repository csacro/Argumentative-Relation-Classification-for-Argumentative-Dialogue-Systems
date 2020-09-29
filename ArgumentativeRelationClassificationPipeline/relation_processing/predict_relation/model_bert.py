import logging

import numpy as np
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel


def generate_prob_matrix(arguments):
	my_args = {
		"max_seq_length": 256,
		"train_batch_size": 16,
		"eval_batch_size": 16,
		"do_lower_case": True,
		"manual_seed": 17
	}
	
	model = ClassificationModel('bert', "relation_processing/predict_relation/model/bert", use_cuda=False, args=my_args)
	num_arguments = len(arguments)
	prob_matrix = np.zeros((num_arguments, num_arguments))
	for rel_from in range(1, num_arguments):
		# starting from 1 as the Major Claim (the 0 argument) has no outgoing relation
		for rel_to in arguments[rel_from].compare_list:
			if rel_from == rel_to:
				# no argument can target itself
				continue
			logging.info("calculating: " + str(rel_from) + "-->" + str(rel_to))
			_, raw_outputs = model.predict([[arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower()]])
			rel = softmax(raw_outputs, axis=1)
			logging.debug(rel)
			prob_matrix[rel_to][rel_from] = rel[0][1]
	return prob_matrix
