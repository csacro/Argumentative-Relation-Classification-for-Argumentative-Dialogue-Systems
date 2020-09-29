import logging

import relation_processing.model_bert as bert
import relation_processing.model_sklearnSVM as sklearnSVM


class RelationProcessor:
    def __init__(self, change_model: bool, change_approach: bool):
        self.__change_model = change_model    # default bert

    def generate_relation_matrix(self, arguments):
        logging.info("generating relation matrix")

        # prob matrix
        if not self.__change_model:
            logging.info("using BERT")
            prob_matrix = bert.generate_prob_matrix(arguments)
        else:
            logging.info("using SVM")
            prob_matrix = sklearnSVM.generate_prob_matrix(arguments)

