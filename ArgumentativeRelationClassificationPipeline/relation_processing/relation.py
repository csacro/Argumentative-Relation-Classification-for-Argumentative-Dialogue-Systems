import logging

import relation_processing.generate_tree.tmg as tmg
import relation_processing.generate_tree.bilp as bilp
import relation_processing.predict_relation.model_bert as bert
import relation_processing.predict_relation.model_sklearnSVM as sklearnSVM


class RelationProcessor:
    def __init__(self, change_model: bool, change_approach: bool):
        self.__change_model = change_model    # default bert
        self.__change_approach = change_approach  # default tmg

    def generate_relation_matrix(self, arguments):
        logging.info("generating relation matrix")

        # probability matrix
        if not self.__change_model:
            logging.info("using BERT")
            prob_matrix = bert.generate_prob_matrix(arguments)
        else:
            logging.info("using SVM")
            prob_matrix = sklearnSVM.generate_prob_matrix(arguments)

        # relation matrix
        if not self.__change_approach:
            logging.info("using Traversing and Modifying Graphs")
            return tmg.make_relation_matrix(prob_matrix)
        else:
            logging.info("using Binary Integer Linear Programming")
            return bilp.make_relation_matrix(prob_matrix)
