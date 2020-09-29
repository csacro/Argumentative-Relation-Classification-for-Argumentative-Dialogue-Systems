import logging

from output_processing.output_helper import OutputHelper
from relation_processing.determine_types import Types


class ReadableFile:

    @staticmethod
    def write_readable(relation_matrix, arguments, filename):
        filename = "output/" + filename
        logging.info("writing readable file")
        relation_row = OutputHelper.relation_matrix_to_row(relation_matrix)
        f = open(filename + ".txt", "w")
        for i in range(1, len(relation_row)):
            arg_from = arguments[i]
            targets, pos_to = Types.calc_argument_relation_aim(i, relation_row, arguments)
            f.write(arg_from.sentence + '\n' + targets + '\n' + arguments[pos_to].sentence + '\n\n')
        f.close()


class AnnotationFile:

    @staticmethod
    def write_annotation(relation_matrix, arguments, filename):
        filename = "output/" + filename
        logging.info("writing annotation file")
        relation_row = OutputHelper.relation_matrix_to_row(relation_matrix)
        an = open(filename + "_annotation.txt", "w")
        for i in range(1, len(relation_row)):
            arg_to = arguments[int(relation_row[i])]
            arg_from = arguments[i]
            # sentence1 is "to" and sentence2 is "from"
            an.write(arg_to.sentence + '\t' + arg_from.sentence + '\n')
        an.close()
