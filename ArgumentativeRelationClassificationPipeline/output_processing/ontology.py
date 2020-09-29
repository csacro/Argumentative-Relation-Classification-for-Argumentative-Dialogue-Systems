import logging

from shutil import copyfile
from output_processing.output_helper import OutputHelper
from relation_processing.determine_types import Types


class OwlOntology:

    @staticmethod
    def write_owl(relation_matrix, arguments, filename):
        output_filename = "output/" + filename
        logging.info("writing owl file")
        relation_row = OutputHelper.relation_matrix_to_row(relation_matrix)
        copyfile('output_processing/Definitions.owl', output_filename + '.owl')
        for i in range(0, len(relation_row)):
            OwlOntology.__make_owl_argument(i, relation_row, arguments, filename).write_component(output_filename + '.owl')
        f = open(output_filename + '.owl', 'a')
        f.write('</rdf:RDF>')
        f.close()

    @staticmethod
    def __make_owl_argument(pos_from, relation_row, arguments, database):
        arg_text = arguments[pos_from].sentence
        arg_database = database
        arg_id = pos_from
        arg_type = Types.calc_argument_type(pos_from, relation_row)
        arg_relation, pos_to = Types.calc_argument_relation_aim(pos_from, relation_row, arguments)
        return OwlOntology.Individual(arg_text, arg_database, arg_id, arg_type, arg_relation, 'Argument_' + str(pos_to))

    class Individual:
        """ This class 'Individual' was provided by Niklas Rach """

        def __init__(self, arg_text, arg_database, arg_id, arg_type, arg_relation, arg_aim):
            self.text = arg_text
            self.component = 'Argument_' + str(arg_id)
            self.type = arg_type
            self.relation = arg_relation
            self.aim = arg_aim
            self.name = 'Debate'
            self.url = "http://www.semanticweb.org/schindler/ontologies/" + self.name + '#'
            self.header = "<owl:NamedIndividual rdf:about=\"" + self.url + self.component + "\">"
            self.footer = "</owl:NamedIndividual>"
            self.rdf_type = "<rdf:type rdf:resource=\"" + self.url + self.type + "\"/>"
            self.rdf_relation = "<" + self.name + ":" + self.relation + " rdf:resource=\"" + self.url + self.aim + "\"/>"
            self.rdf_text = "<" + self.name + ":hasText rdf:datatype=\"http://www.w3.org/2001/XMLSchema#string\">" + self.text + "</" + self.name + ":hasText>"
            self.database = "<" + self.name + ":database rdf:datatype=\"http://www.w3.org/2001/XMLSchema#string\">" + arg_database + "</" + self.name + ":database>"

        def write_component(self, file):
            f = open(file, 'a')
            f.write("\n\n\n")
            f.write("    " + self.header + "\n")
            f.write("        " + self.rdf_type + "\n")
            if self.type != 'MajorClaim':
                f.write("        " + self.rdf_relation + "\n")
            f.write("        " + self.rdf_text.encode('ascii', 'ignore').decode("utf8") + "\n")
            f.write("        " + self.database + "\n")
            f.write("    " + self.footer + "\n")
            f.close()
