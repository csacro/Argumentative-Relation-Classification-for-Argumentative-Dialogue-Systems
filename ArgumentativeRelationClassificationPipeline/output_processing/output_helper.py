class OutputHelper:

    @staticmethod
    def relation_matrix_to_row(relation_matrix):
        assert (relation_matrix.shape[0] == relation_matrix.shape[1])
        l = relation_matrix.shape[0]
        relation_row = [0] * l
        for rel_from in range(0, l):
            for rel_to in range(0, l):
                if relation_matrix[rel_to][rel_from] != 0:
                    relation_row[rel_from] = rel_to
        return relation_row
