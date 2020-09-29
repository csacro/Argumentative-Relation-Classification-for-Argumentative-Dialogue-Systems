import logging

import numpy as np
import cvxpy


def make_relation_matrix(prob_matrix):
    # Binary Integer Linear Programming
    assert (prob_matrix.shape[0] == prob_matrix.shape[1])
    matrix_shape = prob_matrix.shape
    matrix_nm = matrix_shape[0]
    vector_nm_zeros = np.zeros(matrix_nm, dtype=int)
    vector_nm_zero_ones = np.ones(matrix_nm, dtype=int)
    vector_nm_zero_ones[0] = 0

    relation_matrix = cvxpy.Variable(matrix_shape, boolean=True, name="relation_matrix")
    helper_matrix = cvxpy.Variable(matrix_shape, boolean=True, name="helper_matrix")
    objective = cvxpy.Maximize(cvxpy.sum(cvxpy.multiply(prob_matrix, relation_matrix)))  # (0)
    constraints = [cvxpy.sum(relation_matrix, axis=0) == vector_nm_zero_ones,  # (2a) & (2b)
                   helper_matrix[0] == vector_nm_zero_ones,  # (3.2)
                   cvxpy.diag(relation_matrix) == vector_nm_zeros,  # (1)
                   relation_matrix <= helper_matrix,  # (3.0a)
                   cvxpy.diag(helper_matrix) == vector_nm_zeros  # (3.1)
                   ]
    for i in range(0, matrix_nm):
        for j in range(0, matrix_nm):
            for k in range(0, matrix_nm):
                constraints += [
                    helper_matrix[k][i] - helper_matrix[j][i] - helper_matrix[k][j] >= -1]
                # (3.0b)

    problem = cvxpy.Problem(objective, constraints)
    logging.debug("Problem is " + str(problem))
    problem.solve()

    # output
    logging.debug("Problem status is " + problem.status)
    logging.debug("Problem objective is " + str(problem.value))
    for variable in problem.variables():
        logging.debug("Variable %s: value %s" % (variable.name(), variable.value))

    return problem.variables()[0].value
