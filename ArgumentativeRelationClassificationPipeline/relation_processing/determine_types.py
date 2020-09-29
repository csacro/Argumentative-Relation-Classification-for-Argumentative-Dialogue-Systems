class Types:

    @staticmethod
    def calc_argument_type(pos_from, relation_row):
        pos_to = int(relation_row[pos_from])
        if pos_from == 0:
            # Major Claim is argument 0
            return 'MajorClaim'
        else:
            if pos_to != 0:
                # no relation to Major Claim
                return 'Premise'
            if pos_from in relation_row:
                # there exists a relation to this argument
                return 'Claim'
        return 'Premise'

    @staticmethod
    def calc_argument_relation_aim(pos_from, relation_row, arguments):
        if pos_from == 0:
            # Major Claim is argument 0
            return '', ''

        arg_from = arguments[pos_from]
        pos_to = int(relation_row[pos_from])
        arg_to = arguments[pos_to]

        if pos_to == 0:
            relation = 'supports' if arg_from.stance == 'pro' else 'attacks'
        else:
            relation = 'supports' if arg_from.stance == arg_to.stance else 'attacks'

        return relation, pos_to
