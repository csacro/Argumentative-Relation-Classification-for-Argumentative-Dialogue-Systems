import logging

from datetime import timedelta

# feature
h_sentence_pair = 0
h_single_sentence = 0
h_common_words = 0
h_statistics = 0
h_sentence_pair_time = timedelta(0)
h_single_sentence_time = timedelta(0)
h_common_words_time = timedelta(0)
h_statistics_time = timedelta(0)

h_jaro_winkler = 0
h_jaro = 0
h_edit = 0
h_jaro_winkler_time = timedelta(0)
h_jaro_time = timedelta(0)
h_edit_time = timedelta(0)

# model
h_bert = 0
h_sklearn = 0
h_bert_time = timedelta(0)
h_sklearn_time = timedelta(0)


def log_results():
    if h_sklearn > 0:
        logging.debug("h_sentence_pair_time" + str(h_sentence_pair_time))
        logging.debug("h_sentence_pair" + str(h_sentence_pair))
        logging.info("sentence_pair: " + str(h_sentence_pair_time/h_sentence_pair))

        logging.debug("h_single_sentence_time" + str(h_single_sentence_time))
        logging.debug("h_single_sentence" + str(h_single_sentence))
        logging.info("single_sentence: " + str(h_single_sentence_time/h_single_sentence))

        logging.debug("h_common_words_time" + str(h_common_words_time))
        logging.debug("h_common_words" + str(h_common_words))
        logging.info("common_words: " + str(h_common_words_time/h_common_words))

        logging.debug("h_statistics_time" + str(h_statistics_time))
        logging.debug("h_statistics" + str(h_statistics))
        logging.info("statistics: " + str(h_statistics_time/h_statistics))

        logging.debug("h_edit_time" + str(h_edit_time))
        logging.debug("h_edit" + str(h_edit))
        logging.info("edit: " + str(h_edit_time/h_edit))

        logging.debug("h_jaro_time" + str(h_jaro_time))
        logging.debug("h_jaro" + str(h_jaro))
        logging.info("jaro: " + str(h_jaro_time/h_jaro))

        logging.debug("h_jaro_winkler_time" + str(h_jaro_winkler_time))
        logging.debug("h_jaro_winkler" + str(h_jaro_winkler))
        logging.info("jaro_winkler: " + str(h_jaro_winkler_time/h_jaro_winkler))

        logging.debug("h_sklearn_time" + str(h_sklearn_time))
        logging.debug("h_sklearn" + str(h_sklearn))
        logging.info("sklearn: " + str(h_sklearn_time/h_sklearn))
    else:
        logging.debug("h_bert_time" + str(h_bert_time))
        logging.debug("h_bert" + str(h_bert))
        logging.info("bert: " + str(h_bert_time/h_bert))
