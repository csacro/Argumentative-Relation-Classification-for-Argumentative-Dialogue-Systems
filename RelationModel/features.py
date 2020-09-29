import util
import logging

import os
from joblib import dump, load

import numpy as np

from simplerepresentations import RepresentationModel
from nltk.metrics.distance import edit_distance, jaro_similarity, jaro_winkler_similarity
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def make_featurevector(data_df, bert_sentence_pair_file: str = None, bert_sentence_file: str = None,
                       vectorgen: int = 0):
    # prepare data for feature generation
    parentchild = data_df.loc[:, ['Parent', 'Child']].to_numpy()
    label = data_df.loc[:, 'Class'].to_numpy()

    # generate features
    sentence_vector = sentence(parentchild, bert_sentence_file)
    sentence_pair_vector = sentence_pair(parentchild, bert_sentence_pair_file)
    vectorgen_mod4 = vectorgen % 4
    if vectorgen_mod4 == 1 or vectorgen_mod4 == 3 or vectorgen == 12 or vectorgen == 14:
        pos_counting_vector = pos_counting(parentchild)
        common_words_vector = common_words(parentchild)
    if vectorgen_mod4 == 2 or vectorgen_mod4 == 3 or vectorgen == 13 or vectorgen == 14:
        editdistance_vector = editdistance(parentchild)
        jarosimilarity_vector = jarosimilarity(parentchild)
        jarowinklersimilarity_vector = jarowinklersimilarity(parentchild)

    # put features into one vector
    featurevector = sentence_pair_vector
    if vectorgen == 0:
        logging.info("features: sentence_pair_vector")
    elif vectorgen == 1:
        featurevector = append_features(sentence_pair_vector, pos_counting_vector, common_words_vector)
        logging.info("features: sentence_pair_vector, pos_counting_vector, common_words_vector")
    elif vectorgen == 2:
        featurevector = append_features(sentence_pair_vector, editdistance_vector, jarosimilarity_vector,
                                        jarowinklersimilarity_vector)
        logging.info("sentence_pair_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 3:
        featurevector = append_features(sentence_pair_vector, pos_counting_vector, common_words_vector,
                                        editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector)
        logging.info(
            "features: sentence_pair_vector, pos_counting_vector, common_words_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 4:
        featurevector = sentence_vector
        logging.info("features: sentence_vector")
    elif vectorgen == 5:
        featurevector = append_features(sentence_vector, pos_counting_vector, common_words_vector)
        logging.info("features: sentence_vector, pos_counting_vector, common_words_vector")
    elif vectorgen == 6:
        featurevector = append_features(sentence_vector, editdistance_vector, jarosimilarity_vector,
                                        jarowinklersimilarity_vector)
        logging.info("sentence_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 7:
        featurevector = append_features(sentence_vector, pos_counting_vector, common_words_vector,
                                        editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector)
        logging.info(
            "features: sentence_vector, pos_counting_vector, common_words_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 8:
        featurevector = append_features(sentence_pair_vector, sentence_vector)
        logging.info("features: sentence_pair_vector, sentence_vector")
    elif vectorgen == 9:
        featurevector = append_features(sentence_pair_vector, sentence_vector, pos_counting_vector, common_words_vector)
        logging.info("features: sentence_pair_vector, sentence_vector, pos_counting_vector, common_words_vector")
    elif vectorgen == 10:
        featurevector = append_features(sentence_pair_vector, sentence_vector, editdistance_vector,
                                        jarosimilarity_vector,
                                        jarowinklersimilarity_vector)
        logging.info(
            "sentence_pair_vector, sentence_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 11:
        featurevector = append_features(sentence_pair_vector, sentence_vector, pos_counting_vector, common_words_vector,
                                        editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector)
        logging.info(
            "features: sentence_pair_vector, sentence_vector, pos_counting_vector, common_words_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 12:
        featurevector = append_features(pos_counting_vector, common_words_vector)
        logging.info(
            "features: pos_counting_vector, common_words_vector")
    elif vectorgen == 13:
        featurevector = append_features(editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector)
        logging.info(
            "features: editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    elif vectorgen == 14:
        featurevector = append_features(pos_counting_vector, common_words_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector)
        logging.info(
            "features: pos_counting_vector, common_words_vector, editdistance_vector, jarosimilarity_vector, jarowinklersimilarity_vector")
    return featurevector, label


def append_features(result, *featurevectors):
    for vector in featurevectors:
        result = np.append(result, vector, axis=1)
    return result


def common_words(parentchild):
    result = []
    for pair in parentchild:
        result.append(nltk_common_words(pair[0], pair[1]))
    return np.asmatrix(result).transpose()


def nltk_common_words(parent: str, child: str):
    parent_token = nltk.word_tokenize(parent)
    child_token = nltk.word_tokenize(child)

    stopwords = nltk.corpus.stopwords.words('english')
    parent_stopworded = [word for word in parent_token if word not in stopwords]
    child_stopworded = [word for word in child_token if word not in stopwords]

    stemmer = nltk.stem.SnowballStemmer('english')
    parent_stemmed = [stemmer.stem(word) for word in parent_stopworded]
    child_stemmed = [stemmer.stem(word) for word in child_stopworded]

    result = 0
    for word in parent_stemmed:
        try:
            child_stemmed.index(word)
            result += 1
        except ValueError:
            pass
    return result


def pos_counting(parentchild):
    result = []
    for pair in parentchild:
        result.append(nltk_pos_counting(pair[0], pair[1]))
    np.reshape(result, (len(parentchild), 4))
    return result


def nltk_pos_counting(parent: str, child: str):
    parent_pos = nltk.pos_tag(nltk.word_tokenize(parent))
    child_pos = nltk.pos_tag(nltk.word_tokenize(child))

    len_diff = len(parent_pos) - len(child_pos)
    punctuation_diff = 0
    numerical_diff = 0
    modal_diff = 0
    for (word, tag) in parent_pos:
        if tag is '.':
            punctuation_diff += 1
        elif tag is 'NUM':
            numerical_diff += 1
        elif tag is 'MOD':
            modal_diff += 1
    for (word, tag) in child_pos:
        if tag is '.':
            punctuation_diff -= 1
        elif tag is 'NUM':
            numerical_diff -= 1
        elif tag is 'MOD':
            modal_diff -= 1

    result = [len_diff, punctuation_diff, numerical_diff, modal_diff]
    return result


def sentence(parentchild, bert_sentence_file: str = None):
    if os.path.exists(bert_sentence_file):
        # sentence pairs are cached
        logging.info("using cached sentence")
        sentencevector = load(bert_sentence_file)
    else:
        open(bert_sentence_file, 'w')
        pairs = len(parentchild)
        sentencevector = bert_sentence(np.reshape(parentchild, (1, 2 * pairs)).tolist()[0])
        sentencevector = np.reshape(sentencevector, (pairs, 2 * len(sentencevector[0])))
        dump(sentencevector, bert_sentence_file)
    return sentencevector


def sentence_pair(parentchild, bert_sentence_pair_file: str = None):
    if os.path.exists(bert_sentence_pair_file):
        # sentence pairs are cached
        logging.info("using cached sentence pairs")
        sentencevector = load(bert_sentence_pair_file)
    else:
        open(bert_sentence_pair_file, 'w')
        sentencevector = bert_sentence(parent=list(parentchild[:, 0]), child=list(parentchild[:, 1]))
        dump(sentencevector, bert_sentence_pair_file)
    return sentencevector


def bert_sentence(parent, child=None):
    representation_model = RepresentationModel(
        model_type='bert',
        model_name='bert-base-uncased',
        batch_size=16,
        max_seq_length=256,
        combination_method='cat',
        last_hidden_to_use=4,
        use_cuda=False
    )
    sentence_representation, _ = representation_model(text_a=parent, text_b=child)
    return sentence_representation


def jarowinklersimilarity(parentchild):
    similarityvector = []
    for pair in parentchild:
        similarityvector.append(nltk_jarowinklersimilarity(pair[0], pair[1]))
    return np.asmatrix(similarityvector).transpose()


def nltk_jarowinklersimilarity(parent: str, child: str):
    return jaro_winkler_similarity(parent, child)


def jarosimilarity(parentchild):
    similarityvector = []
    for pair in parentchild:
        similarityvector.append(nltk_jarosimilarity(pair[0], pair[1]))
    return np.asmatrix(similarityvector).transpose()


def nltk_jarosimilarity(parent: str, child: str):
    return jaro_similarity(parent, child)


def editdistance(parentchild):
    distancevector = []
    for pair in parentchild:
        distancevector.append(nltk_editdistance(pair[0], pair[1]))
    return np.asmatrix(distancevector).transpose()


def nltk_editdistance(parent: str, child: str):
    return edit_distance(parent, child)
