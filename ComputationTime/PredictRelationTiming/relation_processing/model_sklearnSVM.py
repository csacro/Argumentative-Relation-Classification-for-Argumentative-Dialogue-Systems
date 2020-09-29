import logging

import numpy as np
from joblib import load
from simplerepresentations import RepresentationModel

import nltk
from nltk.metrics.distance import edit_distance, jaro_winkler_similarity, jaro_similarity
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

import Stats
from datetime import datetime


def generate_prob_matrix(arguments):
    representation_model = RepresentationModel(
        model_type='bert',
        model_name='bert-base-uncased',
        batch_size=16,
        max_seq_length=256,
        combination_method='cat',
        last_hidden_to_use=4,
        use_cuda=False
    )

    model = load("relation_processing/model/sklearnSVM/nusvc.joblib")
    num_arguments = len(arguments)
    prob_matrix = np.zeros((num_arguments, num_arguments))
    for rel_from in range(1, num_arguments):
        for rel_to in arguments[rel_from].compare_list:
            if rel_from == rel_to:
                continue
            logging.info("calculating: " + str(rel_from) + "-->" + str(rel_to))

            timer = datetime.now()
            sentence_to, _ = representation_model(text_a=[arguments[rel_to].sentence.lower()])
            sentence_from, _ = representation_model(text_a=[arguments[rel_from].sentence.lower()])
            single_sentences = np.asarray([np.append(sentence_to, sentence_from)])
            Stats.h_single_sentence_time += datetime.now() - timer
            Stats.h_single_sentence += 1

            timer = datetime.now()
            sentence_pair, _ = representation_model(text_a=[arguments[rel_to].sentence.lower()],
                                                    text_b=[arguments[rel_from].sentence.lower()])
            Stats.h_sentence_pair_time += datetime.now() - timer
            Stats.h_sentence_pair += 1

            timer = datetime.now()
            pos_counting = nltk_pos_counting(arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower())
            Stats.h_statistics_time += datetime.now() - timer
            Stats.h_statistics += 1

            timer = datetime.now()
            common_words = nltk_common_words(arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower())
            Stats.h_common_words_time += datetime.now() - timer
            Stats.h_common_words += 1

            timer = datetime.now()
            edit = nltk_editdistance(arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower())
            Stats.h_edit_time += datetime.now() - timer
            Stats.h_edit += 1

            timer = datetime.now()
            jaro = nltk_jarosimilarity(arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower())
            Stats.h_jaro_time += datetime.now() - timer
            Stats.h_jaro += 1

            timer = datetime.now()
            jarowinkler = nltk_jarowinklersimilarity(arguments[rel_to].sentence.lower(), arguments[rel_from].sentence.lower())
            Stats.h_jaro_winkler_time += datetime.now() - timer
            Stats.h_jaro_winkler += 1

            #feature = append_features(sentence_pair, single_sentences, pos_counting, common_words)

            timer = datetime.now()
            rel = model.decision_function(sentence_pair)
            Stats.h_sklearn_time += datetime.now() - timer
            Stats.h_sklearn += 1

            logging.debug(rel)
            prob_matrix[rel_to][rel_from] = rel[0]
    return prob_matrix


def append_features(result, *featurevectors):
    for vector in featurevectors:
        result = np.append(result, vector, axis=1)
    return result


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
    return np.asarray([[result]])


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
    return np.asarray([result])


def nltk_jarowinklersimilarity(parent: str, child: str):
    return np.asarray([[jaro_winkler_similarity(parent, child)]])


def nltk_jarosimilarity(parent: str, child: str):
    return np.asarray([[jaro_similarity(parent, child)]])


def nltk_editdistance(parent: str, child: str):
    return np.asarray([[edit_distance(parent, child)]])
