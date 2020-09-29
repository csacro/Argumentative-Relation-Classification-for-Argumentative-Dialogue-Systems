import logging
import time
import util

import argparse

from common.argument import ArgumentList
from search_engine.argumentext import ArgumenText
from relation_processing.relation import RelationProcessor

import Stats


def main():
    # logging
    util.init_logging(logging.INFO, "PredictRelationTiming_" + time.strftime("%Y%m%d-%H%M%S"))

    # command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('MajorClaim', type=str, help='topic of the discussion')
    parser.add_argument('--search', type=int,
                        help='number of sentences to further process from the search results (if not given all sentences retrieved are used)')
    parser.add_argument('--classify', nargs='+', type=str,
                        help='multiple sentences (group sentences with ""), a text or an url to be used as a source to collect arguments from')
    parser.add_argument('-svm', action='store_true', help='change classifier for estimating relation probabilities from BERT to SVM')
    parser.add_argument('-bilp', action='store_true',
                        help='change from generate tree approach tmg (Traversing and Modifying Graphs) to bilp (Binary Linear Integer Programming)')
    parser.add_argument('--cluster', nargs=2, type=float,
                        help='cluster arguments before processing them (relation only possible within cluster) -> first arg: similarity threshold, second arg: min_cluster_size')
    args = parser.parse_args()
    logging.info(args)

    # search engine
    search_engine = ArgumenText("userID", "apiKey")
    if args.classify is None:
        sentences = search_engine.query_search_api(args.MajorClaim)
        if args.search is not None and args.search < len(sentences):
            stance_pro = [a for a in sentences if a["stanceLabel"] == 'pro']
            stance_con = [a for a in sentences if a["stanceLabel"] == 'contra']
            stance_pro.sort(key=lambda s: s["argumentConfidence"]*s["stanceConfidence"], reverse=True)
            stance_con.sort(key=lambda s: s["argumentConfidence"]*s["stanceConfidence"], reverse=True)
            pro_len = min(int(args.search/2), len(stance_pro))
            con_len = min(args.search - pro_len, len(stance_con))
            diff = args.search - pro_len - con_len
            pro_len += diff
            sentences = stance_pro[:pro_len]
            sentences.extend(stance_con[:con_len])
    else:
        if len(args.classify) is 1:
            args.classify = args.classify[0]
        sentences = search_engine.query_classify_api(args.MajorClaim, args.classify)
    arguments = ArgumentList(args.MajorClaim, sentences)

    # clustering
    if args.cluster is not None:
        clusters = search_engine.query_cluster_api([s["sentenceOriginal"] for s in sentences], args.cluster[0],
                                               args.cluster[1])
        logging.debug(clusters)
        arguments.apply_clusters(clusters)

    # relation processing
    relation_processing = RelationProcessor(args.svm, args.bilp)
    relation_processing.generate_relation_matrix(arguments)

    Stats.log_results()

    exit(0)


if __name__ == "__main__":
    main()
