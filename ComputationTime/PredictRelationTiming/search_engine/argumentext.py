import logging

from requests import post
import validators


class ArgumenText:
    def __init__(self, user_id: str, api_key: str):
        self.__user_id = user_id
        self.__api_key = api_key

    def query_search_api(self, major_claim: str):
        logging.info("querying search api")
        query = {
            "topic": major_claim,
            "index": "cc",
            "sortBy": "none",
            "numDocs": 20,
            "userID": self.__user_id,
            "apiKey": self.__api_key,
            "strictTopicSearch": False,
            "model": "default",
            "predictStance": True,
            "computeAttention": False,
            "showOnlyArguments": True,
            "removeDuplicates": True,
            "filterNonsensicalEntries": True
        }
        logging.debug(query)
        answer = post('https://api.argumentsearch.com/en/search', json=query)
        data = answer.json()
        logging.debug(data.get("metadata"))
        return data.get("sentences")

    def query_classify_api(self, major_claim, classify_input):
        query = {
            "topic": major_claim,
            "sortBy": "none",
            "userID": self.__user_id,
            "apiKey": self.__api_key,
            "model": "default",
            "predictStance": True,
            "computeAttention": False,
            "showOnlyArguments": True,
            "removeDuplicates": True,
            "filterNonsensicalEntries": True
        }

        if isinstance(classify_input, list):
            logging.info("querying classify api: using sentences")
            query["sentences"] = classify_input
        else:
            try:
                is_url = validators.url(classify_input)
            except validators.ValidationFailure:
                is_url = False
            if is_url:
                logging.info("querying classify api: using url")
                query["targetUrl"] = classify_input
            else:
                logging.info("querying classify api: using text")
                query["text"] = classify_input

        logging.debug(query)
        answer = post('https://api.argumentsearch.com/en/classify', json=query)
        data = answer.json()
        logging.debug(data.get("metadata"))
        return data.get("sentences")

    def query_cluster_api(self, cluster_input, threshold, min_cluster_size):
        logging.info("querying cluster api")
        query = {
            "arguments": cluster_input,
            "threshold": threshold,
            "min_cluster_size": min_cluster_size,
            "model": "SBERT",
            "userID": self.__user_id,
            "apiKey": self.__api_key,
        }
        logging.debug(query)
        answer = post('https://api.argumentsearch.com/en/cluster_arguments', json=query)
        data = answer.json()
        logging.debug(data.get("metadata"))
        return data.get("clusters")
